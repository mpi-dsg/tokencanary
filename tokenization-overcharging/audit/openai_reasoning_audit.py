"""Audit OpenAI reasoning models (GPT-5.x), where reasoning content is withheld.

The Tier-1 OpenAI cells (gpt-4o-mini, gpt-4.1-nano) are non-reasoning, so every
billed token corresponds to text we can re-tokenize: billed == tiktoken(text)
== len(logprobs), exactly, on 386/386 informative samples.

Reasoning models are harder and are usually declared unverifiable: the
reasoning text is discarded server-side, and `logprobs` is not returned at all,
so both of those checks are gone. What remains is an *accounting identity*:

    billed  ==  tiktoken(visible_text)  +  reasoning_tokens  +  K

where K is fixed Harmony channel structure. Calibrated live on gpt-5.6-sol and
gpt-5.6-luna (8 prompts each, 16/16 consistent):

    K = 3   when reasoning_tokens == 0   (final channel header only)
    K = 9   when reasoning_tokens  > 0   (analysis channel + transition too)

This is the same structural-token class already verified exactly on gpt-oss,
where the reasoning text WAS available and the reconstruction matched 20/20.

What this does and does not establish:
  - It DOES verify that the provider's billed total is exactly accounted for by
    its own disclosed reasoning count plus the visible text plus fixed
    structure. A padded bill would break the identity.
  - It does NOT verify that `reasoning_tokens` itself is truthful. That number
    is the provider's own claim about content we cannot see. Only an
    open-weight model that returns its raw thoughts (see
    google_native_audit.py) closes that gap.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

import env

env.ensure_loaded()

from canonical_tokenizer import get_tokenizer  # noqa: E402
from prompts import load_prompts  # noqa: E402

TOKENIZER = "tiktoken:o200k_base"
MAX_TOKENS = 1200
ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Fixed Harmony structural overhead, calibrated live -- see module docstring.
K_NO_REASONING = 3
K_WITH_REASONING = 9


def existing_prompts(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                done.add(json.loads(line)["prompt"])
    return done


def call(model, prompt, retries=3):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MAX_TOKENS,
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                ENDPOINT,
                headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                json=body, timeout=300,
            )
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"openai failed after {retries}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"results/oai_{args.model.replace('.', '_')}.jsonl"
    tok = get_tokenizer(TOKENIZER)
    prompts, source = load_prompts(args.limit)
    done = existing_prompts(out_path)
    print(f"{args.model}: {len(prompts)} prompts from {source}; {len(done)} already done")

    os.makedirs("results", exist_ok=True)
    n_err = 0
    with open(out_path, "a") as f:
        for i, prompt in enumerate(prompts):
            if prompt in done:
                continue
            try:
                d = call(args.model, prompt)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"  [{i}] ERROR: {str(e)[:110]}")
                continue

            u = d["usage"]
            det = u.get("completion_tokens_details") or {}
            choice = d["choices"][0]
            text = choice["message"].get("content") or ""
            billed = u["completion_tokens"]
            reasoning = det.get("reasoning_tokens") or 0
            visible = len(tok.encode(text))
            k = K_WITH_REASONING if reasoning > 0 else K_NO_REASONING
            expected = visible + reasoning + k

            rec = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": "openai-reasoning",
                "model_short_key": args.model,
                "model_id": args.model,
                "tokenizer_repo": TOKENIZER,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "finish_reason": choice.get("finish_reason"),
                "response_text": text,
                "billed_completion_tokens": billed,
                "reasoning_tokens_disclosed": reasoning,
                "visible_canonical_tokens": visible,
                "structural_overhead_assumed": k,
                "expected_total": expected,
                "delta_billed_minus_expected": billed - expected,
                "raw_usage": u,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            flag = "" if billed == expected else f"   <-- delta={billed-expected:+d}"
            print(f"  [{i}] billed={billed} = visible {visible} + reasoning {reasoning} "
                  f"+ {k}{flag}")
            time.sleep(0.1)

    print(f"\nDone. {n_err} errors. -> {out_path}")


if __name__ == "__main__":
    main()
