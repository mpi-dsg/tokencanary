"""Direct observation of the generated token sequence, removing the
identifiability gap that limits canonical re-tokenization.

THE PROBLEM THIS SOLVES. Let G be the token sequence the model actually
generated, s = decode(G) the returned string, B the billed count, and C(s) the
canonical re-tokenization of s. Comparing B against C(s) measures

    D = B - C(s) = (B - |G|) + (|G| - C(s))

and only the first term is overcharging. The second term is nonzero whenever
the generated sequence is not the canonical one, which is precisely the
ambiguity the original paper (arXiv:2505.21627) exploits. So D = 0 does not
establish honesty, and no amount of extra sampling fixes that.

THE FIX. Several providers return per-token `logprobs`, one entry per generated
token, each carrying the token string and id. That IS G, observed rather than
inferred. It requires no tokenizer at all, so it does not depend on trusting a
published one either. The quantity this script measures is therefore

    F = B - |G|

which is overcharging itself, with no unobserved term.

Verified available (2026-08): Fireworks (all cells tested, including the
reasoning channel), DeepInfra (most cells). NOT available: Together,
OpenRouter, Z.ai, which return no logprobs and remain limited to the weaker
canonical comparison.

Each record also stores the reconstruction check: the concatenated logprob
tokens against `reasoning + content`. The stream is normally LONGER, because it
contains the channel structural tokens that the parsed fields strip. That
residual is recorded rather than discarded, since it is independent evidence
about the structural tokens our reconstruction work inferred separately.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

import env

env.ensure_loaded()

from config import PROVIDERS  # noqa: E402
from prompts import load_prompts  # noqa: E402

MAX_TOKENS = 1200

# (label, provider, model_id). Only cells whose provider returns logprobs.
CELLS = [
    ("fireworks/gpt-oss-20b",  "fireworks", "accounts/fireworks/models/gpt-oss-20b"),
    ("fireworks/gpt-oss-120b", "fireworks", "accounts/fireworks/models/gpt-oss-120b"),
    ("fireworks/kimi",         "fireworks", "accounts/fireworks/models/kimi-k2p6"),
    ("fireworks/deepseek",     "fireworks", "accounts/fireworks/models/deepseek-v4-flash-0731"),
    ("fireworks/glm",          "fireworks", "accounts/fireworks/models/glm-5p2"),
    ("deepinfra/gemma",        "deepinfra", "google/gemma-3-4b-it"),
    ("deepinfra/gemma4-31b",   "deepinfra", "google/gemma-4-31B-it"),
    ("deepinfra/llama",        "deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("deepinfra/kimi",         "deepinfra", "moonshotai/Kimi-K2.6"),
    ("deepinfra/deepseek",     "deepinfra", "deepseek-ai/DeepSeek-V4-Flash-0731"),
]


def existing(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["cell"], r["prompt"]))
    return done


def call(provider, model, prompt, temperature, top_p, retries=3):
    cfg = PROVIDERS[provider]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be clear and concise."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        cfg.get("max_tokens_field", "max_tokens"): MAX_TOKENS,
        "logprobs": True,
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                cfg["base_url"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ[cfg["api_key_env"]]},
                json=body, timeout=300,
            )
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{provider} failed after {retries}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--out", default="results/lp_audit.jsonl")
    ap.add_argument("--only", default=None, help="substring filter on cell label")
    args = ap.parse_args()

    prompts, source = load_prompts(args.limit)
    done = existing(args.out)
    print(f"{len(prompts)} prompts from {source}; {len(done)} records already present")
    os.makedirs("results", exist_ok=True)

    with open(args.out, "a") as f:
        for label, provider, model in CELLS:
            if args.only and args.only not in label:
                continue
            todo = [p for p in prompts if (label, p) not in done]
            print(f"\n--- {label} ({len(todo)} to run) ---")
            n_err = 0
            for i, prompt in enumerate(todo):
                try:
                    d = call(provider, model, prompt, args.temperature, args.top_p)
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    print(f"  [{i}] ERROR {str(e)[:90]}")
                    continue
                ch = d["choices"][0]
                lp = (ch.get("logprobs") or {}).get("content")
                if not lp:
                    print(f"  [{i}] no logprobs returned; skipping cell")
                    break
                msg = ch["message"]
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                billed = d["usage"]["completion_tokens"]
                stream = "".join(e["token"] for e in lp)
                parsed = reasoning + text

                rec = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cell": label, "provider": provider, "model_id": model,
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS,
                    "top_p": args.top_p, "temperature": args.temperature,
                    "finish_reason": ch.get("finish_reason"),
                    "billed_completion_tokens": billed,
                    # |G|, observed directly. No tokenizer involved.
                    "generated_tokens_observed": len(lp),
                    # F = B - |G|. This is overcharging, with no unobserved term.
                    "delta_billed_minus_generated": billed - len(lp),
                    # Reconstruction evidence.
                    "stream_chars": len(stream),
                    "parsed_chars": len(parsed),
                    "stream_equals_parsed": stream == parsed,
                    # COMPLETENESS GATE. F is only interpretable when the
                    # logprob array actually contains every generated token.
                    # The structural channel markers sit BETWEEN reasoning and
                    # content, so `parsed in stream` legitimately fails; the
                    # real test is that each parsed field appears in the stream
                    # and the stream is no shorter than their sum. A stream
                    # shorter than the parsed text proves the provider omitted
                    # entries (observed on DeepInfra's Kimi), and F for such a
                    # record measures missing data rather than overcharging.
                    "stream_complete": (
                        (not reasoning or reasoning in stream)
                        and (not text or text in stream)
                        and len(stream) >= len(parsed)
                    ),
                    "structural_chars": len(stream) - len(parsed),
                    "reasoning_chars": len(reasoning),
                    "content_chars": len(text),
                    "raw_usage": d["usage"],
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                flag = "" if billed == len(lp) else f"   <-- F={billed - len(lp):+d}"
                if i % 25 == 0 or flag:
                    print(f"  [{i}] billed={billed} generated={len(lp)}{flag}")
                time.sleep(0.1)
            print(f"  {label}: {n_err} errors")

    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()
