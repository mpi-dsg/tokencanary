"""Tier-1 audit of Google's OWN first-party API, including hidden reasoning.

Why this cell is the strongest single test in the study:

Every auditing framework in this literature treats hidden reasoning tokens as
unverifiable -- the provider bills for chain-of-thought the customer never
sees, so there is nothing to recompute against. That is true for
o-series/GPT-5 reasoning (content discarded server-side) and for Anthropic.

It is NOT true here. Google serves `gemma-4-31b-it` -- an open-weight model
with a public tokenizer -- on its own AI Studio API, and the native
`generateContent` endpoint returns BOTH the raw thought text (parts with
`thought: true`) AND a separate `thoughtsTokenCount`. So the hidden-reasoning
billing channel becomes directly checkable: re-tokenize the thought text with
the public tokenizer and compare it to the count the provider billed.

Verified on a first pass that the returned thoughts are raw, not summarized
(6/6 exact matches), which is what makes the comparison meaningful -- a
summary would undercount and produce a spurious "overcharge".

Writes results/gnative_google.jsonl with both channels scored separately.
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

MODEL = "gemma-4-31b-it"
TOKENIZER_REPO = "google/gemma-4-31B-it"
MAX_OUTPUT_TOKENS = 1200
OUT_PATH = "results/gnative_google.jsonl"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def existing_prompts(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                done.add(json.loads(line)["prompt"])
    return done


def call(prompt, temperature, top_p, retries=3):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": temperature,
            "topP": top_p,
        },
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                ENDPOINT, params={"key": os.environ["GOOGLE_API_KEY"]},
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
    raise RuntimeError(f"google native failed after {retries}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    tok = get_tokenizer(TOKENIZER_REPO)
    prompts, source = load_prompts(args.limit)
    done = existing_prompts(OUT_PATH)
    print(f"{len(prompts)} prompts from {source}; {len(done)} already done")

    os.makedirs("results", exist_ok=True)
    n_err = 0
    with open(OUT_PATH, "a") as f:
        for i, prompt in enumerate(prompts):
            if prompt in done:
                continue
            try:
                d = call(prompt, args.temperature, args.top_p)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"  [{i}] ERROR: {str(e)[:110]}")
                continue

            u = d.get("usageMetadata", {})
            cand = d.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", []) or []
            thought = "".join(p.get("text", "") for p in parts if p.get("thought"))
            answer = "".join(p.get("text", "") for p in parts if not p.get("thought"))

            canon_thought = len(tok.encode(thought, add_special_tokens=False))
            canon_answer = len(tok.encode(answer, add_special_tokens=False))
            billed_thought = u.get("thoughtsTokenCount", 0) or 0
            billed_answer = u.get("candidatesTokenCount", 0) or 0
            total = u.get("totalTokenCount", 0) or 0
            prompt_tokens = u.get("promptTokenCount", 0) or 0

            rec = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": "google-native",
                "model_short_key": "gemma",
                "model_id": MODEL,
                "tokenizer_repo": TOKENIZER_REPO,
                "prompt": prompt,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "finish_reason": cand.get("finishReason"),
                "thought_text": thought,
                "response_text": answer,
                "billed_thought_tokens": billed_thought,
                "canonical_thought_tokens": canon_thought,
                "delta_thought": billed_thought - canon_thought,
                "billed_answer_tokens": billed_answer,
                "canonical_answer_tokens": canon_answer,
                "delta_answer": billed_answer - canon_answer,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total,
                # Does the provider's own arithmetic close?
                "total_reconciles": (prompt_tokens + billed_answer + billed_thought) == total,
                "raw_usage": u,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            flag = ""
            if rec["delta_thought"] or rec["delta_answer"]:
                flag = f"  <-- thought{rec['delta_thought']:+d} answer{rec['delta_answer']:+d}"
            print(f"  [{i}] thought {billed_thought}/{canon_thought} "
                  f"answer {billed_answer}/{canon_answer}{flag}")
            time.sleep(0.1)

    print(f"\nDone. {n_err} errors. -> {OUT_PATH}")


if __name__ == "__main__":
    main()
