"""Calibrate a candidate (provider, model) cell before it is allowed into a
real run.

The rule this project has followed throughout: a nonzero billed-vs-canonical
delta is never treated as signal until its cause is identified. That means for
every new cell we must first establish which reconstruction of the generated
stream reproduces the provider's own completion_tokens on honest traffic.

This script sends a handful of real requests per cell and scores several
candidate reconstructions against the billed count, so the choice of
`reasoning_format` / `append_eos` in config.py is made from data rather than
from the assumption that a format transfers across providers.

Usage:  python calibrate_cells.py [provider ...]
"""
import statistics
import sys

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from probe_new_models import CANDIDATES  # noqa: E402
from prompts import load_prompts  # noqa: E402
from providers import build_client  # noqa: E402

# Reasoning models need headroom to emit BOTH a thought block and a visible
# answer. At the study's default max_tokens=300 these models truncate mid-
# thought almost every time, which never exercises the reasoning->answer
# boundary -- exactly the part of the reconstruction most likely to be wrong.
CALIBRATION_MAX_TOKENS = 1000
N_CALIBRATION = 6

VARIANTS = [
    ("simple/no-eos", "simple", False),
    ("simple/eos", "simple", True),
    ("think_tags/no-eos", "think_tags", False),
    ("think_tags/eos", "think_tags", True),
]


def calibrate(provider, short_key, model_id, tokenizer_repo, prompts):
    client = build_client(provider)
    rows = []
    for prompt in prompts:
        try:
            r = client.chat_completion(
                model=model_id,
                system_prompt="You are a helpful assistant. Be clear and concise.",
                user_prompt=prompt,
                temperature=1.0,
                top_p=0.95,
                max_tokens=CALIBRATION_MAX_TOKENS,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    request failed: {str(e)[:90]}")
            continue
        billed = r["billed_completion_tokens"]
        if billed is None:
            continue
        fr = effective_finish_reason(billed, r["finish_reason"], CALIBRATION_MAX_TOKENS)
        rows.append((billed, r["reasoning_text"], r["text"], fr))

    if not rows:
        print("    NO DATA")
        return None

    n_stop = sum(1 for _, _, _, fr in rows if fr == "stop")
    n_reasoning = sum(1 for _, rs, _, _ in rows if rs)
    print(f"    n={len(rows)}  natural-stop={n_stop}  with-reasoning={n_reasoning}")

    best = None
    for label, fmt, eos in VARIANTS:
        deltas = []
        for billed, reasoning, text, fr in rows:
            canonical = canonical_token_count(
                tokenizer_repo, reasoning, text,
                finish_reason=fr, reasoning_format=fmt, append_eos=eos,
            )
            deltas.append(billed - canonical)
        mean = statistics.mean(deltas)
        n_exact = sum(1 for d in deltas if d == 0)
        # Rank by exact matches first, then by how close the mean sits to zero.
        score = (n_exact, -abs(mean))
        print(f"      {label:<20} mean={mean:>8.2f}  exact={n_exact}/{len(deltas)}  "
              f"deltas={deltas}")
        if best is None or score > best[0]:
            best = (score, label, fmt, eos, mean, n_exact)
    print(f"    -> best: {best[1]}  (exact {best[5]}/{len(rows)}, mean {best[4]:.2f})")
    return best


def main():
    targets = sys.argv[1:] or list(CANDIDATES)
    prompts, source = load_prompts(N_CALIBRATION)
    print(f"prompts: {len(prompts)} from {source}\n")
    summary = {}
    for provider in targets:
        for short_key, (model_id, tokenizer_repo) in CANDIDATES[provider].items():
            print(f"=== {provider} / {short_key} ({model_id}) ===")
            res = calibrate(provider, short_key, model_id, tokenizer_repo, prompts)
            if res:
                summary[(provider, short_key)] = (res[2], res[3], res[4], res[5])
            print()
    print("=" * 78)
    print("RECOMMENDED CONFIG")
    print("=" * 78)
    for (p, k), (fmt, eos, mean, n_exact) in sorted(summary.items()):
        print(f"  {p:<11} {k:<9} reasoning_format={fmt!r:<14} append_eos={eos!s:<6} "
              f"(exact {n_exact}, mean {mean:+.2f})")


if __name__ == "__main__":
    main()
