"""Recompute canonical counts for every saved record using the CURRENT
calibrated logic, rather than trusting delta_billed_minus_canonical as
stored -- a lot of saved records predate the EOS/Harmony/Gemma-thinking
fixes. Records that can't be correctly recomputed (missing reasoning_text
or finish_reason on a reasoning-capable model) are excluded and counted,
not silently included with stale numbers.
"""
import glob
import json
import statistics
from collections import defaultdict

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from config import GENERATION, PROVIDER_MODEL_MAP  # noqa: E402


def main():
    groups = defaultdict(list)
    excluded = defaultdict(int)

    for path in sorted(glob.glob("results/*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                provider, model = r["provider"], r["model_short_key"]
                entry = PROVIDER_MODEL_MAP.get(provider, {}).get(model)
                if entry is None:
                    continue  # stale cell no longer in config (e.g. old together/gemma before rename)

                reasoning_format = entry["reasoning_format"]
                has_reasoning_field = "reasoning_text" in r
                has_finish_reason = "finish_reason" in r and r["finish_reason"] is not None

                needs_reasoning = reasoning_format != "simple"
                if needs_reasoning and not has_reasoning_field:
                    excluded[(provider, model, "missing reasoning_text (pre-fix record)")] += 1
                    continue
                if not has_finish_reason:
                    excluded[(provider, model, "missing finish_reason (pre-fix record)")] += 1
                    continue
                if r.get("billed_completion_tokens") is None:
                    excluded[(provider, model, "no billed count (failed request)")] += 1
                    continue

                billed = r["billed_completion_tokens"]
                canonical = canonical_token_count(
                    entry["tokenizer_repo"],
                    r.get("reasoning_text", ""),
                    r["response_text"],
                    finish_reason=effective_finish_reason(
                        billed, r["finish_reason"], GENERATION["max_tokens"]
                    ),
                    reasoning_format=reasoning_format,
                )
                delta = billed - canonical
                groups[(provider, model)].append(delta)

    print("=== Recomputed, current-logic results ===\n")
    print(f"{'provider':<12} {'model':<14} {'n':>4} {'mean':>7} {'median':>7} "
          f"{'min':>5} {'max':>5}  deltas")
    for key, deltas in sorted(groups.items()):
        n = len(deltas)
        mean = statistics.mean(deltas)
        median = statistics.median(deltas)
        print(f"{key[0]:<12} {key[1]:<14} {n:>4} {mean:>7.2f} {median:>7.1f} "
              f"{min(deltas):>5} {max(deltas):>5}  {deltas}")

    if excluded:
        print("\n=== Excluded (cannot recompute correctly from saved data) ===")
        for (provider, model, reason), count in sorted(excluded.items()):
            print(f"  {provider:<12} {model:<14} {count:>3}x  {reason}")


if __name__ == "__main__":
    main()
