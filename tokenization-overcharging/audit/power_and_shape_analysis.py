"""Two things a hostile reader will ask for:
1. Minimum detectable effect size -- "you found nothing" is weak; "we could
   detect systematic inflation as small as X% with 95% confidence and did
   not" is a real claim.
2. Distribution shape -- if a provider ran Algorithm-2-style selective
   padding on a subset of responses, deltas should be bimodal (an honest
   cluster near the calibrated baseline + a padded cluster elsewhere), not
   a single tight noise distribution.
"""
import glob
import json
import math
import statistics
from collections import defaultdict

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from config import GENERATION, PROVIDER_MODEL_MAP  # noqa: E402


def recompute_all():
    groups = defaultdict(list)
    billed_by_cell = defaultdict(list)
    for path in sorted(glob.glob("results/full_*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                provider, model = r["provider"], r["model_short_key"]
                entry = PROVIDER_MODEL_MAP.get(provider, {}).get(model)
                if entry is None or "reasoning_text" not in r or r.get("finish_reason") is None:
                    continue
                if r.get("billed_completion_tokens") is None:
                    continue
                billed = r["billed_completion_tokens"]
                canonical = canonical_token_count(
                    entry["tokenizer_repo"], r.get("reasoning_text", ""),
                    r["response_text"],
                    finish_reason=effective_finish_reason(
                        billed, r["finish_reason"], GENERATION["max_tokens"]
                    ),
                    reasoning_format=entry["reasoning_format"],
                )
                delta = r["billed_completion_tokens"] - canonical
                groups[(provider, model)].append(delta)
                billed_by_cell[(provider, model)].append(r["billed_completion_tokens"])
    return groups, billed_by_cell


def min_detectable_effect(deltas, billed_tokens, alpha=0.05, power=0.80):
    """Rough one-sided minimum detectable mean-shift, in tokens and as a %
    of typical response length, at the given alpha/power, using a normal
    approximation (adequate for n=200 by CLT even though per-sample deltas
    are discrete/skewed)."""
    n = len(deltas)
    if n < 10:
        return None
    sd = statistics.pstdev(deltas) or 0.5  # floor so a zero-variance cell doesn't divide by zero
    # z_alpha + z_beta for one-sided alpha=0.05, power=0.80
    z = 1.645 + 0.84
    mde_tokens = z * sd / math.sqrt(n)
    mean_billed = statistics.mean(billed_tokens)
    mde_pct = 100 * mde_tokens / mean_billed if mean_billed else None
    return mde_tokens, mde_pct, sd, mean_billed


def dip_test_proxy(deltas):
    """Not a formal Hartigan dip test (no scipy/diptest dependency here) --
    a simple, honest proxy: compare the tightness of the empirical
    distribution against what a single honest cluster should look like.
    Report the histogram directly so shape is visually inspectable, plus
    a basic bimodality coefficient (Sarle's), which is a real, citable
    statistic: BC = (skew^2 + 1) / kurtosis(excess)+3, adjusted for n.
    BC > 0.555 (for large n) is the traditional threshold suggesting
    possible bimodality; well below it is evidence against a two-cluster
    (honest + padded) mixture.
    """
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.pstdev(deltas)
    if sd == 0:
        return {"bimodality_coefficient": None, "note": "zero variance -- single point mass, definitionally not bimodal"}
    m3 = sum((d - mean) ** 3 for d in deltas) / n
    m4 = sum((d - mean) ** 4 for d in deltas) / n
    skew = m3 / (sd ** 3)
    kurtosis_excess = m4 / (sd ** 4) - 3
    bc = (skew ** 2 + 1) / (kurtosis_excess + 3 + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))) if n > 3 else None
    return {"skew": skew, "kurtosis_excess": kurtosis_excess, "bimodality_coefficient": bc}


def histogram(deltas, width=40):
    from collections import Counter
    c = Counter(deltas)
    lo, hi = min(deltas), max(deltas)
    lines = []
    maxcount = max(c.values())
    for v in range(lo, hi + 1):
        n = c.get(v, 0)
        bar = "#" * max(1, round(width * n / maxcount)) if n else ""
        lines.append(f"    {v:>4}: {n:>4}  {bar}")
    return "\n".join(lines)


def main():
    groups, billed_by_cell = recompute_all()
    print("=== Minimum detectable effect (one-sided, alpha=0.05, power=0.80) ===\n")
    print(f"{'provider':<12} {'model':<14} {'n':>4} {'mean_delta':>10} {'sd':>6} {'MDE(tokens)':>12} {'MDE(%)':>8}")
    for key in sorted(groups):
        deltas = groups[key]
        billed = billed_by_cell[key]
        n = len(deltas)
        mean_delta = statistics.mean(deltas)
        result = min_detectable_effect(deltas, billed)
        if result is None:
            continue
        mde_tokens, mde_pct, sd, mean_billed = result
        print(f"{key[0]:<12} {key[1]:<14} {n:>4} {mean_delta:>10.3f} {sd:>6.2f} "
              f"{mde_tokens:>12.3f} {mde_pct:>7.2f}%")

    print("\n=== Distribution shape (bimodality check) ===")
    for key in sorted(groups):
        deltas = groups[key]
        if len(deltas) < 10:
            continue
        shape = dip_test_proxy(deltas)
        print(f"\n--- {key[0]} / {key[1]} (n={len(deltas)}) ---")
        print(f"  {shape}")
        print(histogram(deltas))


if __name__ == "__main__":
    main()
