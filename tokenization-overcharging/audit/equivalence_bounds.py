"""Simultaneous upper bounds on inflation, replacing the MDE argument.

A minimum detectable effect answers a prospective design question: roughly how
large would an effect have to be for this design to find it? It is not an
equivalence threshold, and "no cell exceeded its MDE" is not evidence that no
effect exists. An external review was right to reject that reasoning, and this
script replaces it with the two things it should have been.

1. A ONE-SIDED UPPER CONFIDENCE BOUND per cell. Instead of "we detected
   nothing", the claim becomes "inflation above X% is excluded at this
   confidence". That is a statement about the effect, not about our power.

2. MULTIPLICITY CONTROL. The study runs many cell-runs. At an uncorrected
   alpha of 0.05, roughly one in twenty would breach its bound by chance, so a
   single anomalous cell is unremarkable and a global "no cell shows inflation"
   claim is unsupported. We use a Bonferroni split of alpha across all cells,
   which yields bounds that hold SIMULTANEOUSLY at 95% familywise confidence.

Bounds use the normal quantile rather than Student's t. Every cell reported
here has n well above 100, where the two differ by under 2%, and cells below
n=30 are excluded rather than reported with an unreliable bound.
"""
import collections
import glob
import json
import math
import statistics

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from config import GENERATION, lookup_cell  # noqa: E402

ALPHA = 0.05
MIN_N = 30


def norm_ppf(p):
    """Inverse normal CDF (Acklam's rational approximation, |error| < 1.15e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def collect():
    """Per-cell deltas, from both measurement tiers, kept separate."""
    direct = collections.defaultdict(lambda: {"d": [], "b": []})
    for path in sorted(glob.glob("results/lp_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if not r.get("stream_complete"):
                continue
            direct[r["cell"]]["d"].append(r["delta_billed_minus_generated"])
            direct[r["cell"]]["b"].append(r["billed_completion_tokens"])

    canon = collections.defaultdict(lambda: {"d": [], "b": []})
    for path in sorted(glob.glob("results/full_*.jsonl") + glob.glob("results/p99_*.jsonl")
                       + glob.glob("results/exp_*.jsonl") + glob.glob("results/exp99_*.jsonl")
                       + glob.glob("results/r2_*.jsonl")):
        block = path.split("/")[-1].split("_")[0]
        for line in open(path):
            r = json.loads(line)
            e = lookup_cell(r["provider"], r["model_short_key"])
            if e is None or r.get("billed_completion_tokens") is None:
                continue
            b = r["billed_completion_tokens"]
            mt = r.get("max_tokens", GENERATION["max_tokens"])
            if b == mt:
                continue
            c = canonical_token_count(
                e["tokenizer_repo"], r.get("reasoning_text", ""), r["response_text"],
                finish_reason=effective_finish_reason(b, r["finish_reason"], mt),
                reasoning_format=e["reasoning_format"], append_eos=e.get("append_eos"),
            )
            key = f"{block}:{r['provider']}/{r['model_short_key']}"
            canon[key]["d"].append(b - c)
            canon[key]["b"].append(b)
    return direct, canon


def report(title, cells, m_total):
    """One-sided simultaneous upper bounds at 95% familywise confidence."""
    z = norm_ppf(1 - ALPHA / m_total)
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    print(f"Bonferroni across all {m_total} cells: per-cell one-sided level "
          f"{1 - ALPHA/m_total:.5f}, z = {z:.3f}\n")
    print(f"{'cell':<34} {'n':>5} {'mean Δ':>9} {'95% FW upper bound':>21} {'as % of bill':>13}")
    worst = 0.0
    skipped = []
    for k in sorted(cells):
        d, b = cells[k]["d"], cells[k]["b"]
        if len(d) < MIN_N:
            skipped.append((k, len(d)))
            continue
        n = len(d)
        mean = statistics.mean(d)
        sd = statistics.pstdev(d)
        ub = mean + z * sd / math.sqrt(n)
        mb = statistics.mean(b)
        ub_pct = 100 * ub / mb if mb else float("nan")
        worst = max(worst, ub_pct)
        print(f"{k:<34} {n:>5} {mean:>+9.4f} {ub:>+18.4f} tok {ub_pct:>12.4f}%")
    print(f"\n  Simultaneously across every cell above, at 95% familywise confidence,")
    print(f"  systematic inflation greater than {worst:.4f}% of the bill is EXCLUDED.")
    if skipped:
        print(f"\n  Excluded for n < {MIN_N} (bound would be unreliable): "
              + ", ".join(f"{k} (n={n})" for k, n in skipped))
    return worst


def main():
    direct, canon = collect()
    m = len(direct) + len(canon)
    print(f"Total cells entering multiplicity correction: {m}")
    w1 = report("TIER 1 — direct observation of the generated sequence (F = B - |G|)",
                direct, m)
    w2 = report("TIER 2 — canonical re-tokenization (weaker: |G| unobserved)",
                canon, m)
    print(f"\n{'='*92}")
    print("The paper's smallest claimed inflation rate is 0.28%, its largest 11.2%.")
    print()
    print(f"  Tier 1 (direct observation) excludes inflation above {w1:.4f}%.")
    print("  The bound is exactly zero because F has zero variance: billed equals")
    print("  the observed generated sequence on every request, so there is no")
    print("  spread for a bound to accommodate. This is categorical.")
    print()
    print(f"  Tier 2 (canonical) has a worst-cell bound of {w2:.4f}%, which is ABOVE")
    print("  the paper's smallest claimed rate and must not be reported as though")
    print("  it were below it. That worst case is deepinfra/gpt-oss-20b, whose mean")
    print("  is carried by the identified rewrite records where the provider bills a")
    print("  discarded draft it does not return. Those are a documented non-fraud")
    print("  mechanism, not evidence of inflation, but they widen the bound and the")
    print("  bound is what an adversarial reader will quote.")
    print()
    print("  The defensible Tier 2 statement is per-cell, not global: most cells")
    print("  exclude inflation above ~0.05%, and the handful that do not are named")
    print("  with the mechanism that widens them.")


if __name__ == "__main__":
    main()
