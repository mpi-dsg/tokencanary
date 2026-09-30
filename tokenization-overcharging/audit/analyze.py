"""Summarize a results .jsonl: per (provider, model) delta distribution."""
import argparse
import json
import statistics
import sys
from collections import defaultdict


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    args = parser.parse_args()

    records = load_records(args.results_path)
    if not records:
        print("No records found.")
        return 1

    groups = defaultdict(list)
    for r in records:
        key = (r["provider"], r["model_short_key"])
        if r["delta_billed_minus_canonical"] is not None:
            groups[key].append(r["delta_billed_minus_canonical"])

    print(f"{'provider':<12} {'model':<8} {'n':>5} {'mean':>7} {'median':>7} "
          f"{'%>0':>6} {'%<0':>6} {'%=0':>6}")
    for (provider, model), deltas in sorted(groups.items()):
        n = len(deltas)
        mean = statistics.mean(deltas)
        median = statistics.median(deltas)
        pct_pos = 100 * sum(1 for d in deltas if d > 0) / n
        pct_neg = 100 * sum(1 for d in deltas if d < 0) / n
        pct_zero = 100 * sum(1 for d in deltas if d == 0) / n
        print(f"{provider:<12} {model:<8} {n:>5} {mean:>7.2f} {median:>7.1f} "
              f"{pct_pos:>5.1f}% {pct_neg:>5.1f}% {pct_zero:>5.1f}%")

    print("\nInterpretation guide:")
    print("- A constant nonzero delta across ~all requests (e.g. always +1) is")
    print("  likely a formatting/special-token artifact, not overcharging --")
    print("  check implementation-notes.md calibration section before concluding fraud.")
    print("- billed > canonical *and growing with response length/complexity* is the")
    print("  actual pattern the original paper's threat model predicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
