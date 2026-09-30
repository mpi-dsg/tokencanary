"""Positive control: would this method have detected inflation if it happened?

A null result without a positive control is weak, and an external review
flagged its absence as a blocker. Two questions have to be answered separately.

1. WAS INFLATION AVAILABLE on our actual commercial outputs? The attack
   substitutes a longer but still valid tokenization of the same string, so its
   surface is set by how much longer a valid segmentation can be. We construct
   one directly and report the ratio as a strict lower bound. If the surface
   were near zero the attack would be inapplicable to real traffic, and a null
   finding would say nothing.

2. WOULD WE HAVE SEEN IT? We inject synthetic inflation into the recorded bills
   at a range of effect sizes, re-run the detector unchanged, and report the
   detection rate per cell. This converts "no effect exceeded its MDE" into the
   defensible form: an effect of size X would have been caught in Y% of
   replications.

Effect sizes are anchored on the original paper: 0.28% is its smallest reported
rate and 11.2% its largest.
"""
import argparse
import collections
import glob
import json
import math
import random
import statistics

import env

env.ensure_loaded()

from canonical_tokenizer import get_tokenizer  # noqa: E402

EFFECTS = [0.0005, 0.001, 0.0028, 0.01, 0.05, 0.112]
N_BOOT = 400
Z = 1.645  # one-sided alpha = 0.05


def headroom(records, tokenizer_repo, sample=25):
    """Attack surface on real outputs: how much longer can a VALID tokenization
    of the same string be?

    Rather than enumerate the vocabulary, which lives in a normalized string
    space that differs per tokenizer and will not match raw text, we construct
    an alternative segmentation directly: encode the response one character at
    a time and concatenate. That is a genuine, valid tokenization of exactly
    the same string, so its length is a strict lower bound on the true maximum.
    Reporting a lower bound is the conservative direction for this argument.

    This measures the surface BEFORE the paper's plausibility filter. The
    fraction of it that survives top-p plausibility is what their own 0.28% to
    11.2% figures quantify, and it needs model logits we do not have.
    """
    tok = get_tokenizer(tokenizer_repo)
    ratios = []
    for r in random.sample(records, min(sample, len(records))):
        text = r.get("response_text") or ""
        if len(text) < 40 or len(text) > 800:
            continue
        try:
            canon = len(tok.encode(text, add_special_tokens=False))
            if canon <= 0:
                continue
            alt = sum(max(1, len(tok.encode(ch, add_special_tokens=False)))
                      for ch in text)
        except Exception:  # noqa: BLE001
            continue
        ratios.append(alt / canon - 1.0)
    return statistics.median(ratios) if ratios else None


def detection_rate(deltas, billed, effect):
    """Inject `effect` proportional inflation and re-run the detector.

    The detector is the one used throughout: a one-sided test that the mean
    delta exceeds zero. We bootstrap the cell to get a detection probability
    rather than a single yes/no.
    """
    n = len(deltas)
    if n < 5:
        return None
    hits = 0
    idx = range(n)
    for _ in range(N_BOOT):
        pick = [random.choice(idx) for _ in range(n)]
        # inflated bill adds effect * billed tokens to the observed delta
        vals = [deltas[i] + effect * billed[i] for i in pick]
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1e-9
        if m / (sd / math.sqrt(n)) > Z:
            hits += 1
    return hits / N_BOOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom-sample", type=int, default=15)
    args = ap.parse_args()
    random.seed(0)

    cells = collections.defaultdict(list)
    for path in sorted(glob.glob("results/lp_audit_p95.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("stream_complete"):
                cells[r["cell"]].append(r)

    print("=" * 88)
    print("POSITIVE CONTROL 1: would injected inflation have been detected?")
    print("=" * 88)
    print("Detection probability by injected effect size, using the directly")
    print("observed generated-token counts (F = billed - |G|).\n")
    hdr = "  ".join(f"{e*100:>6.2f}%" for e in EFFECTS)
    print(f"{'cell':<24} {'n':>4}  {hdr}")
    for c in sorted(cells):
        rs = cells[c]
        deltas = [r["delta_billed_minus_generated"] for r in rs]
        billed = [r["billed_completion_tokens"] for r in rs]
        row = []
        for e in EFFECTS:
            p = detection_rate(deltas, billed, e)
            row.append("  n/a " if p is None else f"{100*p:>6.1f}")
        print(f"{c:<24} {len(rs):>4}  " + "  ".join(row))

    print()
    print("=" * 88)
    print("POSITIVE CONTROL 2: was inflation available on these outputs?")
    print("=" * 88)
    print("Median attack surface: a constructed alternative tokenization of the")
    print("SAME returned string, giving a strict lower bound on the maximum.\n")
    # The logprob records store character counts rather than the text itself,
    # so headroom is measured on the main-study files, which carry
    # `response_text` for the same models.
    from config import lookup_cell  # noqa: PLC0415
    text_cells = collections.defaultdict(list)
    for path in sorted(glob.glob("results/exp_*.jsonl") + glob.glob("results/r2_*.jsonl")
                       + glob.glob("results/full_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("response_text"):
                text_cells[(r["provider"], r["model_short_key"])].append(r)

    for key in sorted(text_cells):
        entry = lookup_cell(*key)
        if entry is None:
            continue
        h = headroom(text_cells[key], entry["tokenizer_repo"], sample=args.headroom_sample)
        label = f"{key[0]}/{key[1]}"
        if h is None:
            print(f"  {label:<24} no usable samples")
        else:
            print(f"  {label:<24} median headroom = +{100*h:>6.0f}%")
    print()
    print("  For comparison, the paper's own claimed inflation range is")
    print("  +0.28% to +11.2%, which is the fraction of this surface that")
    print("  survives its top-p plausibility filter.")


if __name__ == "__main__":
    main()
