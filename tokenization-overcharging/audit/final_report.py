"""Final consolidated report: p=0.95 (main study) and p=0.99 (their most
extreme, most favorable-to-fraud setting) reported separately, each with
delta stats, minimum detectable effect, and shape diagnostics.
"""
import glob
import json
import math
import statistics
from collections import Counter, defaultdict

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from config import GENERATION, lookup_cell  # noqa: E402


def recompute(glob_pattern):
    """Recompute every saved record with current calibrated logic.

    Returns per-cell deltas split into ALL samples and the INFORMATIVE subset.
    A response billed at exactly its max_tokens cap was truncated, so its
    billed count is pinned to the cap and cannot exceed it no matter what the
    provider does -- such a sample carries no evidence about inflation in
    either direction, and pooling it silently inflates the apparent n.
    """
    groups = defaultdict(list)
    informative = defaultdict(list)
    billed_by_cell = defaultdict(list)
    billed_informative = defaultdict(list)
    for path in sorted(glob.glob(glob_pattern)):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                provider, model = r["provider"], r["model_short_key"]
                entry = lookup_cell(provider, model)
                if entry is None or "reasoning_text" not in r or r.get("finish_reason") is None:
                    continue
                if r.get("billed_completion_tokens") is None:
                    continue
                billed = r["billed_completion_tokens"]
                # Older records predate per-record max_tokens; they all ran at
                # the study default, so fall back to it rather than guess.
                max_tokens = r.get("max_tokens", GENERATION["max_tokens"])
                canonical = canonical_token_count(
                    entry["tokenizer_repo"], r.get("reasoning_text", ""),
                    r["response_text"],
                    finish_reason=effective_finish_reason(billed, r["finish_reason"], max_tokens),
                    reasoning_format=entry["reasoning_format"],
                    append_eos=entry.get("append_eos"),
                )
                delta = billed - canonical
                key = (provider, model)
                groups[key].append(delta)
                billed_by_cell[key].append(billed)
                # Uninformative iff the bill is pinned EXACTLY at the cap
                # (truncated, so it cannot exceed it whatever the provider
                # does). A response billed ABOVE the cap is not truncated --
                # the provider simply did not enforce max_tokens -- so it is
                # informative and must not be silently dropped.
                if billed != max_tokens:
                    informative[key].append(delta)
                    billed_informative[key].append(billed)
    return groups, billed_by_cell, informative, billed_informative


def mde(deltas, billed_tokens):
    n = len(deltas)
    sd = statistics.pstdev(deltas) or 0.01
    z = 1.645 + 0.84
    mde_tokens = z * sd / math.sqrt(n)
    mean_billed = statistics.mean(billed_tokens)
    return mde_tokens, 100 * mde_tokens / mean_billed if mean_billed else None


def report(label, glob_pattern):
    groups, billed_by_cell, informative, billed_informative = recompute(glob_pattern)
    print(f"\n{'='*104}\n{label}\n{'='*104}")
    print("n_inf = samples not truncated at max_tokens (a capped response cannot "
          "show inflation); stats below are on that subset.")
    print(f"{'provider':<12} {'model':<14} {'n':>4} {'n_inf':>6} {'mean':>7} "
          f"{'%pos':>6} {'%neg':>6} {'%zero':>6} {'MDE%':>7}")
    total_n = 0
    total_inf = 0
    detectable = []
    for key in sorted(groups):
        n = len(groups[key])
        total_n += n
        deltas = informative[key]
        total_inf += len(deltas)
        if not deltas:
            print(f"{key[0]:<12} {key[1]:<14} {n:>4} {0:>6}   -- all samples truncated "
                  f"at max_tokens; no evidence either way")
            continue
        mean = statistics.mean(deltas)
        pct_pos = 100 * sum(1 for d in deltas if d > 0) / len(deltas)
        pct_neg = 100 * sum(1 for d in deltas if d < 0) / len(deltas)
        pct_zero = 100 * sum(1 for d in deltas if d == 0) / len(deltas)
        mde_tokens, mde_pct = mde(deltas, billed_informative[key])
        # A positive mean only means something if it clears this cell's own
        # minimum detectable effect. Reporting a bare `mean > 0` boolean
        # overstates single-sample noise as signal: e.g. one +1 delta among
        # 200 requests yields mean=+0.005 with MDE=0.01 tokens, i.e. an
        # effect an order of magnitude below what the sample can resolve.
        flag = " <-- ABOVE MDE" if mean > mde_tokens else ""
        if mean > mde_tokens:
            detectable.append((key, mean, mde_tokens))
        print(f"{key[0]:<12} {key[1]:<14} {n:>4} {len(deltas):>6} {mean:>7.3f} "
              f"{pct_pos:>5.1f}% {pct_neg:>5.1f}% {pct_zero:>5.1f}% {mde_pct:>6.2f}%{flag}")
    print(f"\nTotal n={total_n} requests, of which {total_inf} informative "
          f"({100*total_inf/total_n:.0f}%).")
    if detectable:
        print("Cells with a mean overcharge exceeding their own minimum detectable effect:")
        for key, mean, mde_tokens in detectable:
            print(f"  {key[0]}/{key[1]}: mean=+{mean:.3f} tokens vs MDE={mde_tokens:.3f}")
    else:
        print("No cell shows a mean overcharge exceeding its own minimum detectable "
              "effect. Every positive mean present is below the noise floor of its "
              "own sample (i.e. not a detectable effect, not evidence of inflation).")
    return groups


if __name__ == "__main__":
    p95 = report("p=0.95 (main study, matches original paper's headline setting)", "results/full_*.jsonl")
    p99 = report("p=0.99 (original paper's MOST extreme, highest-inflation-claimed setting)", "results/p99_*.jsonl")
    # Reported separately, not pooled: these cells run a different model set
    # (DeepSeek / Kimi / GLM) at a higher max_tokens, so their samples are not
    # drawn from the same experimental condition as the two blocks above.
    if glob.glob("results/exp_*.jsonl"):
        exp = report(
            "EXPANSION: DeepSeek / Kimi / GLM / GPT across resellers and "
            "first-party APIs (p=0.95, max_tokens=1200)",
            "results/exp_*.jsonl",
        )
    if glob.glob("results/exp99_*.jsonl"):
        exp99 = report(
            "EXPANSION at p=0.99 (the original paper's most "
            "favorable-to-fraud sampling setting)",
            "results/exp99_*.jsonl",
        )
    if glob.glob("results/r2_*.jsonl"):
        r2 = report(
            "ROUND 2: gpt-oss on all four resellers, gemma-4-31B matched to "
            "Together, GLM-5.2 (current) replacing 4.6",
            "results/r2_*.jsonl",
        )
    if glob.glob("results/gnative_*.jsonl"):
        print(f"\n{'='*104}")
        print("GOOGLE NATIVE: hidden reasoning-token billing, verified against a "
              "public tokenizer")
        print("=" * 104)
        import statistics as _st
        recs = [json.loads(l) for p in glob.glob("results/gnative_*.jsonl")
                for l in open(p)]
        for label, dk, bk in [("thought (hidden reasoning)", "delta_thought", "billed_thought_tokens"),
                              ("answer  (visible output)", "delta_answer", "billed_answer_tokens")]:
            ds = [r[dk] for r in recs]
            print(f"  {label:<28} n={len(ds)}  exact={sum(1 for d in ds if d == 0)}/{len(ds)}  "
                  f"mean={_st.mean(ds):+.4f}  billed_tokens={sum(r[bk] for r in recs):,}")
        rec_ok = sum(1 for r in recs if r.get("total_reconciles"))
        th = sum(r["billed_thought_tokens"] for r in recs)
        an = sum(r["billed_answer_tokens"] for r in recs)
        print(f"  provider arithmetic reconciles: {rec_ok}/{len(recs)}")
        print(f"  hidden-reasoning share of billed output: {100*th/(th+an):.1f}%")
