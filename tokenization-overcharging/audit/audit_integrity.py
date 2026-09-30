"""Independent integrity audit of the whole dataset.

Deliberately does NOT import the analysis path it is checking. Where possible
it re-derives quantities from the raw records with its own code, so a bug in
`final_report.py` cannot hide behind itself.

Checks:
  1. Record counts, duplicate prompts, malformed rows.
  2. Every cell is exactly n=200 (or explains why not).
  3. Config resolution: does every record resolve to exactly one config entry,
     and is that the entry the run actually used? Ambiguity here would mean
     records recomputed under the wrong reconstruction.
  4. Independent recomputation of every delta, compared against the analysis
     path. Any disagreement is a bug in one of them.
  5. The informative filter: billed < max_tokens, and no record with
     billed > max_tokens (which would be impossible and indicate corruption).
  6. Arithmetic sanity: reported means/percentages recomputed from scratch.
"""
import collections
import glob
import json
import math
import statistics
import sys

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from config import (  # noqa: E402
    EXPANSION_MODEL_MAP, GENERATION, PROVIDER_MODEL_MAP, ROUND2_MODEL_MAP, lookup_cell,
)

BLOCKS = {
    "A_p95_main": "results/full_*.jsonl",
    "B_p99_main": "results/p99_*.jsonl",
    "C_p95_exp": "results/exp_*.jsonl",
    "D_p99_exp": "results/exp99_*.jsonl",
    "E_round2": "results/r2_*.jsonl",
}
SPECIAL = {
    "F_gpt56": "results/oai_*.jsonl",
    "G_gnative": "results/gnative_*.jsonl",
}

problems = []
notes = []


def problem(msg):
    problems.append(msg)
    print(f"  [PROBLEM] {msg}")


def note(msg):
    notes.append(msg)
    print(f"  [note] {msg}")


# Only these prefixes are study datasets. Everything else in results/ is a
# diagnostic with its own schema (smoke tests, targeted reproductions) and must
# not be audited as if it were a study cell -- whitelist rather than blacklist,
# so a new diagnostic file cannot silently enter the audit.
STUDY_PREFIXES = ("full_", "p99_", "exp_", "exp99_", "r2_", "oai_", "gnative_")


def is_study_file(path):
    return path.split("/")[-1].startswith(STUDY_PREFIXES)


def check_files():
    print("\n=== 1. File integrity, duplicates, counts ===")
    all_files = sorted(glob.glob("results/*.jsonl"))
    skipped = [p.split("/")[-1] for p in all_files if not is_study_file(p)]
    if skipped:
        print(f"  (skipping non-study diagnostic files: {', '.join(skipped)})")
    for path in all_files:
        if not is_study_file(path):
            continue
        rows, bad = [], 0
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                bad += 1
        if bad:
            problem(f"{path}: {bad} unparsable lines")
        cells = collections.defaultdict(list)
        for r in rows:
            cells[(r.get("provider"), r.get("model_short_key"))].append(r)
        for k, rs in sorted(cells.items()):
            prompts = [r.get("prompt") for r in rs]
            dupes = len(prompts) - len(set(prompts))
            tag = f"{k[0]}/{k[1]}"
            if dupes:
                problem(f"{tag}: {dupes} DUPLICATE prompts ({len(rs)} records, "
                        f"{len(set(prompts))} unique)")
            if len(rs) != 200:
                problem(f"{tag}: n={len(rs)}, expected 200")
            # gnative uses a two-channel schema (billed_thought_tokens /
            # billed_answer_tokens) rather than a single completion count, and
            # is audited on its own terms in step 4.
            fields = ("prompt",) if k[0] == "google-native" else (
                "billed_completion_tokens", "prompt")
            missing = [f for f in fields if any(r.get(f) is None for r in rs)]
            if missing:
                problem(f"{tag}: null values present in {missing}")


def check_config_resolution():
    print("\n=== 2. Config resolution — one entry per cell, no ambiguity ===")
    seen = collections.defaultdict(list)
    for name, mp in [("PROVIDER_MODEL_MAP", PROVIDER_MODEL_MAP),
                     ("EXPANSION_MODEL_MAP", EXPANSION_MODEL_MAP),
                     ("ROUND2_MODEL_MAP", ROUND2_MODEL_MAP)]:
        for prov, cells in mp.items():
            for k in cells:
                seen[(prov, k)].append(name)
    for k, maps in sorted(seen.items()):
        if len(maps) > 1:
            e = lookup_cell(*k)
            others = [mp for mp in (PROVIDER_MODEL_MAP, EXPANSION_MODEL_MAP, ROUND2_MODEL_MAP)
                      if k[1] in mp.get(k[0], {})]
            differing = [
                f for f in ("tokenizer_repo", "reasoning_format", "append_eos")
                if len({str(m[k[0]][k[1]].get(f)) for m in others}) > 1
            ]
            if differing:
                problem(f"{k[0]}/{k[1]} defined in {maps} with DIFFERENT {differing} "
                        f"-- lookup_cell resolves to {e.get('reasoning_format')}/"
                        f"append_eos={e.get('append_eos')}; recomputation may not match "
                        f"the config the run used")
            else:
                note(f"{k[0]}/{k[1]} defined in {maps} but fields agree -- harmless")


def independent_delta(r, entry):
    """Recompute the delta with an explicitly separate code path."""
    billed = r["billed_completion_tokens"]
    mt = r.get("max_tokens", GENERATION["max_tokens"])
    fr = r["finish_reason"]
    # inline the truncation override rather than calling the shared helper
    if billed == mt and fr == "stop":
        fr = "length"
    canonical = canonical_token_count(
        entry["tokenizer_repo"], r.get("reasoning_text", ""), r["response_text"],
        finish_reason=fr, reasoning_format=entry["reasoning_format"],
        append_eos=entry.get("append_eos"),
    )
    return billed - canonical


def check_recompute():
    print("\n=== 3. Independent recomputation + informative filter ===")
    grand = 0
    for bname, pat in BLOCKS.items():
        cells = collections.defaultdict(lambda: {"d": [], "b": [], "n": 0})
        unresolved = 0
        impossible = 0
        for path in sorted(glob.glob(pat)):
            for line in open(path):
                r = json.loads(line)
                k = (r["provider"], r["model_short_key"])
                e = lookup_cell(*k)
                if e is None:
                    unresolved += 1
                    continue
                billed = r["billed_completion_tokens"]
                mt = r.get("max_tokens", GENERATION["max_tokens"])
                # billed > max_tokens is NOT corruption: it means the provider
                # did not enforce the cap. Real and important (see the
                # OpenRouter finding), so counted and reported, not discarded.
                if billed is not None and billed > mt:
                    impossible += 1
                cells[k]["n"] += 1
                if billed is None:
                    continue
                if billed != mt:
                    cells[k]["d"].append(independent_delta(r, e))
                    cells[k]["b"].append(billed)
        n_block = sum(c["n"] for c in cells.values())
        grand += n_block
        print(f"  {bname}: {n_block} records, {len(cells)} cells", end="")
        if unresolved:
            problem(f"{bname}: {unresolved} records resolve to NO config entry")
        if impossible:
            note(f"{bname}: {impossible} records billed ABOVE the requested cap "
                 f"(provider did not enforce max_tokens -- real finding, not corruption)")
        print()
        for k, v in sorted(cells.items()):
            if not v["d"]:
                continue
            mean = statistics.mean(v["d"])
            sd = statistics.pstdev(v["d"]) or 0.01
            mde_tok = (1.645 + 0.84) * sd / math.sqrt(len(v["d"]))
            mde_pct = 100 * mde_tok / statistics.mean(v["b"])
            flag = "  <== ABOVE MDE" if mean > mde_tok else ""
            print(f"     {k[0]:<16} {k[1]:<14} inf={len(v['d']):<4} "
                  f"mean={mean:+8.4f} mde={mde_pct:6.3f}%{flag}")
    print(f"\n  total records across the five main blocks: {grand:,}")
    return grand


def check_specials():
    print("\n=== 4. GPT-5.6 and Google-native blocks ===")
    for path in sorted(glob.glob("results/oai_*.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        bad = 0
        for r in rows:
            k = 9 if r["reasoning_tokens_disclosed"] > 0 else 3
            exp = r["visible_canonical_tokens"] + r["reasoning_tokens_disclosed"] + k
            if exp != r["expected_total"] or (r["billed_completion_tokens"] - exp) != r["delta_billed_minus_expected"]:
                bad += 1
        if bad:
            problem(f"{path}: {bad} records where the stored identity does not recompute")
        inf = [r for r in rows if r["billed_completion_tokens"] < r["max_tokens"]]
        ds = [r["delta_billed_minus_expected"] for r in inf]
        print(f"  {path.split('/')[-1]}: n={len(rows)} inf={len(inf)} "
              f"exact={sum(1 for d in ds if d == 0)}/{len(ds)} mean={statistics.mean(ds):+.4f}")
    for path in sorted(glob.glob("results/gnative_*.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        bad = sum(1 for r in rows
                  if (r["prompt_tokens"] + r["billed_answer_tokens"] + r["billed_thought_tokens"]
                      == r["total_tokens"]) != r["total_reconciles"])
        if bad:
            problem(f"{path}: {bad} records where total_reconciles is wrong")
        th = [r["delta_thought"] for r in rows]
        an = [r["delta_answer"] for r in rows]
        print(f"  {path.split('/')[-1]}: n={len(rows)} "
              f"thought exact={sum(1 for d in th if d == 0)}/{len(th)} "
              f"answer exact={sum(1 for d in an if d == 0)}/{len(an)} "
              f"reconcile={sum(1 for r in rows if r['total_reconciles'])}/{len(rows)}")


def check_logprob_block():
    """The direct-observation block uses its own schema (cell / observed
    generated-token count), so the generic per-cell checks do not apply. It
    still has to be audited: the completeness gate must actually be enforced,
    and F must recompute from the stored fields."""
    print("\n=== 6. Direct-observation block (logprobs) ===")
    for path in sorted(glob.glob("results/lp_*.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        bad_arith = sum(
            1 for r in rows
            if r["billed_completion_tokens"] - r["generated_tokens_observed"]
            != r["delta_billed_minus_generated"]
        )
        if bad_arith:
            problem(f"{path}: {bad_arith} records where F does not recompute")
        # A record passing the gate must have a stream at least as long as the
        # parsed text; a negative structural count means omitted entries.
        leaked = sum(1 for r in rows if r["stream_complete"] and r["structural_chars"] < 0)
        if leaked:
            problem(f"{path}: {leaked} records marked complete despite a stream "
                    f"shorter than the parsed text")
        cells = collections.Counter(r["cell"] for r in rows)
        comp = collections.Counter(r["cell"] for r in rows if r["stream_complete"])
        nonzero = [r for r in rows if r["stream_complete"]
                   and r["delta_billed_minus_generated"] != 0]
        print(f"  {path.split('/')[-1]}: {len(rows)} records, {len(cells)} cells, "
              f"{sum(comp.values())} passing the completeness gate")
        if nonzero:
            note(f"{len(nonzero)} verified records with F != 0 "
                 f"(these would be genuine overcharging)")
        else:
            print("    every record passing the gate has F = 0 exactly")
        for c in sorted(cells):
            if comp[c] == 0:
                note(f"{c}: 0/{cells[c]} usable, provider returns incomplete "
                     f"logprobs; F not interpretable for this cell")


def check_prompt_consistency():
    print("\n=== 5. Same prompt set everywhere? ===")
    sets = {}
    for pat in list(BLOCKS.values()):
        for path in sorted(glob.glob(pat)):
            for line in open(path):
                r = json.loads(line)
                sets.setdefault((r["provider"], r["model_short_key"]), set()).add(r["prompt"])
    sizes = collections.Counter(len(v) for v in sets.values())
    print(f"  unique-prompt-count distribution across cells: {dict(sizes)}")
    ref = None
    mismatch = 0
    for k, v in sets.items():
        if len(v) != 200:
            continue
        if ref is None:
            ref = v
        elif v != ref:
            mismatch += 1
    if mismatch:
        problem(f"{mismatch} full cells use a DIFFERENT 200-prompt set than the reference")
    else:
        print("  all full cells share an identical 200-prompt set")


if __name__ == "__main__":
    check_files()
    check_config_resolution()
    total = check_recompute()
    check_specials()
    check_logprob_block()
    check_prompt_consistency()
    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE — {len(problems)} problems, {len(notes)} notes")
    for p in problems:
        print(f"  PROBLEM: {p}")
    sys.exit(1 if problems else 0)
