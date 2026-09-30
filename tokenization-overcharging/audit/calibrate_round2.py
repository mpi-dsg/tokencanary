"""Calibration for the round-2 cells that fix inconsistent model choices.

Three problems being fixed, all of them carry-over rather than deliberate:
  1. gpt-oss ran only on Fireworks, though Together, DeepInfra and OpenRouter
     all serve it -- an orphaned cell that should be a 4-way comparison on a
     fully open, ungated OpenAI model.
  2. gemma ran as gemma-4-31B on Together but gemma-3-4b (older generation AND
     much smaller) on DeepInfra/OpenRouter, though DeepInfra carries
     gemma-4-31B-it.
  3. GLM ran as 4.6 on Z.ai and DeepInfra but 5.2 on Fireworks, so the current
     model has no cross-provider comparison.

Every cell is calibrated against live responses before being run. Formats do
NOT transfer across providers -- established the hard way on Kimi, where
DeepInfra bills the identical checkpoint with no think-tags while three other
resellers bill it with them.
"""
import statistics
import sys

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from prompts import load_prompts  # noqa: E402
from providers import build_client  # noqa: E402

MAX_TOKENS = 1200
N = 6

# (label, provider, model_id, tokenizer_repo)
CANDIDATES = [
    # 1. gpt-oss everywhere it is actually served
    ("together/gpt-oss-20b",    "together",   "openai/gpt-oss-20b",  "openai/gpt-oss-20b"),
    ("together/gpt-oss-120b",   "together",   "openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    ("deepinfra/gpt-oss-20b",   "deepinfra",  "openai/gpt-oss-20b",  "openai/gpt-oss-20b"),
    ("deepinfra/gpt-oss-120b",  "deepinfra",  "openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    ("openrouter/gpt-oss-20b",  "openrouter", "openai/gpt-oss-20b",  "openai/gpt-oss-20b"),
    ("openrouter/gpt-oss-120b", "openrouter", "openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    # 2. gemma-4-31B on DeepInfra so it matches Together exactly
    ("deepinfra/gemma4-31b",    "deepinfra",  "google/gemma-4-31B-it", "google/gemma-4-31B-it"),
    # 3. GLM-5.2 -- the current model -- on the two platforms still on 4.6
    ("deepinfra/glm-5.2",       "deepinfra",  "zai-org/GLM-5.2",     "zai-org/GLM-5.2"),
    ("zai/glm-5.2",             "zai",        "glm-5.2",             "zai-org/GLM-5.2"),
]

VARIANTS = [
    ("simple/no-eos",      "simple",     False),
    ("simple/eos",         "simple",     True),
    ("think_tags/no-eos",  "think_tags", False),
    ("think_tags/eos",     "think_tags", True),
    ("harmony/no-eos",     "harmony",    False),
    ("harmony/eos",        "harmony",    True),
    ("gemma_think/no-eos", "gemma_thinking", False),
    ("gemma_think/eos",    "gemma_thinking", True),
]


def main():
    only = set(sys.argv[1:])
    prompts, source = load_prompts(N)
    print(f"prompts: {len(prompts)} from {source}\n")
    summary = {}
    for label, provider, model_id, tok_repo in CANDIDATES:
        if only and label not in only:
            continue
        print(f"=== {label}  ({model_id}) ===")
        try:
            client = build_client(provider)
        except Exception as e:  # noqa: BLE001
            print(f"    client error: {str(e)[:100]}\n")
            continue
        # Z.ai caps temperature at 1.0; everything else runs the study default.
        temp = 1.0 if provider == "zai" else 1.0
        rows = []
        for p in prompts:
            try:
                r = client.chat_completion(
                    model=model_id,
                    system_prompt="You are a helpful assistant. Be clear and concise.",
                    user_prompt=p, temperature=temp, top_p=0.95,
                    max_tokens=MAX_TOKENS,
                )
            except Exception as e:  # noqa: BLE001
                print(f"    request failed: {str(e)[:110]}")
                continue
            b = r["billed_completion_tokens"]
            if b is None:
                continue
            rows.append((b, r["reasoning_text"], r["text"],
                         effective_finish_reason(b, r["finish_reason"], MAX_TOKENS)))
        if not rows:
            print("    NO DATA\n")
            continue
        print(f"    n={len(rows)}  with_reasoning={sum(1 for x in rows if x[1])}  "
              f"natural_stop={sum(1 for x in rows if x[3]=='stop')}")
        best = None
        for vname, fmt, eos in VARIANTS:
            ds = []
            for b, rs, tx, fr in rows:
                try:
                    c = canonical_token_count(tok_repo, rs, tx, finish_reason=fr,
                                              reasoning_format=fmt, append_eos=eos)
                except Exception:  # noqa: BLE001
                    ds = None
                    break
                ds.append(b - c)
            if not ds:
                continue
            nex = sum(1 for d in ds if d == 0)
            score = (nex, -abs(statistics.mean(ds)))
            print(f"      {vname:<20} mean={statistics.mean(ds):>8.2f} exact={nex}/{len(ds)}")
            if best is None or score > best[0]:
                best = (score, vname, fmt, eos, statistics.mean(ds), nex, len(ds))
        if best:
            print(f"    -> best: {best[1]}  (exact {best[5]}/{best[6]}, mean {best[4]:+.2f})")
            summary[label] = best
        print()

    print("=" * 78)
    print("RECOMMENDED CONFIG")
    print("=" * 78)
    for label, b in summary.items():
        ok = "OK " if b[5] >= b[6] - 1 else "!! "
        print(f"  {ok}{label:<26} reasoning_format={b[2]!r:<18} append_eos={b[3]!s:<6} "
              f"(exact {b[5]}/{b[6]}, mean {b[4]:+.2f})")


if __name__ == "__main__":
    main()
