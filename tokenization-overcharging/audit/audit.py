"""Main audit run: query each (provider, model) cell with real prompts,
recompute canonical token counts, log billed-vs-canonical deltas.

Safety: defaults to a small smoke-test batch. Pass --full to run the
complete pilot (config.N_PROMPTS per cell). Always run discover.py first.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count, effective_finish_reason  # noqa: E402
from config import (  # noqa: E402
    GENERATION,
    GENERATION_OVERRIDES,
    MAX_REQUESTS_HARD_CAP,
    N_PROMPTS,
    PROVIDER_MODEL_MAP,
)
from prompts import load_prompts  # noqa: E402
from providers import build_client  # noqa: E402


def _existing_prompts(out_path):
    """Set of prompt texts already successfully recorded per (provider,
    model) in out_path. Matching by prompt content, not position/count --
    failures can be scattered anywhere in the sequence (confirmed: a run
    with 9 failures had them spread throughout, not just at the tail), so
    "resume from index N" would silently skip early failures forever while
    redundantly repeating later successes."""
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                key = (r["provider"], r["model_short_key"])
                done.setdefault(key, set()).add(r["prompt"])
    return done


def run(n_prompts, out_path, resume=False, max_new_requests=None):
    prompts, source = load_prompts(n_prompts)
    print(f"Loaded {len(prompts)} prompts from {source}")

    existing = _existing_prompts(out_path) if resume else {}
    existing_counts = {k: len(v) for k, v in existing.items()}
    if existing_counts:
        print(f"Resuming {out_path}: {existing_counts}")

    total_cells = sum(
        1 for m in PROVIDER_MODEL_MAP.values() for e in m.values()
        if e.get("reasoning_format") != "unresolved"
    )
    total_requests = total_cells * len(prompts) - sum(existing_counts.values())
    if total_requests > MAX_REQUESTS_HARD_CAP:
        raise RuntimeError(
            f"{total_requests} requests exceeds MAX_REQUESTS_HARD_CAP="
            f"{MAX_REQUESTS_HARD_CAP}; reduce n_prompts or raise the cap deliberately."
        )
    print(f"Plan: {total_cells} (provider, model) cells x {len(prompts)} prompts, "
          f"{sum(existing_counts.values())} already done, {total_requests} remaining")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_done = 0
    n_errors = 0
    stop = False
    with open(out_path, "a") as f:
        for provider_key, model_entries in PROVIDER_MODEL_MAP.items():
            if stop:
                break
            client = build_client(provider_key)
            gen = {**GENERATION, **GENERATION_OVERRIDES.get(provider_key, {})}
            for short_key, entry in model_entries.items():
                if stop:
                    break
                if entry.get("reasoning_format") == "unresolved":
                    print(f"\n--- {provider_key} / {short_key}: SKIPPED "
                          f"(reasoning_format=unresolved, see implementation-notes.md) ---")
                    continue
                done_prompts = existing.get((provider_key, short_key), set())
                if len(done_prompts) >= len(prompts):
                    print(f"\n--- {provider_key} / {short_key}: already complete "
                          f"({len(done_prompts)}/{len(prompts)}), skipping ---")
                    continue
                model_id = entry["model_id"]
                tokenizer_repo = entry["tokenizer_repo"]
                reasoning_format = entry["reasoning_format"]
                # Per-cell overrides. max_tokens especially: heavy reasoning
                # models truncate mid-thought at the study default, and a
                # response billed at exactly max_tokens is structurally
                # incapable of showing inflation (billing is pinned to the
                # cap), so those samples carry no evidence either way.
                append_eos = entry.get("append_eos")
                cell_max_tokens = entry.get("max_tokens", gen["max_tokens"])
                print(f"\n--- {provider_key} / {short_key} ({model_id}) "
                      f"[{len(done_prompts)}/{len(prompts)} already done] ---")
                for i, prompt in enumerate(prompts):
                    if prompt in done_prompts:
                        continue
                    try:
                        result = client.chat_completion(
                            model=model_id,
                            system_prompt=gen["system_prompt"],
                            user_prompt=prompt,
                            temperature=gen["temperature"],
                            top_p=gen["top_p"],
                            max_tokens=cell_max_tokens,
                        )
                    except Exception as e:  # noqa: BLE001
                        n_errors += 1
                        print(f"  [{i}] ERROR: {e}")
                        continue

                    billed = result["billed_completion_tokens"]
                    canonical = canonical_token_count(
                        tokenizer_repo,
                        result["reasoning_text"],
                        result["text"],
                        finish_reason=effective_finish_reason(
                            billed, result["finish_reason"], cell_max_tokens
                        ),
                        reasoning_format=reasoning_format,
                        append_eos=append_eos,
                    )

                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "provider": provider_key,
                        "model_short_key": short_key,
                        "model_id": model_id,
                        "tokenizer_repo": tokenizer_repo,
                        "prompt": prompt,
                        # Recorded per-request so later recomputation never has
                        # to guess which cap was in force; cells differ, and
                        # `billed == max_tokens` is what marks a sample as
                        # uninformative for inflation.
                        "max_tokens": cell_max_tokens,
                        "finish_reason": result["finish_reason"],
                        # Present only where the provider returns logprobs
                        # (OpenAI). One entry per generated token, so this is
                        # a billed-count check independent of any tokenizer.
                        "logprob_entries": result.get("logprob_entries"),
                        "response_text": result["text"],
                        "reasoning_text": result["reasoning_text"],
                        "billed_completion_tokens": billed,
                        "canonical_completion_tokens": canonical,
                        "delta_billed_minus_canonical": (
                            billed - canonical if billed is not None else None
                        ),
                        "response_id": result["response_id"],
                        "raw_usage": result["raw_usage"],
                    }

                    if provider_key == "openrouter" and result["response_id"]:
                        gen_stats = client.generation_stats(result["response_id"])
                        if gen_stats:
                            record["openrouter_generation_stats"] = {
                                "native_tokens_completion": gen_stats.get(
                                    "native_tokens_completion"
                                ),
                                "provider_name": gen_stats.get("provider_name"),
                            }

                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    n_done += 1

                    delta = record["delta_billed_minus_canonical"]
                    flag = "" if delta == 0 else f"  <-- delta={delta}"
                    print(f"  [{i}] billed={billed} canonical={canonical}{flag}")

                    if max_new_requests is not None and n_done >= max_new_requests:
                        print(f"\n[chunk limit] stopping after {n_done} new requests "
                              f"this call (max_new_requests={max_new_requests})")
                        stop = True
                        break

                    time.sleep(0.1)  # light rate-limit courtesy

    print(f"\nDone. {n_done} requests logged, {n_errors} errors. Results: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help=f"Run the full pilot ({N_PROMPTS} prompts/cell). Default is a 5-prompt smoke test.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Override prompt count explicitly.")
    parser.add_argument("--out", default=None, help="Output path (for --resume, must point at an existing file).")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed (provider, model) records in --out.")
    args = parser.parse_args()

    n_prompts = args.limit if args.limit is not None else (N_PROMPTS if args.full else 5)
    if args.out:
        out_path = args.out
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(os.path.dirname(__file__), "results", f"pilot_{timestamp}.jsonl")
    run(n_prompts, out_path, resume=args.resume)


if __name__ == "__main__":
    main()
