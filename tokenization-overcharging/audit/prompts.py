"""Prompt sourcing for the audit pilot.

Primary source: a subset of the same LMSYS Chatbot Arena data the original
paper used (English, first-turn user message length in [20, 100] chars),
for direct comparability. That dataset is gated on HF, same as the model
repos. Falls back to an ungated instruction dataset if lmsys access isn't
available on the configured HF_TOKEN -- logged as a deviation either way.
"""
import os

from datasets import load_dataset

MIN_LEN, MAX_LEN = 20, 100


def _from_lmsys(n):
    ds = load_dataset(
        "lmsys/lmsys-chat-1m", split="train", token=os.environ.get("HF_TOKEN")
    )
    prompts = []
    seen = set()
    for row in ds:
        if row.get("language") != "English":
            continue
        conv = row.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        text = conv[0]["content"].strip()
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        if len(prompts) >= n:
            break
    return prompts, "lmsys/lmsys-chat-1m"


def _from_dolly_fallback(n):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    prompts = []
    seen = set()
    for row in ds:
        text = (row.get("instruction") or "").strip()
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        if len(prompts) >= n:
            break
    return prompts, "databricks/databricks-dolly-15k (fallback: no lmsys access)"


def load_prompts(n):
    try:
        prompts, source = _from_lmsys(n)
        if len(prompts) < n:
            raise RuntimeError(
                f"lmsys yielded only {len(prompts)}/{n} prompts after filtering"
            )
        return prompts, source
    except Exception as e:  # noqa: BLE001 - gated-access failure, quota, etc.
        print(f"[prompts] lmsys source unavailable ({e}); falling back to Dolly")
        return _from_dolly_fallback(n)


if __name__ == "__main__":
    import env

    env.ensure_loaded()
    from config import N_PROMPTS

    prompts, source = load_prompts(N_PROMPTS)
    print(f"Loaded {len(prompts)} prompts from {source}")
    for p in prompts[:5]:
        print(f"  - {p}")
