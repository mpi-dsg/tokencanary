"""Targeted investigation: is OpenRouter-llama's wide delta variance caused
by multi-backend routing (ruled out -- only Cloudflare serves this model),
Unsloth tokenizer mismatch (one fresh sample matched exactly, not ruled in),
or something else (e.g. degenerate high-temp output from a small, weak
model)? Run a fresh pinned batch and inspect.
"""
import json
import time

import env

env.ensure_loaded()

from canonical_tokenizer import canonical_token_count  # noqa: E402
from prompts import load_prompts  # noqa: E402
from providers import build_client  # noqa: E402
from config import GENERATION  # noqa: E402

client = build_client("openrouter")
prompts, _ = load_prompts(20)

results = []
for i, prompt in enumerate(prompts):
    payload_extra = {"provider": {"order": ["cloudflare"], "allow_fallbacks": False}}
    # Reuse chat_completion's request shape but inject provider pin via a raw call.
    import requests
    import os
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.2-1b-instruct",
        "messages": [
            {"role": "system", "content": GENERATION["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        "temperature": GENERATION["temperature"],
        "top_p": GENERATION["top_p"],
        "max_tokens": GENERATION["max_tokens"],
        **payload_extra,
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
    data = resp.json()
    if "choices" not in data:
        print(f"[{i}] ERROR: {data}")
        continue
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    finish_reason = data["choices"][0].get("finish_reason")
    billed = data.get("usage", {}).get("completion_tokens")
    canonical = canonical_token_count(
        "unsloth/Llama-3.2-1B-Instruct", "", text, finish_reason=finish_reason, reasoning_format="simple"
    )
    delta = billed - canonical if billed is not None else None
    results.append({"prompt": prompt, "text": text, "finish_reason": finish_reason, "billed": billed, "canonical": canonical, "delta": delta})
    flag = "" if delta == 0 else f"  <-- delta={delta}"
    print(f"[{i}] billed={billed} canonical={canonical} finish={finish_reason}{flag}")
    time.sleep(0.2)

with open("results/openrouter_llama_pinned_investigation.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

deltas = [r["delta"] for r in results if r["delta"] is not None]
print(f"\nn={len(deltas)} mean={sum(deltas)/len(deltas):.2f} min={min(deltas)} max={max(deltas)}")
