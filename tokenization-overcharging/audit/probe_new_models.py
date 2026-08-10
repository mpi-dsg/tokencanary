"""Pre-flight probe for candidate expansion cells (DeepSeek / Kimi / GLM).

Same discipline every other cell in this study went through: before a model is
added to config.PROVIDER_MODEL_MAP, confirm with a real request that (a) it is
actually servable on that provider's pay-per-token tier (catalog presence is
NOT servability -- see implementation-notes.md on Together), and (b) which
field its reasoning text arrives in, so canonical recomputation can reconstruct
what the provider actually billed for.

Prints one line per cell plus the raw message-key set, which is what tells us
whether an existing reasoning_format applies or a new one has to be calibrated.
"""
import json
import sys

import env

env.ensure_loaded()

from config import PROVIDERS  # noqa: E402
from providers import build_client  # noqa: E402

# Deliberately matched across providers where possible: the SAME underlying
# checkpoint served by several resellers lets us compare providers against each
# other, not just each against its own tokenizer.
CANDIDATES = {
    "together": {
        "deepseek": ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731"),
        "kimi": ("moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.6"),
        "glm": ("zai-org/GLM-4.6", "zai-org/GLM-4.6"),
    },
    "fireworks": {
        "deepseek": ("accounts/fireworks/models/deepseek-v4-flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731"),
        "kimi": ("accounts/fireworks/models/kimi-k2p6", "moonshotai/Kimi-K2.6"),
        "glm": ("accounts/fireworks/models/glm-5p2", "zai-org/GLM-5.2"),
    },
    "deepinfra": {
        "deepseek": ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731"),
        "kimi": ("moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.6"),
        "glm": ("zai-org/GLM-4.6", "zai-org/GLM-4.6"),
    },
    "openrouter": {
        "deepseek": ("deepseek/deepseek-v4-flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731"),
        "kimi": ("moonshotai/kimi-k2.6", "moonshotai/Kimi-K2.6"),
        "glm": ("z-ai/glm-4.6", "zai-org/GLM-4.6"),
    },
}

PROBE_PROMPT = "Which is a species of fish: Tope or Rope?"


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for provider, cells in CANDIDATES.items():
        if only and provider != only:
            continue
        client = build_client(provider)
        for short_key, (model_id, tokenizer_repo) in cells.items():
            try:
                r = client.chat_completion(
                    model=model_id,
                    system_prompt=None,
                    user_prompt=PROBE_PROMPT,
                    temperature=1.0,
                    top_p=0.95,
                    # Generous: reasoning models need headroom to emit BOTH a
                    # reasoning block and visible content. A truncated probe
                    # tells us nothing about the reasoning/answer boundary.
                    max_tokens=800,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{provider:<11} {short_key:<9} FAIL {str(e)[:110]}")
                continue
            msg = r["raw"]["choices"][0].get("message", {})
            keys = sorted(k for k in msg if msg[k])
            print(
                f"{provider:<11} {short_key:<9} OK   billed={r['billed_completion_tokens']:<5} "
                f"finish={r['finish_reason']:<7} reasoning_len={len(r['reasoning_text']):<5} "
                f"content_len={len(r['text']):<5} msg_keys={keys}"
            )


if __name__ == "__main__":
    main()
