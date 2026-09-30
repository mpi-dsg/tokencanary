"""Shared configuration for the token-billing audit pilot."""

# Hand-verified as of the discover.py run logged in implementation-notes.md.
# Do not swap this for fuzzy auto-matching: catalogs are inconsistent enough
# (see deviations log) that model choice per provider needs a human call.
#
# short_key -> per-provider {"model_id": <id to send in API request>,
#                             "tokenizer_repo": <HF repo for canonical recompute>}
PROVIDER_MODEL_MAP = {
    "together": {
        # Every small (1B-8B) Llama/Gemma candidate 400s here -- moved to
        # dedicated-only endpoints. These two larger models are confirmed,
        # via live probe, still on real shared serverless. Not the same
        # size class as the original paper's models, but same audit logic
        # applies: public tokenizer, real pay-per-token billing.
        "llama": {
            "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "tokenizer_repo": "meta-llama/Llama-3.3-70B-Instruct",  # gated, needs HF license accepted
            "reasoning_format": "simple",  # no reasoning field observed
        },
        "gemma": {
            "model_id": "google/gemma-4-31B-it",
            "tokenizer_repo": "google/gemma-4-31B-it",  # ungated
            # Resolved via official docs (ai.google.dev/gemma/docs/capabilities/thinking):
            # real format is <|channel>thought\n{reasoning}\n<channel|>{text}.
            # Verified against real vocab and closes delta to 0/-1 on real
            # data. See implementation-notes.md.
            "reasoning_format": "gemma_thinking",
        },
    },
    "fireworks": {
        # No Llama/Gemma at all in current catalog (see implementation-notes.md).
        # Substituted a different open-weight family that IS actually served
        # here and has a fully public (ungated) tokenizer: OpenAI's gpt-oss.
        "gpt-oss-20b": {
            "model_id": "accounts/fireworks/models/gpt-oss-20b",
            "tokenizer_repo": "openai/gpt-oss-20b",  # ungated
            "reasoning_format": "harmony",  # verified against real data
        },
        "gpt-oss-120b": {
            "model_id": "accounts/fireworks/models/gpt-oss-120b",
            "tokenizer_repo": "openai/gpt-oss-120b",  # ungated
            "reasoning_format": "harmony",
        },
    },
    "deepinfra": {
        # No 1B/3B Llama or 1B Gemma on this reseller's current catalog.
        # Substituted with smallest available in-family model; see
        # implementation-notes.md deviations log.
        "llama": {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "tokenizer_repo": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "reasoning_format": "simple",
        },
        "gemma": {
            "model_id": "google/gemma-3-4b-it",
            "tokenizer_repo": "google/gemma-3-4b-it",
            "reasoning_format": "simple",
        },
    },
    "zai": {
        # Zhipu/Z.ai's own first-party API for their own GLM model --
        # unlike the resellers above, this tests whether the entity that
        # actually trained the model is honest about its own billing, not
        # a third party reselling it.
        "glm": {
            "model_id": "glm-4.6",
            "tokenizer_repo": "zai-org/GLM-4.6",
            # "think_tags" is the canonical name; "glm_thinking" was the
            # original alias and resolves identically (verified: 0/200 records
            # differ when recomputed both ways). Unified so the same cell is
            # not defined two ways across maps.
            "reasoning_format": "think_tags",
        },
    },
    "openrouter": {
        "llama": {
            "model_id": "meta-llama/llama-3.2-1b-instruct",
            # meta-llama/Llama-3.2-1B-Instruct is gated and, unlike the other
            # gated repos in this project, stuck in manual review (not an
            # instant grant). Using Unsloth's ungated re-upload instead --
            # same weights/tokenizer, no gating. Not byte-verified against
            # the original (still can't access it to compare), but Unsloth's
            # entire fine-tuning-tooling business depends on tokenizer
            # fidelity to these exact checkpoints. See implementation-notes.md.
            "tokenizer_repo": "unsloth/Llama-3.2-1B-Instruct",
            "reasoning_format": "simple",
        },
        # No 1B Gemma-3 on OpenRouter's catalog either; same substitution as DeepInfra.
        # NOTE: a live probe showed OpenRouter route this exact model to a
        # "DeepInfra" backend (response field provider="DeepInfra"). These
        # two cells may not be independent samples for gemma -- check the
        # openrouter_generation_stats.provider_name field per-request during
        # analysis rather than assuming independence.
        "gemma": {
            "model_id": "google/gemma-3-4b-it",
            "tokenizer_repo": "google/gemma-3-4b-it",
            "reasoning_format": "simple",
        },
    },
}

# --------------------------------------------------------------------------
# Expansion cells: DeepSeek / Kimi / GLM across the same four funded reseller
# accounts. Kept in a separate map so the original 8-cell study stays cleanly
# separable in analysis; merge into PROVIDER_MODEL_MAP at run time.
#
# Every entry below was calibrated by `calibrate_cells.py` against real
# responses -- `reasoning_format` and `append_eos` are the variant that
# reproduced the provider's own completion_tokens, NOT an assumption carried
# over from another provider. That distinction matters: the SAME checkpoint is
# billed under different conventions by different resellers (DeepInfra bills
# Kimi-K2.6 with no think-tags and no EOS; Fireworks and Together bill the
# same checkpoint WITH think-tags). Assuming a format transfers across
# providers would have manufactured multi-token phantom deltas.
#
# max_tokens is raised for these cells: they are heavy reasoning models, and
# at the study default (300) they truncate mid-thought almost every time. A
# response billed at exactly max_tokens cannot exhibit inflation at all --
# billing is pinned to the cap -- so those samples are evidentially empty.
EXPANSION_MAX_TOKENS = 1200

EXPANSION_MODEL_MAP = {
    "together": {
        "deepseek": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "tokenizer_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "reasoning_format": "think_tags",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "kimi": {
            "model_id": "moonshotai/Kimi-K2.6",
            "tokenizer_repo": "moonshotai/Kimi-K2.6",
            "reasoning_format": "think_tags",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        # No GLM cell on Together: zai-org/GLM-4.6 is catalog-listed but 400s
        # with "Unable to access non-serverless model ... create a dedicated
        # endpoint" -- the same size-class wall documented for Llama/Gemma in
        # implementation-notes.md, independently reconfirmed here.
    },
    "fireworks": {
        "deepseek": {
            "model_id": "accounts/fireworks/models/deepseek-v4-flash-0731",
            "tokenizer_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "reasoning_format": "think_tags",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "kimi": {
            "model_id": "accounts/fireworks/models/kimi-k2p6",
            "tokenizer_repo": "moonshotai/Kimi-K2.6",
            "reasoning_format": "think_tags",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "glm": {
            "model_id": "accounts/fireworks/models/glm-5p2",
            "tokenizer_repo": "zai-org/GLM-5.2",
            "reasoning_format": "think_tags",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
    },
    "deepinfra": {
        # Returns no reasoning field for this model at all, and billing matches
        # visible content + EOS exactly -- i.e. no hidden reasoning tokens are
        # being billed here. Calibrated, not assumed.
        "deepseek": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "tokenizer_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "reasoning_format": "simple",
            "append_eos": True,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "kimi": {
            "model_id": "moonshotai/Kimi-K2.6",
            "tokenizer_repo": "moonshotai/Kimi-K2.6",
            "reasoning_format": "simple",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "glm": {
            "model_id": "zai-org/GLM-4.6",
            "tokenizer_repo": "zai-org/GLM-4.6",
            "reasoning_format": "think_tags",
            "append_eos": True,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
    },
    "openrouter": {
        "deepseek": {
            "model_id": "deepseek/deepseek-v4-flash-0731",
            "tokenizer_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "reasoning_format": "think_tags",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "kimi": {
            "model_id": "moonshotai/kimi-k2.6",
            "tokenizer_repo": "moonshotai/Kimi-K2.6",
            "reasoning_format": "think_tags",
            "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        # GLM deliberately NOT run on OpenRouter. GLM is already covered three
        # ways -- Z.ai's own first-party API (the model's creator), DeepInfra
        # (GLM-4.6) and Fireworks (GLM-5.2) -- so a fourth reseller vantage on
        # the same family adds nothing but spend. Calibrated and verified
        # servable (think_tags + append_eos), so it can be reinstated cheaply
        # if a reason appears.
    },
    "openai": {
        # Non-reasoning models only. Calibrated live: billed completion_tokens
        # equals tiktoken(o200k_base) of the returned text EXACTLY (8/8 on
        # both models), with no EOS appended, and independently equals the
        # logprobs entry count (8/8). Reasoning models (o-series / GPT-5
        # reasoning) are deliberately excluded from this Tier-1 cell: their
        # reasoning content is discarded server-side, so the reasoning-token
        # portion of the bill has no ground truth to check against and would
        # not belong in a cell claiming verified counts.
        "gpt-4o-mini": {
            "model_id": "gpt-4o-mini",
            "tokenizer_repo": "tiktoken:o200k_base",
            "reasoning_format": "simple",
            "append_eos": False,
            "max_tokens": 400,
        },
        "gpt-4.1-nano": {
            "model_id": "gpt-4.1-nano",
            "tokenizer_repo": "tiktoken:o200k_base",
            "reasoning_format": "simple",
            "append_eos": False,
            "max_tokens": 400,
        },
    },
    "zai": {
        # Re-run of the first-party Z.ai cell at the raised cap. The original
        # pass (results/full_zai.jsonl, max_tokens=300) is real data but only
        # 6% informative: GLM-4.6 reasons heavily and 187 of 200 responses hit
        # the cap, and a capped response cannot show inflation either way.
        # This is the most interesting cell in the study -- the only one
        # testing a lab billing for its OWN model rather than a reseller --
        # so it should not be the weakest-evidenced one.
        "glm": {
            "model_id": "glm-4.6",
            "tokenizer_repo": "zai-org/GLM-4.6",
            "reasoning_format": "think_tags",
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
    },
}

# --------------------------------------------------------------------------
# Round 2: fixes for inconsistent model choices that were carry-over rather
# than deliberate. Each entry calibrated live by `calibrate_round2.py`.
#
#   - gpt-oss ran only on Fireworks although Together, DeepInfra and
#     OpenRouter all serve it. Now a 4-way comparison on a fully open,
#     ungated OpenAI model. Notably every provider bills it under the SAME
#     Harmony convention -- unlike Kimi, where conventions diverged.
#   - gemma ran as gemma-4-31B on Together but gemma-3-4b (older generation,
#     much smaller) on DeepInfra. DeepInfra carries gemma-4-31B-it, so this
#     matches Together exactly. DeepInfra surfaces no reasoning field for it
#     and bills content + EOS.
#   - GLM ran as 4.6 on Z.ai/DeepInfra but 5.2 on Fireworks, leaving the
#     current model without a cross-provider comparison. Note DeepInfra bills
#     GLM-4.6 WITH an EOS but GLM-5.2 WITHOUT -- conventions differ between
#     model versions on the same provider, not just between providers.
ROUND2_MODEL_MAP = {
    "together": {
        "gpt-oss-20b": {
            "model_id": "openai/gpt-oss-20b", "tokenizer_repo": "openai/gpt-oss-20b",
            "reasoning_format": "harmony", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "gpt-oss-120b": {
            "model_id": "openai/gpt-oss-120b", "tokenizer_repo": "openai/gpt-oss-120b",
            "reasoning_format": "harmony", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
    },
    "deepinfra": {
        "gpt-oss-20b": {
            "model_id": "openai/gpt-oss-20b", "tokenizer_repo": "openai/gpt-oss-20b",
            "reasoning_format": "harmony", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "gpt-oss-120b": {
            "model_id": "openai/gpt-oss-120b", "tokenizer_repo": "openai/gpt-oss-120b",
            "reasoning_format": "harmony", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "gemma4-31b": {
            "model_id": "google/gemma-4-31B-it", "tokenizer_repo": "google/gemma-4-31B-it",
            "reasoning_format": "simple", "append_eos": True,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "glm-5.2": {
            "model_id": "zai-org/GLM-5.2", "tokenizer_repo": "zai-org/GLM-5.2",
            "reasoning_format": "think_tags", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
    },
    "openrouter": {
        "gpt-oss-20b": {
            "model_id": "openai/gpt-oss-20b", "tokenizer_repo": "openai/gpt-oss-20b",
            "reasoning_format": "harmony", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
        "gpt-oss-120b": {
            "model_id": "openai/gpt-oss-120b", "tokenizer_repo": "openai/gpt-oss-120b",
            "reasoning_format": "harmony", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
    },
    "zai": {
        "glm-5.2": {
            "model_id": "glm-5.2", "tokenizer_repo": "zai-org/GLM-5.2",
            "reasoning_format": "think_tags", "append_eos": False,
            "max_tokens": EXPANSION_MAX_TOKENS,
        },
    },
}


def lookup_cell(provider, model_short_key):
    """Config entry for a (provider, model) cell, searching the main study map
    first and then the expansion map. Returns None if the cell is unknown --
    callers must treat that as "cannot recompute", never as "delta is zero"."""
    for mp in (PROVIDER_MODEL_MAP, EXPANSION_MODEL_MAP, ROUND2_MODEL_MAP):
        entry = mp.get(provider, {}).get(model_short_key)
        if entry is not None:
            return entry
    return None


PROVIDERS = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "zai": {
        # Zhipu/Z.ai's own first-party API for their own GLM models -- this
        # is the actual model creator, not a reseller. Base URL confirmed
        # via docs.z.ai (2026-08), not assumed.
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZAI_API_KEY",
    },
    "openai": {
        # OpenAI publishes its tokenizer (tiktoken o200k_base) even though it
        # does not publish weights, so visible output IS verifiable against
        # real ground truth here -- contrary to the usual "closed model, no
        # ground truth" assumption. Two independent checks are available and
        # both are recorded: tiktoken recomputation, and the logprobs entry
        # count (one entry per generated token, no tokenizer required).
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        # Newer OpenAI models reject the legacy parameter name.
        "max_tokens_field": "max_completion_tokens",
        "want_logprobs": True,
    },
}

# Generation settings, matching the original paper's headline (highest-signal)
# setting: temperature 1.3, top_p 0.95, ~200-300 token outputs.
GENERATION = {
    "system_prompt": "You are a helpful assistant. Be clear and concise.",
    "temperature": 1.3,
    "top_p": 0.95,
    "max_tokens": 300,
}

# Per-provider overrides where the standard GENERATION settings aren't
# accepted. Confirmed live: Z.ai rejects temperature > 1.0 ("The
# temperature parameter is illegal", code 1210) -- capped at their max.
GENERATION_OVERRIDES = {
    "zai": {"temperature": 1.0},
}

# Pilot sizing (small paid run first).
N_PROMPTS = 200

# Hard safety cap: abort a run if it exceeds this many total API requests,
# regardless of N_PROMPTS * models * providers math. Cheap insurance against
# a config bug looping unexpectedly.
MAX_REQUESTS_HARD_CAP = 2000
