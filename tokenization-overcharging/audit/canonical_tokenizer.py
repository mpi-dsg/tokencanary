"""Canonical token-count ground truth via each model's real public tokenizer.

This is the whole point of restricting the pilot to open-weight resellers:
for these models the tokenizer is public, so we can independently recompute
what a faithful provider *should* report, rather than trusting the provider's
own count (which is all the closed-API tier can ever do).
"""
import json
import os

from transformers import AutoTokenizer

_cache = {}

# Local cache dir for repos that cannot be loaded straight from the Hub id.
_SNAPSHOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tokenizer_cache")

# Some repos don't load through a plain AutoTokenizer.from_pretrained(repo_id)
# under the pinned transformers version. Rather than upgrade transformers
# mid-study -- which could silently shift the canonical counts underpinning
# the 3200 requests already collected -- each exception gets an explicit,
# documented loader. Keys are HF repo ids; values are the strategy.
#
#   "tokenizer_json": load tokenizer.json directly with the `tokenizers`
#       library, bypassing the transformers auto-class machinery. Byte-for-byte
#       the same tokenizer a Fast transformers tokenizer would wrap.
#   "local_snapshot": download to a dot-free local directory, then load with
#       trust_remote_code. Needed for repos whose *name* contains a dot
#       (e.g. "Kimi-K2.6"), which breaks the transformers_modules Python
#       module path and raises a confusing ModuleNotFoundError.
TOKENIZER_LOADERS = {
    # transformers 4.46.3 doesn't know GLM-5.2's "TokenizersBackend" class.
    "zai-org/GLM-5.2": "tokenizer_json",
    # Avoids pulling this repo's remote modeling code just to tokenize.
    "deepseek-ai/DeepSeek-V4-Flash-0731": "tokenizer_json",
    # Dot in the repo name breaks dynamic module import.
    "moonshotai/Kimi-K2.6": "local_snapshot",
}


class _FastTokenizerAdapter:
    """Presents a raw `tokenizers.Tokenizer` with the same surface the rest of
    this module expects from a transformers tokenizer: `encode(text,
    add_special_tokens=...) -> list[int]` and an `.eos_token` attribute."""

    def __init__(self, tok, eos_token):
        self._tok = tok
        self.eos_token = eos_token

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids


def _load_tokenizer_json(repo_id, token):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(repo_id, "tokenizer.json", token=token))
    eos = None
    try:
        with open(hf_hub_download(repo_id, "tokenizer_config.json", token=token)) as f:
            cfg = json.load(f)
        eos = cfg.get("eos_token")
        if isinstance(eos, dict):  # sometimes serialized as an AddedToken dict
            eos = eos.get("content")
    except Exception:  # noqa: BLE001 - eos is optional; absence just means no append
        eos = None
    return _FastTokenizerAdapter(tok, eos)


def _load_local_snapshot(repo_id, token):
    from huggingface_hub import snapshot_download

    safe_name = repo_id.split("/")[-1].replace(".", "_")
    local_dir = snapshot_download(
        repo_id,
        token=token,
        allow_patterns=[
            "tiktoken.model", "*.py", "tokenizer_config.json",
            "tokenizer.json", "chat_template.jinja", "*.txt", "*.model",
        ],
        local_dir=os.path.join(_SNAPSHOT_DIR, safe_name),
    )
    return AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)


class _TiktokenAdapter:
    """Adapter for OpenAI's published `tiktoken` encodings, so proprietary
    OpenAI models can be audited with the same machinery as open-weight ones.

    OpenAI publishes the tokenizer even though it does not publish weights --
    `o200k_base` covers GPT-4o, GPT-4.1, GPT-5 and o3. That makes their
    *visible* output verifiable against real ground truth, unlike Anthropic
    and Gemini's proprietary models. It is one notch weaker than an
    open-weight tokenizer (we cannot check the tokenizer against the weights),
    but it is the provider's own published artifact, so a mismatch would be an
    internal contradiction rather than a difference of opinion.

    No eos_token: tiktoken encodings carry no chat-turn terminator concept,
    so EOS handling for these cells is decided empirically per cell via
    `append_eos`, not inferred from the tokenizer.
    """

    def __init__(self, enc):
        self._enc = enc
        self.eos_token = None

    def encode(self, text, add_special_tokens=False):
        # disallowed_special=() so special-token-looking substrings in model
        # output are encoded as ordinary text instead of raising.
        return self._enc.encode(text, disallowed_special=())


def get_tokenizer(repo_id):
    if repo_id not in _cache:
        token = os.environ.get("HF_TOKEN")
        # Pseudo-repo form "tiktoken:<encoding>" selects an OpenAI encoding.
        if repo_id.startswith("tiktoken:"):
            import tiktoken

            _cache[repo_id] = _TiktokenAdapter(
                tiktoken.get_encoding(repo_id.split(":", 1)[1])
            )
            return _cache[repo_id]
        strategy = TOKENIZER_LOADERS.get(repo_id)
        if strategy == "tokenizer_json":
            _cache[repo_id] = _load_tokenizer_json(repo_id, token)
        elif strategy == "local_snapshot":
            _cache[repo_id] = _load_local_snapshot(repo_id, token)
        else:
            _cache[repo_id] = AutoTokenizer.from_pretrained(repo_id, token=token)
    return _cache[repo_id]


# OpenAI Harmony format (openai/harmony spec). The billed stream starts one
# token before `<|channel|>` -- verified by bracketing against real Fireworks
# data: `<|start|>assistant<|channel|>...` (2-token prefix) overshoots by
# exactly 1, while a bare `<|channel|>...` (0-token prefix) undershoots by
# exactly 1. A 1-token prefix gives 20/20 exact matches on BOTH gpt-oss-20b
# and gpt-oss-120b. Note `<|start|>` and `assistant` are each exactly one
# token, so "assistant<|channel|>" and "<|start|><|channel|>" are
# count-identical -- token counting cannot distinguish them, and no number in
# this study depends on which reading is correct.
HARMONY_OPENING = "assistant<|channel|>analysis<|message|>"
HARMONY_TRANSITION = "<|end|><|start|>assistant<|channel|>final<|message|>"

# Gemma "thinking mode" format, per https://ai.google.dev/gemma/docs/capabilities/thinking
# -- confirmed against real gemma-4-31B-it vocab (tokens <|channel> id 100,
# <channel|> id 101 both real) and verified to close the delta to 0/-1 on
# real saved responses (implementation-notes.md). Note the asymmetric
# bracket placement is correct, not a typo: opening is `<|channel>`,
# closing is `<channel|>`.
GEMMA_THINK_OPEN = "<|channel>thought\n"
GEMMA_THINK_CLOSE = "\n<channel|>"

# Plain <think>...</think> wrapper. Originally calibrated on GLM-4.6 (real
# single vocab tokens 151350/151351; exact match on a natural completion, -1
# on a truncated one). Since confirmed as the shared convention across three
# further open-weight families, each verified independently rather than
# assumed:
#   - DeepSeek-V4-Flash-0731: <think>/</think> are real added tokens 128821/128822.
#   - GLM-5.2: real added tokens 154841/154842.
#   - Kimi-K2.6: its own official chat_template.jinja renders assistant turns as
#     `<think>{{reasoning}}</think>{{content}}` -- read from the template, not
#     reverse-engineered.
# "think_tags" is the canonical name; "glm_thinking" is kept as an alias so
# already-collected Z.ai records keep recomputing identically.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_TAG_FORMATS = {"think_tags", "glm_thinking"}

# Backwards-compatible aliases (older code/notes reference these names).
GLM_THINK_OPEN = THINK_OPEN
GLM_THINK_CLOSE = THINK_CLOSE

# Formats that do NOT bill a separate eos_token on natural completion. This
# turns out to be the general rule for reasoning formats: the closing
# structural token (`</think>`, `<channel|>`, Harmony's final-channel message)
# already terminates the turn, so appending eos_token on top double-counts.
# Verified independently per format against real informative (non-truncated)
# samples:
#   - think_tags:      GLM-4.6 via Z.ai native, plus every calibrated
#                      DeepSeek/Kimi/GLM reseller cell.
#   - gemma_thinking:  20/20 exact on Together's gemma-4-31B-it (with EOS it
#                      is a constant -1).
#   - harmony:         20/20 exact on both Fireworks gpt-oss models (with EOS
#                      it is a constant -2).
# The plain "simple" (non-reasoning) format is the opposite: those models DO
# bill their own EOS, which is what the original +1 calibration established.
# This is a per-format DEFAULT; a cell may override with `append_eos` in
# config when live data shows that reseller differs.
NO_EOS_APPEND_FORMATS = {"think_tags", "glm_thinking", "gemma_thinking", "harmony"}


def reconstruct_full_text(reasoning, text, reasoning_format):
    """Rebuild the string a faithful provider actually tokenizes for
    billing, including structural tokens between reasoning and the visible
    answer that plain concatenation would miss. Verified against real data
    per implementation-notes.md -- do not add new formats here without the
    same live-data verification, or a plausible-looking number becomes
    ground truth by accident.
    """
    if not reasoning:
        return text
    if reasoning_format == "harmony":
        # Verified: closes gpt-oss delta from +9/+10 to -1/-2 on real data.
        # BUT: if `text` is empty, generation was truncated mid-reasoning
        # and never reached the "final" channel at all -- the transition
        # tokens were never actually generated, and adding them anyway
        # overshoots by ~9 tokens (confirmed on real data, see
        # implementation-notes.md). Only include the transition when there's
        # real final-channel content to transition into.
        if text:
            return HARMONY_OPENING + reasoning + HARMONY_TRANSITION + text
        return HARMONY_OPENING + reasoning
    if reasoning_format == "gemma_thinking":
        # Same truncation caveat as harmony: if `text` is empty, generation
        # was cut off mid-thought and never reached the closing marker.
        if text:
            return GEMMA_THINK_OPEN + reasoning + GEMMA_THINK_CLOSE + text
        return GEMMA_THINK_OPEN + reasoning
    if reasoning_format in THINK_TAG_FORMATS:
        # Same truncation caveat as the other reasoning formats: if `text` is
        # empty the model was cut off inside the thought and never emitted the
        # closing tag, so adding it would overcount.
        if text:
            return THINK_OPEN + reasoning + THINK_CLOSE + text
        return THINK_OPEN + reasoning
    # "simple": plain concatenation, confirmed sufficient where no
    # structural tokens were found to exist (i.e. no reasoning at all).
    return reasoning + text


def effective_finish_reason(billed_completion_tokens, reported_finish_reason, max_tokens):
    """Some backends mislabel max_tokens-truncated responses as "stop"
    instead of "length" -- confirmed on real data (Cloudflare, serving
    OpenRouter's llama-3.2-1b-instruct): every response that hit exactly
    the configured max_tokens was labeled "stop", which is not plausible as
    genuine coincidence at that frequency. billed == max_tokens is a more
    reliable truncation signal than the reported label. Applied defensively
    for any provider, since nothing rules out the same bug elsewhere.
    See implementation-notes.md.
    """
    if billed_completion_tokens == max_tokens and reported_finish_reason == "stop":
        return "length"
    return reported_finish_reason


def canonical_token_count(
    repo_id, reasoning, text, finish_reason, reasoning_format, append_eos=None
):
    """Count tokens the way a faithful provider's completion_tokens should:
    the full generated content (reconstructed per reasoning_format) plus the
    model's real EOS token IF generation actually finished naturally.

    `append_eos` overrides the per-format default when a specific
    (provider, model) cell was empirically shown to bill differently from
    others sharing its reasoning format. None = use the format default.
    Pass True/False only with live-data evidence, same as everything else here.

    Confirmed empirically (implementation-notes.md) that a faithful
    provider's completion_tokens = len(encode(text, add_special_tokens=False))
    + 1, and that +1 is exactly the model's own eos_token -- a real token
    the model generated to end the turn, not padding. But that token only
    exists when generation stopped on its own (finish_reason == "stop").
    When a response is truncated by max_tokens (finish_reason == "length"),
    there is no EOS to count, and appending one would introduce a new,
    wrong +1 in the other direction.
    """
    tok = get_tokenizer(repo_id)
    full_text = reconstruct_full_text(reasoning, text, reasoning_format)
    base = len(tok.encode(full_text, add_special_tokens=False))
    if append_eos is None:
        append_eos = reasoning_format not in NO_EOS_APPEND_FORMATS
    if finish_reason == "stop" and tok.eos_token and append_eos:
        return len(tok.encode(full_text + tok.eos_token, add_special_tokens=False))
    return base
