# Implementation notes, token-billing audit pilot

## Deviations from the original plan

- **Python 3.14 → 3.13 venv.** `tokenizers` (Rust/PyO3-backed) has no prebuilt
  wheel for 3.14 yet and fails to build from source (PyO3 doesn't support
  3.14 as of the pinned `tokenizers==0.20.3`). Rebuilt the venv on 3.13,
  which is fully supported. No functional impact.

- **Switched venv/deps from pip to `uv`**, per the local Python tooling rule
  (`~/.claude/rules/python.md`). No functional impact, just tooling.

- **Fireworks dropped from the pilot.** Its current serverless `/models`
  catalog (checked via `discover.py`, 20 models total) carries no Llama or
  Gemma model of any size, it's shifted to GLM, Kimi, MiniMax, DeepSeek-V4,
  Qwen3, GPT-OSS, Nemotron. This isn't a bug in our code. The catalog has
  moved on since the original paper (arXiv:2505.21627, May 2025) was written.
  Pilot proceeds with 3 resellers (Together, DeepInfra, OpenRouter) instead
  of 4. Fireworks could be revisited later against a *current* model family
  if we want a 4th data point, but that would no longer be the "same models
  as the original paper" comparison.

- **DeepInfra and OpenRouter don't carry Gemma-3-1B-it.** Smallest Gemma-3
  available on either is `google/gemma-3-4b-it`. Substituted 4B for both.
  Tokenizer is the exact 4B repo's own tokenizer (not the 1B's), no
  cross-size tokenizer mismatch.

- **DeepInfra doesn't carry any Llama-3.2 (1B or 3B).** Smallest Llama on
  DeepInfra is `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`. Substituted 8B
  for DeepInfra's Llama cell. "Turbo" denotes DeepInfra's quantized/optimized
  serving branding, not a different base checkpoint, using the tokenizer
  from `meta-llama/Meta-Llama-3.1-8B-Instruct` (unquantized repo) on the
  assumption that quantization does not change the vocabulary/tokenizer.
  **To verify during calibration pass**, not just assumed.

  Net effect: Together is the only reseller with an exact match on both
  original-paper models (Llama-3.2-1B-Instruct, Gemma-3-1B-it). DeepInfra and
  OpenRouter are same-family, different-size substitutes. This is fine for
  the pilot's actual question (does billed diverge from canonical?) but means
  cross-reseller comparisons at matched model size are only clean on Together
  alone. DeepInfra/OpenRouter results should be read as "this reseller, this
  model" rather than pooled with Together's numbers.

- **Together dropped entirely, not just a catalog mismatch, a real serving-tier
  finding.** `discover.py`'s catalog check (model id present in `/models`)
  passed for both target models, but a live smoke-test request 400'd:
  `"Unable to access non-serverless model meta-llama/Llama-3.2-1B-Instruct.
  ... create a dedicated endpoint."` Investigated further: the catalog's
  `running` field is not a servability indicator (0 of 273 listed models
  have `running: true` at any given snapshot, it appears to mean something
  like "currently mid-request," not "available"). Probed 7 small Llama/Gemma
  candidates on Together directly with tiny (`max_tokens=1`) real
  completions. All 7 failed the same way, including `gemma-2-9b-it` failing
  differently ("dedicated endpoint ... is not running"). Conclusion: Together
  has moved this entire size class off shared pay-per-token serverless onto
  dedicated (hourly-billed) endpoints only. This is the same pattern as the
  Fireworks finding, independently confirmed on a second reseller, real
  evidence the cheap-serverless market for small open-weight models has
  contracted since the original paper (May 2025). Worth a line in the
  eventual writeup on its own.

  **Lesson for `discover.py`:** catalog presence is not proof of servability
  on any of these resellers. The only reliable check is a live tiny
  completion. `discover.py` was NOT rewritten to do this automatically (yet)
  because DeepInfra and OpenRouter were both confirmed working via manual
  probes, if we add more resellers later, upgrade the pre-flight check to
  fire one `max_tokens=1` request per configured cell instead of trusting
  `/models` alone.

- **All 4 resellers restored, at the cost of a uniform model lineup.** User
  funded all 4 accounts and wants all 4 used, correctly. Rather than force
  Together/Fireworks into the Llama-3.2/Gemma-3-1B mold that they no longer
  serve, found real substitutes confirmed live via tiny completion probes:
  - Together: `Llama-3.3-70B-Instruct-Turbo` + `gemma-4-31B-it` (both
    confirmed on real shared serverless, unlike every small candidate).
  - Fireworks: `gpt-oss-20b` + `gpt-oss-120b` (OpenAI's open-weight release
    -- not Llama/Gemma family at all, but genuinely served here with fully
    public, ungated tokenizers on HF).

  Net effect: 4 providers x 2 models = 8 cells, but the model lineup is no
  longer uniform across providers (Together/Fireworks use much larger,
  differently-priced, differently-familied models than DeepInfra/OpenRouter).
  This does NOT weaken the core audit logic -- billed-vs-canonical-tokenizer
  comparison works identically regardless of model choice, as long as the
  tokenizer is public and the reseller genuinely bills pay-per-token for it,
  which is now verified true for all 8 cells via live probes, not catalog
  listings. It does mean: don't pool deltas across providers as if they're
  the same experiment: report and reason about each (provider, model) cell
  on its own terms, exactly as analyze.py already does.

- **One more gated repo to clear:** `meta-llama/Llama-3.3-70B-Instruct`
  (Together's llama tokenizer). `gemma-4-31B-it`, `gpt-oss-20b`, and
  `gpt-oss-120b` are all confirmed ungated (200 on tokenizer_config.json
  with no auth) -- no extra HF steps needed for those three.

- **OpenRouter routes `google/gemma-3-4b-it` to a DeepInfra backend.** A live
  probe's response included `"provider": "DeepInfra"`. This means the
  OpenRouter-gemma and DeepInfra-gemma cells may not be independent samples
, they could be hitting the same underlying infrastructure through two
  different billing layers, which is actually a useful cross-check (does
  OpenRouter's billed count match DeepInfra's for what might be the same
  actual computation?) but must not be read as two independent samples when
  computing pooled statistics. `audit.py` records
  `openrouter_generation_stats.provider_name` per-request specifically so
  this can be checked, not assumed, for every individual response.

- **Reasoning-field extraction bug, found and fixed during the smoke test.**
  Together's `gemma-4-31B-it` and Fireworks' `gpt-oss-20b`/`gpt-oss-120b` are
  reasoning-capable and return hidden chain-of-thought in a field separate
  from `content` -- field name varies (`reasoning` on Together, actually
  `reasoning_content` on Fireworks). `providers.py` originally only read
  `content`, so canonical recomputation missed all the reasoning text the
  provider actually billed for, producing enormous false deltas (up to
  +300) that looked like massive overcharging but were purely an extraction
  bug on our end. Fixed by concatenating `reasoning (or reasoning_content) +
  content` before canonical tokenization -- deltas dropped from the 50-300
  range to single digits immediately after the fix, on the same models,
  same prompts. This is exactly the kind of "verify before treating delta as
  signal" failure mode flagged before any of this started. Glad it surfaced
  on 5-prompt smoke tests instead of after a paid 200-prompt run.

  One of the gpt-oss-20b probes hit `finish_reason: "length"` with zero
  visible `content` at all -- the entire 300-token budget went to reasoning.
  Worth remembering for the real run: `max_tokens=300` may be too tight for
  these reasoning models to ever produce a visible answer on harder prompts.

## Calibration, done properly (2026-08-04)

User pushed back hard on treating deltas as signal without understanding
them first ("otherwise we're shooting ourselves in the foot"). Correct call.
Investigated every source of nonzero delta against real data, not
assumption, before allowing any real run:

1. **The universal +1 on non-reasoning models is the model's own EOS token.**
   Verified three ways on real records (DeepInfra-llama, DeepInfra-gemma,
   Together-llama): `encode(text, add_special_tokens=True)` and
   `encode(text + eos_token, add_special_tokens=False)` both reproduce the
   billed count exactly. The EOS token (`<|eot_id|>` for Llama, `<eos>` for
   Gemma) is real generated output marking end-of-turn, correctly billed by
   the provider -- our original "canonical" count was wrong to exclude it,
   not the provider's billing. Fix: only append `eos_token` when
   `finish_reason == "stop"` (a truncated response never emitted one, and
   appending it anyway would introduce a new, wrong +1). Requires capturing
   `finish_reason`, which `providers.py` didn't do before -- added.

2. **The lone -1 case (DeepInfra-llama, `finish_reason=length`) is ordinary
   tokenization-boundary noise, not bias.** Response text was degenerate
   temperature-1.3 output ("...WAIT double WA tardinclude death Assert io
   encalcul tape Fond"). Re-encoding a decoded string doesn't always
   reproduce the same token count as the original sampling path at unusual
   BPE boundaries -- this is the same "tokenization multiplicity"
   phenomenon the whole literature studies (see Chatzi et al. in the
   related-work review). 1 of 9 samples, magnitude 1, and in the direction
   of undercharging if anything. Not something to correct for. A real,
   bounded, already-understood noise floor.

3. **Reasoning-capable models bill for hidden chain-of-thought in a field
   separate from `content`.** Together's `gemma-4-31B-it` returns it under
   `reasoning`. Fireworks' `gpt-oss-*` under `reasoning_content` (OpenAI's
   "Harmony" format). Missing this made faithful providers look like they
   were overcharging by up to 300 tokens per response -- entirely an
   extraction bug on our end, not evidence of anything. Fixed by
   reconstructing the real generated stream per model family
   (`canonical_tokenizer.reconstruct_full_text`, keyed by a `reasoning_format`
   config field per model):
   - `gpt-oss` / Harmony: real structural tokens exist as single vocab
     entries (`<|start|>`, `<|channel|>`, `<|message|>`, `<|end|>`,
     confirmed via `tokenizer.get_vocab()`, not guessed). Reconstructed
     `<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
     <|start|>assistant<|channel|>final<|message|>{text}`. Closed the delta
     from +9/+10 down to a **stable constant -2** across 6/8 fresh samples
     (0 on the other 2) -- small, bounded, doesn't scale with response
     length. Not chased further. Diminishing returns past this point and the
     residual is safely in the conservative direction.
   - **Bug found and fixed during verification**: when a response is
     truncated *while still inside the reasoning channel* (`content` empty,
     `finish_reason=length`), the model never actually reached the "final"
     channel, so the transition tokens were never generated. The first version
     of the harmony fix added them unconditionally, which overshot by ~9
     tokens in exactly the opposite direction (canonical > billed). Confirmed
     by checking `response_text == ''` on the overshooting records. Fixed:
     only include the channel-transition when `text` is non-empty.
   - `google/gemma-4-31B-it` (Together): found one real structural token
     (`<|think|>`, vocab id 98) that closes part of the gap (delta 5→4), but
     **no documented chat template exists** for this repo to pin down the
     rest. Rather than guess and let a plausible-looking reconstruction
     become ground truth by accident, marked `reasoning_format: "unresolved"`
     in config and made `audit.py` skip this cell entirely until it's
     actually solved. This is the one cell NOT included in any real run yet.

4. **OpenRouter's Llama-3.2-1B-Instruct tokenizer access got stuck in
   Meta's manual review** (unlike Llama-3.3-70B-Instruct and Gemma-3-4B-it,
   which were instant grants once the HF token's gated-repo scope was fixed).
   User has hit this with Meta before and didn't want to keep waiting.
   Swapped to `unsloth/Llama-3.2-1B-Instruct`, a fully public re-upload of
   the same checkpoint -- confirmed ungated via direct HTTP check. Not
   byte-verified against the original (still inaccessible to compare
   directly), but Unsloth's fine-tuning tooling business depends on
   tokenizer fidelity to these exact checkpoints, so this is a reasonable,
   documented assumption rather than a silent one. Also worth noting:
   OpenRouter was never actually a blocked *provider* -- its gemma cell was
   live the whole time. This tokenizer was only needed to fill out the
   symmetric 2-model grid, not because OpenRouter itself was inaccessible.

**State after all of the above**, on fresh validation data with the fixed
code (not stale pre-fix records): DeepInfra-llama, DeepInfra-gemma,
OpenRouter-llama, OpenRouter-gemma, Together-llama all show exact or
near-exact matches, explainable residual only. Fireworks gpt-oss-20b/120b
show a stable, small, constant -2.

**Together-gemma resolved too (was: excluded).** User asked whether any of
this was documented online rather than reverse-engineered blind. It was:
official Google docs (ai.google.dev/gemma/docs/capabilities/thinking) give
the exact format -- `<|channel>thought\n{reasoning}\n<channel|>{text}`
(note the asymmetric bracket placement, confirmed correct, not a typo:
opening is `<|channel>`, closing is `<channel|>`). Verified both tokens are
real single vocab entries (ids 100 and 101) before trusting the docs
example, then tested the exact reconstruction (with the real newlines the
docs example shows) against saved real responses: 3 of 4 came back as an
**exact match (delta=0)**, the 4th (max_tokens-truncated) at -1. A follow-up
fresh run with proper finish_reason capture confirmed it: 4/5 succeeded,
delta -1 on the non-truncated ones (same small tokenization-noise class as
everywhere else), exact 0 on the truncated one. Same empty-content
truncation guard applied as the Harmony fix (if the model never reached the
closing marker, don't add it). `reasoning_format` in config.py changed from
`"unresolved"` to `"gemma_thinking"`. The cell is back in.

Lesson worth keeping: the OpenAI Harmony format is officially, thoroughly
documented (openai/harmony on GitHub, developers.openai.com cookbook), so
that reconstruction was really "read the spec," not reverse-engineering.
Gemma-4's thinking format is real and documented too, just less prominently
surfaced -- there's an active HN/HF discussion literally titled "Chat
template is too complicated that even Gemma 4 itself has no idea how to
parse it," which is exactly why `AutoTokenizer.from_pretrained` returned
`chat_template=None` for this repo. Good general habit going forward:
before reverse-engineering a token-format mystery from vocab inspection
alone, search for the model's official docs and HF discussion threads
first -- would have saved the blind `<|think|>`-alone guess.

**All 7 cells are now genuinely calibrated. None excluded.**

## Full study run (2026-08-04/05)

Ran all 8 cells at n=200 (4 providers x 2 models). Hit real infrastructure
friction along the way, all logged and worked through rather than papered
over:

- Background task runner killed long-running processes repeatedly and
  unpredictably -- not cleanly explained by concurrency count or wall-clock
  duration alone (tried 4-way parallel, 2-way, and solo. Solo got killed
  too on the slowest cell). Together's `gemma-4-31B-it` in particular has
  extremely variable per-request latency (observed single-request gaps up
  to 6+ minutes). Worked around by (a) adding a `max_new_requests` chunk
  limit to `audit.py` so a single call exits cleanly instead of being cut
  off mid-flight, and (b) for the worst cell, launching as a fully detached
  OS-level daemon (`nohup ... & disown`) outside the harness's tracked
  background-task system entirely, polling the output file directly.
- Fixed a real resume bug: the first resume implementation matched by
  position/count ("skip the first N prompts"), which is wrong when
  failures are scattered through the sequence rather than clustered at the
  end (confirmed: a batch with 9 failures had them spread throughout, not
  at the tail). Rewrote to match by actual prompt content
  (`_existing_prompts`), so resuming never skips a genuine failure or
  silently produces incomplete coverage. Also deduped result files where
  the old buggy resume logic had already produced ~14 redundant repeats
  before the fix landed.

### OpenRouter-llama fully resolved, not just flagged

At n=200 this cell showed real outlier variance (delta range -9 to +3,
mean -1.79) versus every other cell's tight ±0-2 spread. Investigated
properly rather than accepted as an unexplained caveat:

1. **Ruled out multi-backend routing.** Queried OpenRouter's `/models/.../endpoints`
   for `meta-llama/llama-3.2-1b-instruct` -- only one backend (Cloudflare)
   serves this model at all. The variance isn't routing noise.
2. **Found a real backend bug via pinned re-testing.** Pinned requests to
   `provider: {order: ["cloudflare"], allow_fallbacks: false}` and ran a
   fresh 20-sample batch. Every single response that hit exactly
   `max_tokens=300` was labeled `finish_reason: "stop"` -- implausible as
   genuine coincidence at that frequency (a small model naturally finishing
   at exactly the configured cap, repeatedly). Conclusion: Cloudflare
   mislabels truncated generations as "stop" instead of "length" for this
   model. Our EOS-conditional canonical logic trusted that label, so it was
   wrongly appending a phantom EOS token on every truncated response.
   **Fix**: `canonical_tokenizer.effective_finish_reason()` overrides the
   reported label to "length" whenever `billed_completion_tokens ==
   max_tokens` and the report says "stop" -- applied for all providers
   defensively (nothing rules out the same bug elsewhere), not just
   Cloudflare. Centralized into one function used by `audit.py`,
   `recompute_analysis.py`, and `power_and_shape_analysis.py` so the three
   scripts can't drift out of sync on this logic. Closed exactly 1
   token/case as expected. Mean delta improved -1.79 -> -1.19, worst case
   -9 -> -8.
3. **Explained the remaining residual by direct inspection, not
   inference.** Pulled the actual text of the worst remaining case:
   completely degenerate, multi-script garbage ("...сообщ...广...略...spielen
   damages ?\n\nYoui"). This is Llama-3.2-1B-Instruct (the smallest, weakest
   model in the entire study) at temperature 1.3 pushed to the full 300-token
   cap -- the single worst-case regime in the whole design for model
   coherence. Re-tokenizing heavily degenerate, mixed-script text is exactly
   where ordinary tokenization-boundary ambiguity is largest (matches Chatzi
   et al.'s published finding that this concentrates in non-English/mixed
   content, cited in the related-work review). Not a third unexplained
   mechanism -- the same "tokenization multiplicity" noise already
   identified elsewhere, just larger in magnitude because this cell pairs
   the weakest model with the most adversarial sampling settings.

Net: OpenRouter-llama's noise is now fully attributed to two identified,
non-fraud causes (a backend metadata bug + expected tokenization noise on
degenerate output), not left as an open question. And critically: even
unresolved, the bias ran negative in every single sample -- toward
undercharging, never toward the inflation the original paper's threat
model predicts.

### Power analysis (see power_and_shape_analysis.py)

Computed minimum detectable effect (one-sided, alpha=0.05, power=0.80) per
cell. Range: 0.01%-0.20% of typical response length. Even the weakest cell
(OpenRouter-llama at 0.20%) has more statistical power than the original
paper's own lowest reported realistic inflation rate (0.28% at p=0.90 for
Llama-3.2-1B-Instruct, their Figure 2). We are not merely "not finding
anything" -- we can bound the undetected effect size below their own
smallest claimed number, in 7 of 8 cells comfortably so.

### Bimodality check -- and an honest caveat about the diagnostic itself

Computed Sarle's bimodality coefficient per cell as a check for selective
(Algorithm-2-style) padding, which would show as a two-cluster mixture
(honest + padded) rather than single-mode noise. Several cells returned BC
values above the textbook 0.555 "possibly bimodal" threshold (up to 0.98).
**This is a known artifact, not evidence of anything**: Sarle's BC is
calibrated for reasonably continuous distributions and behaves erratically
on heavily discrete, near-degenerate point masses like ours (e.g.
together/llama is 199 exact zeros and 1 outlier -- the formula spikes on
this shape even though no reasonable reading calls it "bimodal"). The
substantively meaningful check is the histogram itself: every cell shows
one dominant mode with a smoothly-decaying scatter around it, never a
second concentration of mass at some other value. Reporting the BC number
honestly alongside this explanation rather than omitting it, since a
hostile reader would otherwise find and weaponize it first.

## Open items before spending real money (audit.py)

- **HF_TOKEN missing from `.env`.** Canonical tokenization requires
  downloading each model's real tokenizer from Hugging Face. Both
  `meta-llama/Llama-3.2-1B-Instruct` and `google/gemma-3-1b-it` (and their
  larger siblings used as substitutes) are gated repos, confirmed via a
  live 403 GatedRepoError, not an assumption. Needs, on the same HF account:
  1. Accept the Meta Llama 3.2 Community License on the model page.
  2. Accept the Gemma Terms of Use on the model page.
  3. Generate a read-scoped access token under Settings → Access Tokens.
  4. Add `HF_TOKEN=...` to the project `.env`.

- **Calibration pass not yet run.** Before trusting any billed-vs-canonical
  delta as signal, need to confirm on a handful of real responses per
  provider whether `usage.completion_tokens` includes any special/boundary
  tokens (EOS, etc.) that wouldn't appear in `tokenizer.encode(text,
  add_special_tokens=False)`. A constant per-request offset (e.g. always
  +1) would indicate a formatting artifact, not overcharging, and needs to
  be netted out before analysis.

- **Prompt dataset not yet finalized.** Plan is to reuse a subset of the
  original paper's LMSYS Chatbot Arena prompts (English, length [20,100]
  chars) if `lmsys/lmsys-chat-1m` access is available on the same HF account
  (it's gated too). Otherwise fall back to an ungated instruction dataset
  (e.g. Dolly) with the same filtering, and note the substitution here.

## Z.ai (native GLM API) added (2026-08-06)

User has an existing Z.ai subscription (`~/.zai`, read directly by
`env.ensure_loaded()` -- never copied into `.env` or displayed). Added as a
5th provider: `https://api.z.ai/api/paas/v4` (confirmed via docs.z.ai, not
assumed), model `glm-4.6`. This is meaningfully different from the 4
resellers: it's the model creator's own first-party billing, not a
third party reselling someone else's weights.

- **Temperature capped at 1.0** -- confirmed live (400: "The temperature
  parameter is illegal", code 1210) when sending our standard 1.3. Added
  `GENERATION_OVERRIDES` in config.py, merged per-provider in audit.py,
  rather than lowering the global setting for everyone.
- **New reasoning format, verified before use, not assumed.** GLM-4.6
  returns `reasoning_content` (already handled by existing field-detection
  logic) and discloses `reasoning_tokens` separately in
  `completion_tokens_details`, same convention as OpenAI/OpenRouter. No
  `chat_template` on the HF tokenizer (same situation as Gemma-4). Full
  vocab search found a real `<think>`/`</think>` pair (ids 151350/151351).
  Tested `<think>{reasoning}</think>{text}` against two real responses:
  exact match (delta=0) on a natural completion, -1 (ordinary bounded
  noise) on a max_tokens-truncated one. Notably, appending `eos_token`
  (`<|endoftext|>`) on top of the natural-completion case *overshot* by 1 --
  unlike Llama/Gemma, GLM's disclosed `eos_token` is evidently not the
  actual per-turn billing terminator. New `reasoning_format:
  "glm_thinking"` skips the EOS-append step entirely (added
  `NO_EOS_APPEND_FORMATS` set in canonical_tokenizer.py rather than special
  casing this in the shared function).
- **Smoke test (n=5) clean**: 4/5 exact match, 1 at -1 (truncated, same
  noise class documented everywhere else in this project). Ready for a
  real batch.

## Open: DeepSeek/Kimi/GLM extension across existing resellers

User asked about extending Tier-1 to more open-weight model families
(DeepSeek, Kimi, GLM/Zhipu) already visible in Fireworks/DeepInfra's
catalogs from earlier discovery. Verified live servability across all 4
existing accounts:

- Together: Kimi-K2.6 OK. DeepSeek-V4-Flash-0731 OK (after trying V3.1,
  R1-0528, and a distill, all 400 -- same dedicated-endpoint pattern as
  before). No GLM variant works (dedicated-endpoint 400s and one
  persistent 503 tried).
- Fireworks: deepseek-v4-flash, kimi-k2p6, glm-5p2 all OK.
- DeepInfra: DeepSeek-V3.1, Kimi-K2.6, GLM-4.6 all OK.
- OpenRouter: deepseek-chat-v3.1, kimi-k2, glm-4.6 all OK.

11 verified-servable cells, not yet built into PROVIDER_MODEL_MAP or run --
paused pending user decision on scope (full parity at n=200/both sampling
settings vs. a lighter first pass) given the potential added wall-clock
time on top of an already-long session. Each new model will need its own
reasoning-format verification against real data before trusting any
numbers, same as every format in this file so far -- do not assume any of
these behave like Llama/Gemma/gpt-oss/Gemma-4/GLM without checking.

## Session recovery + expansion (2026-08-07)

A session crash led to the belief that work had been lost. It had not: the
project had been moved to `MPI/token-billing-audit/` earlier at the user's
request, and the restored `MPI/breaking-overcharging/` directory was a
pre-move husk containing only a stale `.venv` and `__pycache__`. Verified
every file and all 3200 collected records intact, confirmed the pipeline
reproduced end to end, copied the session transcript in as
`session-transcript-2026-08-04.txt`, and trashed the husk.

### Z.ai native GLM run completed

n=200, 0 errors. Mean delta -0.93, range [-1, 0], **zero positive deltas**.
But see the truncation caveat below: only 13 of those 200 are informative,
because GLM-4.6 reasons heavily and 187 responses hit the 300-token cap.

### Methodological gap found and closed: capped responses carry no evidence

A response billed at exactly `max_tokens` was truncated, so its billed count
is pinned to the cap and *cannot exceed it no matter what the provider does*.
Such samples are structurally incapable of exhibiting inflation, in either
direction. Reporting them inside an "n=200 per cell" headline overstates the
informative sample, and a hostile reader would find this immediately.

Informative fractions in the existing study turned out to range from 6%
(zai/glm) to 84% (deepinfra/gemma). `final_report.py` now reports `n_inf`
alongside `n` and computes all statistics (including MDE) on the informative
subset only. `audit.py` now records `max_tokens` per request so this can never
be mis-inferred later, and expansion cells run at a raised cap
(`EXPANSION_MAX_TOKENS = 1200`) so heavy reasoning models finish naturally
instead of truncating mid-thought.

Note this correction makes the result *stronger*, not weaker: restricted to
informative samples, five of nine cells show a delta of exactly zero on
100.0% of requests.

### Two long-standing constant residuals fully closed

Both previously-documented "small, bounded, unexplained" offsets turned out to
be errors in our own reconstruction, and both are now exact:

1. **Harmony (Fireworks gpt-oss) constant -2 -> exactly 0.** Bracketed the
   opening prefix against real informative samples:
   `<|start|>assistant<|channel|>...` (2-token prefix) overshoots by exactly 1,
   a bare `<|channel|>...` (0-token prefix) undershoots by exactly 1. A
   1-token prefix gives **20/20 exact matches on both gpt-oss-20b and
   gpt-oss-120b**. Honest caveat recorded in code: `<|start|>` and `assistant`
   are each exactly one token, so the two candidate 1-token readings are
   count-identical and token counting cannot distinguish them -- no number
   here depends on which is correct.
2. **Reasoning formats do not bill a separate EOS.** The closing structural
   token (`</think>`, `<channel|>`, Harmony's final-channel message) already
   terminates the turn. Appending `eos_token` on top double-counts. Verified
   per format on informative samples: gemma_thinking 20/20 exact without EOS
   (constant -1 with it), harmony 20/20 exact without EOS (constant -2 with
   it). Plain non-reasoning ("simple") models are the opposite and DO bill
   their own EOS -- that is what the original +1 calibration established.

### Expansion cells calibrated from data, not assumption

Added DeepSeek-V4-Flash-0731, Kimi-K2.6 and GLM (4.6 / 5.2) across the four
funded reseller accounts -- 11 servable cells (Together carries no serverless
GLM. Same dedicated-endpoint wall documented earlier, reconfirmed).

`calibrate_cells.py` was written to make this rigorous rather than assumed: it
sends real requests per cell and scores four candidate reconstructions
(simple/think_tags x with/without EOS) against the provider's own
`completion_tokens`, and the winner is what goes into config.

**This mattered.** The same checkpoint is billed under different conventions
by different resellers:

- Kimi-K2.6: Fireworks, Together and OpenRouter all bill it WITH think-tags,
  DeepInfra bills the same checkpoint with **no think-tags and no EOS**.
- GLM-4.6: DeepInfra and OpenRouter bill it with think-tags **plus** EOS,
  Fireworks' GLM-5.2 bills without EOS.
- DeepSeek-V4-Flash-0731: DeepInfra returns **no reasoning field at all** and
  bills visible content + EOS exactly (6/6 exact) -- i.e. no hidden reasoning
  tokens are being billed there.

Assuming a format transfers across providers would have manufactured phantom
multi-token deltas and, on the DeepInfra/Kimi cell, a -2/response "finding"
that is purely our own error. `append_eos` is therefore a per-cell config
override, not a per-format constant.

### Supporting fixes

- **`providers.py` sent `{"role": "system", "content": null}`** when no system
  prompt was given. Together 400s and DeepInfra 422s on that rather than
  ignoring it, which reads as "model unavailable" and is easy to misdiagnose
  (it briefly looked like 6 cells were dead). The system turn is now omitted
  when empty. No collected data is affected -- the study always passed a real
  string.
- **Tokenizer loading**: three repos don't load via plain
  `AutoTokenizer.from_pretrained` under the pinned transformers 4.46.3 --
  GLM-5.2 (unknown `TokenizersBackend` class), DeepSeek-V4-Flash (would pull
  remote modeling code), Kimi-K2.6 (the **dot** in the repo name breaks the
  `transformers_modules` Python module path). Rather than upgrade transformers
  mid-study -- which could silently shift the canonical counts underpinning
  3200 already-collected requests -- each exception gets an explicit,
  documented loader strategy in `canonical_tokenizer.TOKENIZER_LOADERS`.
  Added `tiktoken`/`blobfile` to requirements (pinned to actually-installed
  versions).
- **Kimi's reasoning format was read from its own published
  `chat_template.jinja`** (`<think>{{reasoning}}</think>{{content}}`), not
  reverse-engineered -- following the lesson recorded earlier in this file
  about checking official docs before guessing from vocab inspection.
- **`final_report.py` no longer prints a bare "any positive mean" boolean.**
  It fired on a +0.005 mean driven by a single +1 sample in 200, which
  overstates noise as signal in the direction least favorable to our own
  argument. It now compares each cell's mean against that cell's own minimum
  detectable effect and says so explicitly.

### Expansion run results (2026-08-07)

Launched all 4 providers as detached daemons. Together (400/400), Fireworks
(600/600) and DeepInfra (600/600) completed with **zero errors**.

**OpenRouter ran out of funds**: 345 of 600 requests failed with HTTP 402
Payment Required. Its `/api/v1/credits` endpoint reports
`total_credits: 0, total_usage: $0.19`, i.e. the account has no purchased
credit balance, not a balance that was drained by this run. The earlier
main-study and p=0.99 runs succeeded there because they used very cheap small
models ($0.023 total across 800 requests). The expansion cells use Kimi-K2.6
($4.50/M output) and GLM, which need real credit. Completed on OpenRouter:
deepseek 200/200, kimi 54/200, glm 1/200. Those two partial cells are reported
with their true n and NOT pooled up to look complete.

Results across the expansion block: 1855 requests, 1397 informative (75% --
much better than the main study's 47%, which is exactly what raising
max_tokens to 1200 was for). No cell shows a mean overcharge exceeding its own
MDE. Cell means range from -0.26 to +0.035 tokens/response.

Interrogated every positive-direction (overcharge-direction) sample rather
than reporting only aggregates: **25 of 1397 informative samples (1.8%) are
positive, every single one by 1 or 2 tokens**, against a comparable
undercharging tail. That is a symmetric ±1-token noise floor, consistent with
ordinary tokenization-boundary ambiguity, and it is not what the paper's
threat model predicts, a systematic 0.28%-11.2% inflation would be roughly
1 to 33 extra tokens on *every* response, not 1-2 tokens on 1.8% of them with
an equal-and-opposite tail.

Cost: the whole expansion cost well under the funded balances on the three
providers that completed. Precise per-request cost is recorded for OpenRouter
(it returns a cost field). The others do not return one.

## Tier 2: closed / frontier API accounting (2026-08-07)

Scope clarified by the user: Tier 2 is not only about flagging suspicious
billing, it is about what a provider could ship to *prove* it is not doing
this. Two corrections to the earlier scoping came out of actually checking
rather than assuming:

1. **OpenAI's tokenizer is public.** `tiktoken` ships `o200k_base`, which
   covers GPT-4o, GPT-4.1, GPT-5 and o3. The earlier note that closed models
   have "no public tokenizer, no ground truth" is wrong for OpenAI. It holds
   only for Anthropic and Gemini's proprietary models.
2. **Google serves an open-weight model on its own first-party API.**
   `gemma-4-31b-it` is available through AI Studio, and we already have its
   public tokenizer in this study (Together serves the same checkpoint). That
   gives real Tier-1 ground truth on a frontier lab's own billing, the same
   trick that made the Z.ai cell valuable. OpenAI does NOT do this -- no
   `gpt-oss` entry exists on its first-party API.

### The reconciliation test (`closed_api_accounting.py`)

A check that needs no tokenizer and that no framework in this literature
performs: **do the provider's own published usage numbers add up?** Plus, for
OpenAI, `len(logprobs.content)` vs `completion_tokens` -- one logprob entry per
generated token, so a mismatch is an internal contradiction in the provider's
own response.

Results (n=8 prompts per target, max_tokens=800):

| target | reconciles | median unexplained | reasoning field | logprob check |
|---|---|---|---|---|
| openai/gpt-4o-mini | 8/8 | 0 | present, zero | **8/8 match** |
| openai/gpt-5-nano | 8/8 | 0 | present, nonzero | n/a |
| google-native/gemma-4-31b-it | 7/7 | 0 | present, nonzero | n/a |
| google-native/gemini-2.5-flash | 7/7 | 0 | present, nonzero | n/a |
| **google-compat/gemma-4-31b-it** | **0/8** | **195** | **absent** | n/a |
| anthropic/claude-haiku-4-5 | n/a (publishes no total) | n/a | absent | n/a |

**Methodological point that had to be fixed before any of this was
trustworthy:** reasoning-token placement differs by provider. OpenAI nests
`reasoning_tokens` INSIDE `completion_tokens`. Gemini reports
`thoughtsTokenCount` ALONGSIDE `candidatesTokenCount`. A first version of the
script used one formula for both and reported Google's fully-transparent
native API as reconciling 0/7 -- i.e. it made an honest, fully-disclosing
provider look like it was hiding ~300 tokens/response. Corrected to per-
provider semantics. The native API reconciles exactly.

### The actual finding: Google's OpenAI-compatibility layer

Google's **native** API is fully transparent -- `promptTokenCount +
candidatesTokenCount + thoughtsTokenCount == totalTokenCount` on every sample.

Google's **OpenAI-compatible** endpoint, serving the same model, is not:
`prompt_tokens + completion_tokens` never equals `total_tokens` (0/8), with a
median of ~195 tokens/response unaccounted for, and
`completion_tokens_details.reasoning_tokens` -- the field OpenAI's schema
defines for exactly this, and which Google demonstrably computes internally
since the native endpoint returns it -- is left absent.

This is **not** fraud: the tokens appear in `total_tokens`, and the native API
discloses everything. But it means a customer cannot reconcile a bill from
`prompt + completion` on that surface, and, more importantly, **any auditing
tool built against the OpenAI-compatible interface -- the industry norm, and
what CoIn/PALACE-style tools would target -- systematically mis-accounts
Google's billing.** It would read `completion_tokens` far below actual usage.

It also makes the "what should providers ship" argument concrete and cheap:
populate a field that already exists in the schema and whose value the
provider already computes. That is the entire fix.

### Known limitation, not yet closed

The Anthropic row is under-exercised: `claude-haiku-4-5` was queried without
extended thinking enabled, so no thinking tokens were generated and the
disclosure question is untested rather than answered. Anthropic also publishes
no combined total, so there is nothing to reconcile -- which is not a pass, it
is an absence of the surface the test needs. Re-run with extended thinking on
before making any claim about Anthropic.

## Tier-2 build-out results (2026-08-07, continued)

### OpenAI: complete, and verified two independent ways

`exp_openai.jsonl`, n=200 per model, gpt-4o-mini and gpt-4.1-nano.

| model | n | informative | tiktoken exact | logprob exact | mean delta |
|---|---|---|---|---|---|
| gpt-4.1-nano | 200 | 195 | **195/195** | **195/195** | 0.000 |
| gpt-4o-mini | 200 | 191 | **191/191** | **191/191** | 0.000 |

Two *independent* verification paths agree exactly on every informative
sample: recomputation against OpenAI's published `o200k_base` tokenizer, and
the logprobs entry count (one entry per generated token, which needs no
tokenizer at all and so does not depend on trusting the published one). No
EOS is billed. This is the strongest per-cell result in the study.

Reasoning models (o-series / GPT-5 reasoning) are deliberately EXCLUDED from
this cell: OpenAI discards reasoning content server-side, so the
reasoning-token share of the bill has no ground truth and does not belong in a
cell claiming verified counts. `gpt-5-nano` does disclose
`completion_tokens_details.reasoning_tokens` and reconciles 8/8 in the
accounting test, which is a weaker but real check.

### Correction: Anthropic DOES disclose reasoning tokens

An earlier row in this file recorded Anthropic's reasoning field as "absent".
That was an artifact of the probe: extended thinking must be explicitly
enabled, and without it the model does not think, so
`output_tokens_details.thinking_tokens` is legitimately missing. With
`thinking: {type: enabled, budget_tokens: 1024}` the field is present and
non-zero. The earlier reading understated Anthropic's transparency and is
withdrawn.

Anthropic self-consistency check (their own `count_tokens` endpoint, n=8):
`output_tokens - thinking_tokens` tracks `count_tokens` on the visible text
with a **constant** offset of -10 (one sample -11) across responses spanning
31 to 295 visible tokens. A constant offset is the fixed role/message wrapper
that `count_tokens` includes. What would matter is drift or a gap that grows
with length, and there is none. So: thinking tokens fully disclosed, nested
inside `output_tokens`, internally self-consistent. Note this is a
provider-internal check, NOT independent ground truth -- Anthropic publishes
no tokenizer and no combined total, so it remains the least externally
verifiable of the three, without anything contradictory in it.

API constraints worth recording: `budget_tokens` must be >= 1024 AND
`max_tokens` must exceed it, so this cell cannot run at the shared
MAX_TOKENS=800 used by the other targets.

### Two accounts ran dry mid-run

- **Z.ai balance exhausted at 83/200** on the max_tokens=1200 re-run. The API
  returns `HTTP 429 code 1113 "Insufficient balance or no resource package"`.
  Kept: 83 records at max_tokens=1200 (`exp_zai.jsonl`) plus the earlier
  complete 200 at max_tokens=300 (`full_zai.jsonl`).
- **Gemini free tier quota exceeded** for `gemini-2.5-flash`. The
  `gemma-4-31b-it` cell -- the one that actually matters, since it is the
  open-weight Tier-1 cell -- continued working.

**Real bug this exposed, now fixed.** `providers.py` treated *every* HTTP 429
as transient rate limiting and backed off exponentially. Z.ai's
insufficient-balance error is also a 429, so an exhausted account turned into
a silent multi-hour churn through the remaining prompt list that looked
exactly like a hung process (the daemon showed 40 minutes elapsed with 2
seconds of CPU and no new records). The client now inspects the response body
and raises immediately on billing/quota conditions (and on HTTP 402, which is
how OpenRouter signals the same thing) instead of retrying them. Fail fast
beats a hang that reads as progress.

### Google native Gemma: hidden reasoning-token billing verified to the token

`gnative_google.jsonl`, n=200, zero errors, **all 200 informative** (none hit
the 1200-token cap).

| channel | exact | mean delta | delta distribution | billed tokens |
|---|---|---|---|---|
| THOUGHT (hidden reasoning) | **194/200** | +0.015 | {-1:2, 0:194, +1:3, +2:1} | 83,838 |
| ANSWER (visible output) | **192/200** | -0.020 | {-1:6, 0:192, +1:2} | 81,533 |

Provider arithmetic (`promptTokenCount + candidatesTokenCount +
thoughtsTokenCount == totalTokenCount`) reconciles **200/200**.

Why this is the most important cell in the study: **50.7% of the billed output
tokens were hidden reasoning** -- tokens the customer never sees. That is the
exact channel every auditing framework in this literature treats as
unverifiable, and the channel that matters far more in 2026 than the
retokenization channel the original paper analyzes. Here it is verified
against a public tokenizer, on a frontier lab's own first-party API, to a mean
error of +0.015 tokens per response across 83,838 billed reasoning tokens.

The handful of +-1/+2 deltas are the same ordinary tokenization-boundary noise
documented throughout this file, symmetric across both channels (3 positive
and 2 negative on thoughts, 2 positive and 6 negative on answers).

### Final totals

**6,083 real billed requests across 22 distinct (provider, model) cells on 7
platforms** -- four resellers (Together, Fireworks, DeepInfra, OpenRouter) and
three first-party APIs (Z.ai, OpenAI, Google). Expansion block: 2,483
requests, 80% informative. No cell at any sampling setting shows a mean
overcharge exceeding its own minimum detectable effect.

## The one cell that flagged, and why it is not inflation (2026-08-07)

At p=0.99, `deepinfra/kimi` became the **only cell in the entire study** whose
mean delta exceeded its own minimum detectable effect: mean +0.573
tokens/response, 52% of informative samples positive (p=0.95 on the same cell:
+0.035, 4.1% positive).

Investigated rather than explained away, in this order:

1. **Delta distribution is a clean two-point split**, not a spread:
   `{-3:1, -1:1, 0:49, +1:61, +2:5}`. A mechanism that pads output would not
   produce "exactly zero or exactly one".
2. **Ruled out a reconstruction-format error.** Tested `simple`, newline
   separators, `<think>...</think>`, and closing-tag-only. The 49/61 split
   simply *flips* between `simple` (49 exact) and `reasoning + </think> + text`
   (61 exact) -- i.e. the provider bills one extra token on some responses and
   not others, and no single static format fits both groups.
3. **Ruled out a cross-boundary merge artifact.** Encoding
   `reasoning + text` as one string versus encoding each separately and
   summing gives *identical* results (both strings end/begin with spaces, so
   no token merges across the join). This hypothesis was wrong and is recorded
   as such.
4. **Compared the two groups directly** (content emptiness, leading/trailing
   newlines, reasoning and content lengths). No structural feature
   distinguishes them.
5. **The decisive test: does the delta scale with response length?**
   Proportional inflation must hold the *percentage* constant and grow the
   absolute token count. A fixed convention token must do the opposite.

   | billed tokens | n | mean delta | as % of billed |
   |---|---|---|---|
   | 0-249 | 23 | +0.565 | 0.452% |
   | 250-499 | 36 | +0.556 | 0.148% |
   | 500-749 | 34 | +0.559 | 0.089% |
   | 750-999 | 15 | +0.533 | 0.061% |
   | 1000-1249 | 9 | +0.778 | 0.069% |

   The absolute delta is flat at roughly half a token. The percentage decays
   as 1/length. Maximum delta anywhere in the cell: **2 tokens**. This is a
   per-response end-of-turn convention token billed on about half of Kimi
   responses at this sampling setting, not proportional inflation.

**Deliberately NOT fixed by recalibration.** A per-response `append_eos` could
be fitted to make this delta vanish, and that is exactly the error this
project exists to avoid: tuning the reconstruction until the residual
disappears turns any finding into a null result. The scaling test is the
honest dismissal, and it is a stronger argument than a tuned fit.

**What this cell actually demonstrates, and it is worth leading with:** the
method is precise enough that a *half-token per response* effect (0.06%-0.45%
depending on length) trips the detector -- and precise enough to then classify
it correctly as a bounded convention artifact rather than a mechanism. The
original paper's smallest claimed inflation rate is 0.28% *systematic*. On a
700-token response that is 2 to 78 extra tokens, scaling with length. What we
found is capped at 2 tokens total and shrinks proportionally as responses grow.
The two are not the same phenomenon and cannot be confused given this test.

## FINAL STATE (2026-08-07)

All runs complete. **8,592 real billed requests, 22 distinct (provider, model)
cells, 7 platforms**, four resale platforms (Together, Fireworks, DeepInfra,
OpenRouter) and three first-party APIs (Z.ai, OpenAI, Google).

Blocks:

| block | requests | informative | result |
|---|---|---|---|
| main study, p=0.95 | 1,800 | 847 (47%) | no cell above MDE |
| main study, p=0.99 | 1,600 | 799 (50%) | no cell above MDE |
| expansion, p=0.95 | 2,600 | 2,074 (80%) | no cell above MDE |
| expansion, p=0.99 | 2,391 | 1,836 (77%) | one cell above MDE, dismissed by scaling test |
| Google native (hidden reasoning) | 200 | 200 (100%) | 194/200 exact on the reasoning channel |

Cells at exactly 100.0% zero delta on informative samples: deepinfra/gemma,
fireworks/gpt-oss-20b, fireworks/gpt-oss-120b, together/gemma, together/llama
(p99), together/deepseek, openai/gpt-4.1-nano (both settings), openai/gpt-4o-mini,
zai/glm (p95 old cap).

Z.ai re-run at the raised cap finished 200/200: **85% informative, up from 6%**
at the old 300-token cap, 168/170 exact, mean +0.012. The methodological fix
did exactly what it was meant to.

Deliverables in this repository:

- `the-case-against.md`, the argument (structural + empirical).
- `verifiable-billing-spec.md`, the Tier-2b artifact: what a provider ships
  to *prove* it is not overcharging, in four ascending levels, with the
  reasoning why disclosing token boundaries alone is insufficient (the
  attack produces a valid alternative segmentation of the same string, so the
  canonical segmentation must be checkable, which requires publishing the
  tokenizer).
- `implementation-notes.md`, this file: every deviation, bug, dead end and
  correction, including the ones where an earlier conclusion was withdrawn.
- Audit code: `audit.py`, `canonical_tokenizer.py`, `providers.py`,
  `config.py`, `prompts.py`, `calibrate_cells.py`, `probe_new_models.py`.
- Closed/frontier API work: `closed_api_accounting.py`, `google_native_audit.py`.
- Analysis: `final_report.py`, `power_and_shape_analysis.py`,
  `recompute_analysis.py`.
- Raw data: `results/*.jsonl` (every request, prompt, response, reasoning
  text, billed and recomputed counts).

Known limitations, stated rather than buried:

- OpenAI reasoning models (o-series / GPT-5 reasoning) are excluded from the
  verified Tier-1 cell: reasoning content is discarded server-side, so that
  share of the bill has no ground truth. Their accounting reconciles (8/8) but
  that is a weaker check.
- Anthropic remains the least externally verifiable of the three frontier
  labs: no published tokenizer, no combined total. Nothing contradictory was
  found, and thinking tokens are disclosed and internally self-consistent, but
  "self-consistent" is not "independently verified".
- Gemini's proprietary models (gemini-2.5-flash) were only lightly sampled
  before the free-tier quota cut in. The open-weight Gemma cell carries the
  Tier-1 weight for Google.
- `gemma-4-31b-it` on Google's native API is an open-weight model, so it does
  not prove anything about how Google bills its *proprietary* Gemini models.
  It proves the hidden-reasoning channel is checkable at all, and honest where
  checked.

## GPT-5.6 (2026-08-07)

The first OpenAI key had no access to anything past `gpt-5.2`, all four 5.6
names returned 404 "does not exist", which is an access-grant problem, not a
credits problem (the credits error is a distinct message on
`/chat/completions`). A second key with full org access resolved
`gpt-5.6-sol`, `-terra`, `-luna`. Note the bare `gpt-5.6` alias 404s even on
the working key. The explicit model names are required.

**Correction worth recording:** an earlier statement in this session that
"there is no gpt-5.6, latest is gpt-5.2" was wrong. GPT-5.6 launched
2026-07-09, after the assistant's knowledge cutoff, and the first key's model
listing appeared to corroborate the mistake. Checking the web settled it.
Lesson: an API model listing reflects *that account's entitlements*, not what
exists.

### Reasoning models need a different verification claim

The Tier-1 OpenAI cells (gpt-4o-mini, gpt-4.1-nano) are non-reasoning:
`billed == tiktoken(text) == len(logprobs)`, exactly, 386/386. GPT-5.x loses
BOTH of those checks, reasoning content is discarded server-side, and
`logprobs` is not returned at all for these models. What remains is an
accounting identity, calibrated live (16/16 consistent across sol and luna):

    billed == tiktoken(visible) + reasoning_tokens + K
    K = 3  when reasoning_tokens == 0    (final channel header only)
    K = 9  when reasoning_tokens  > 0    (analysis channel + transition)

Same Harmony structural-token class already verified exactly on gpt-oss, where
the reasoning text WAS available and the reconstruction matched 20/20.

### Results (n=200 each, 0 errors)

| model | n | informative | exact | mean Δ | reasoning tokens billed | hidden share |
|---|---|---|---|---|---|---|
| gpt-5.6-sol | 200 | 196 | **194/196** | +0.061 | 16,339 | 36.6% |
| gpt-5.6-luna | 200 | 194 | **191/194** | +0.098 | 17,741 | 35.1% |

**Roughly a third of the bill is reasoning tokens the customer never sees, and
the total is exactly accounted for on 385 of 390 informative samples.**

The 5 exceptions are all a constant +6/+7 (never scaling), all on responses
with high reasoning counts (639-1024) producing factual-lookup answers , 
consistent with an additional Harmony channel (a tool/commentary preamble) not
covered by the two-mode K. Bounded, discrete, 1.4% of samples. Not corrected
for, and reported as an unexplained residual rather than tuned away.

**What this establishes and what it does not.** It verifies the billed total is
fully accounted for by the provider's own disclosed reasoning count plus the
visible text plus fixed structure, a padded bill breaks the identity. It does
NOT verify that `reasoning_tokens` is itself truthful. That is the provider's
own claim about content nobody can see. Only an open-weight model returning
raw thoughts closes that gap, which is exactly what the Google Gemma cell does
(194/200 exact on the reasoning channel itself). Keep the distinction sharp:
**OpenAI is self-consistent. The Google open-weight cell is independently
verified.**

## Round 2 partial finding: billed tokens that cannot be reconstructed

DeepInfra's gpt-oss-20b cell showed a mean of +9.86, far out of line with the
same model on Together (+0.08). It is not billing behavior. The distribution is
`{0: 118, 1: 20, 2: 7, ...}` plus **three records at +441, +443, +655** which
carry almost the entire mean.

Reading those records explains them. The reasoning text says, variously:
"Oops the assistant messed up, it output wrong. We need to produce corrected
answer. Let's rewrite.", "The assistant's attempt lost in garbled output.",
"We seem to have got into a messy unrunning mid." The model produced a garbled
first attempt, detected it, and rewrote. **The discarded draft was genuinely
generated, so billing it is correct, but it is returned in neither `content`
nor `reasoning`, so the customer cannot reconstruct those tokens.**

That is a real *verifiability* gap rather than a dishonesty one, and it is the
same category the original paper worries about, arrived at innocently. Worth
its own paragraph in the writeup: even a perfectly honest provider can bill
tokens that no external party can reproduce from the response.

Caveats held open rather than resolved: n=3 on one provider is not enough to
call this provider-specific (the 20b is the weakest model in the study and runs
at temperature 1.3, the most degeneracy-prone setting). OpenRouter's gpt-oss-20b
shows a separate, opposite pattern (mean -1.86) that is not yet explained. Both
cells were still running when this was written.

## Round 2 complete (2026-08-08), 1,800 requests, 0 errors

| provider | model | n | n_inf | mean Δ | % exact |
|---|---|---:|---:|---:|---:|
| DeepInfra | gemma-4-31B | 200 | 199 | -0.030 | 99.5% |
| DeepInfra | GLM-5.2 | 200 | 144 | -0.056 | 97.2% |
| DeepInfra | gpt-oss-120b | 200 | 182 | +1.698 | 95.1% |
| DeepInfra | gpt-oss-20b | 200 | 163 | +9.859 | 72.4% |
| OpenRouter | gpt-oss-120b | 200 | 169 | -0.858 | 82.2% |
| OpenRouter | gpt-oss-20b | 200 | 156 | -1.583 | 63.5% |
| Together | gpt-oss-120b | 200 | 158 | +0.057 | 93.7% |
| Together | gpt-oss-20b | 200 | 170 | +0.082 | 91.2% |
| Z.ai | GLM-5.2 | 200 | 159 | +0.006 | 99.4% |

The three model-choice fixes landed cleanly:

- **gemma-4-31B on DeepInfra now matches Together exactly** (was gemma-3-4b, an
  older and much smaller model): 199/200 informative, 99.5% exact.
- **GLM-5.2, the current model, is as clean as 4.6 was**: Z.ai 99.4% exact,
  DeepInfra 97.2%.
- **gpt-oss went from one orphaned provider to four**, which is what exposed
  the two findings below.

### DeepInfra's gpt-oss means are driven by four "rewrite" records

gpt-oss-20b's +9.86 mean comes almost entirely from three records at +441,
+443, +655. The 120b's +1.70 from a single +301. Excluding those, the cells sit
at +0.425 and +0.044.

Those records are the model discarding a garbled first attempt and rewriting
("Oops the assistant messed up, it output wrong... Let's rewrite."). The
discarded draft was genuinely generated, so billing it is correct, but it is
returned in neither `content` nor `reasoning`. **A perfectly honest provider
can bill tokens no external party can reconstruct from the response.** That is
a verifiability gap, not a dishonesty one, and it deserves its own paragraph in
the writeup, it is the strongest real-world instance of the general problem
the original paper gestures at.

Not claimed: that this is DeepInfra-specific. Four events across two cells, on
the weakest models in the study at temperature 1.3, is not enough.

### OpenRouter's negative means are multi-backend convention divergence, RESOLVED

Both OpenRouter gpt-oss cells ran negative (-1.58, -0.86). The delta
distributions are not spreads but contain a **discrete cluster at exactly -10**
(13 records on 20b, 14 on 120b, plus every SiliconFlow-served request).

-10 is exactly the cost of the Harmony structural tokens our reconstruction
adds: `assistant<|channel|>analysis<|message|>` = 4 tokens, and
`<|end|><|start|>assistant<|channel|>final<|message|>` = 6 tokens. Confirmed by
re-scoring those records with plain concatenation, which recovers most of them.

**OpenRouter routes a single customer-facing model across many upstream
backends, and those backends do not agree on whether Harmony structural tokens
are billed.** Some bill them, some do not. Worse for auditability: only about
5% of responses disclose which backend served them
(`provider_name` is absent on the rest).

This is the convention-divergence finding again, but sharper: previously
different *resellers* billed the same checkpoint differently. Here a *single
reseller* bills the same model differently request-to-request depending on
invisible routing. A customer cannot pick one correct reconstruction even in
principle. All of it in the undercharging direction.

### Together's two cells flagged marginally

together/gpt-oss-120b (+0.057 vs MDE 0.056) and gpt-oss-20b (+0.082 vs MDE
0.060) technically exceed threshold. Both are ~0.01% of the bill, and their
MDEs are that tight only because the great majority of samples are exactly
zero. Recorded rather than dismissed, but far below the paper's smallest
claimed rate and not length-scaling.

## FINAL TOTALS

**10,800 real billed requests · 33 distinct cells · 8 platforms**, four resale
platforms (Together, Fireworks, DeepInfra, OpenRouter) and four first-party
APIs (Z.ai, OpenAI non-reasoning, OpenAI reasoning/GPT-5.6, Google).

## Integrity audit + the max_tokens finding (2026-08-08)

`audit_integrity.py` re-derives every quantity from raw records with its own
code rather than importing the analysis path it checks. It surfaced four
things. Final status is **0 problems**.

1. **Our own filter was wrong.** "Informative" was `billed < max_tokens`, which
   silently dropped records billed ABOVE the cap. Those are not truncated and
   must be counted. Corrected to `billed != max_tokens`. No conclusion changed,
   the OpenRouter cells gained samples.

2. **That bug was concealing a real finding, see below.**

3. `zai/glm` was defined in two maps with different `reasoning_format` strings
   (`glm_thinking` vs `think_tags`). Verified identical in effect (0/200
   records differ when recomputed both ways) and unified to `think_tags`.

4. A false positive in the audit script itself: the Google-native file uses a
   two-channel schema, so the generic null-check did not apply. Fixed.

Also verified: all 30 full cells share an identical 200-prompt set, no
duplicate prompts anywhere, every cell exactly n=200, no record fails to
resolve to a config entry, and every delta recomputes identically through an
independent code path.

### `max_tokens` does not bound the reasoning channel on OpenRouter

24 records billed above the requested cap, all OpenRouter, none anywhere else.
Decomposition shows the visible output stays within the cap in 23 of 24. The
overage is reasoning. Worst case: cap 1200, billed 7,093 = 1,202 visible +
5,889 reasoning.

Cross-provider control, same checkpoint and same requested cap (Kimi-K2.6,
1200): Together 82 pinned at cap / 0 over. Fireworks 126 / 0. DeepInfra 68 / 0,
OpenRouter 38 / **18 over**. Three platforms count reasoning against the cap,
OpenRouter does not. OpenAI documents that `max_completion_tokens` includes
reasoning, and the GPT-5.6 data confirms it.

Framing that matters: this is NOT "the cap is ignored" and NOT overcharging , 
the tokens were generated and billing them is correct. It is that the
customer's only spend control does not cover the channel that dominates the
bill on reasoning models, and the semantics differ per platform with nothing in
the response to signal which applies.

## Direct observation of the generated token sequence (2026-08-09)

An external review identified the deepest problem with the study as originally
built, and it was correct. Writing G for the token sequence the model actually
generated, s = decode(G), B for the billed count and C(s) for the canonical
re-tokenization:

    D = B - C(s) = (B - |G|) + (|G| - C(s))

Only the first term is overcharging. The second is unobserved, and it is
nonzero exactly when the generated sequence is not the canonical one, which is
the ambiguity arXiv:2505.21627 exploits. So D = 0 does not establish honesty,
and no amount of extra sampling repairs that. The reviewer called this fatal to
the broad reading of most cells, and they were right.

### The fix: observe |G| instead of inferring it

Several providers return per-token `logprobs`, one entry per generated token,
each carrying the token string and id. That IS G, observed. It needs no
tokenizer, so it does not depend on trusting a published one either. The
measured quantity becomes

    F = B - |G|

which is overcharging itself, with no unobserved term.

Availability, checked live: **Fireworks** returns logprobs on every cell tested
including the reasoning channel, **DeepInfra** on most, **OpenAI** already in
use. Together, OpenRouter, Z.ai and Google return none, so those cells remain
limited to the weaker canonical comparison and must be reported that way.

### The completeness gate, and the false positive it caught

F is interpretable only when the logprob array really contains every generated
token. The check: each parsed field must appear in the concatenated stream, and
the stream must be no shorter than their sum. Structural channel markers sit
between reasoning and content, so a naive `parsed in stream` test fails
legitimately and must not be used.

This mattered immediately. DeepInfra's Kimi produced F = +5, +4, +7, which is
the overcharging direction and would have been the study's only positive
finding. The gate rejected it: the stream runs SHORTER than the parsed text on
those records, which is impossible for a complete array, so DeepInfra omits
logprob entries for that model and F there measures missing data. Reported as
uninterpretable rather than as a finding.

### Result

n=200 per cell, 1,800 requests, 0 errors, temperature 1.3 and top_p 0.95.

| cell | n | verifiably complete | F = 0 | mean F |
|---|---:|---:|---:|---:|
| fireworks/deepseek | 200 | 200 | 200/200 | 0.0000 |
| fireworks/glm | 200 | 200 | 200/200 | 0.0000 |
| fireworks/gpt-oss-120b | 200 | 200 | 200/200 | 0.0000 |
| fireworks/gpt-oss-20b | 200 | 197 | 197/197 | 0.0000 |
| fireworks/kimi | 200 | 200 | 200/200 | 0.0000 |
| deepinfra/gemma | 200 | 200 | 200/200 | 0.0000 |
| deepinfra/llama | 200 | 194 | 194/194 | 0.0000 |
| deepinfra/deepseek | 200 | 3 | 3/3 | 0.0000 |
| deepinfra/kimi | 200 | 0 | logprobs incomplete | n/a |

**1,394 requests with a verifiably complete token stream. 1,394 at exactly
F = 0. Maximum |F| across the whole set: 0.** Summed over those samples,
870,631 billed tokens against 870,631 observed generated tokens, a difference
of zero.

980 of the verified samples carried a reasoning channel, totalling 1,669,213
characters of hidden chain-of-thought that the customer never sees, and every
one of those tokens is accounted for.

### Independent corroboration of the earlier calibration

`structural_chars`, the characters present in the generated stream but absent
from the parsed fields, clusters on a small set of per-model constants: 82 for
gpt-oss, 18 for Kimi on Fireworks, 16 for GLM, 13 for DeepInfra's Gemma, 10 for
its Llama, 8 for DeepSeek. Those are the channel markers that the earlier
calibration work inferred indirectly by fitting reconstructions. Seeing them
appear as fixed constants in the raw token stream is independent confirmation
of that work, arrived at by a different route.

### What this does and does not cover

It covers 7 cells directly. Together, OpenRouter, Z.ai and Google expose no
logprobs, and DeepInfra's Kimi and DeepSeek return incomplete arrays, so those
remain on canonical comparison only. The correct reporting structure is
therefore two-tier: **directly verified** where the generated sequence is
observable, **canonically consistent** elsewhere, never conflating the two.

## Positive control (2026-08-09)

An external review called the missing positive control a blocker, correctly: a
null result means nothing unless the attack was available and the method could
have seen it. Both halves are now answered, offline, from data already held.

### Would injected inflation have been detected?

We inject proportional inflation into the recorded bills at a range of effect
sizes, re-run the detector unchanged, and bootstrap (400 replicates) to get a
detection probability per cell. Effect sizes are anchored on the original
paper, whose smallest claimed rate is 0.28% and largest 11.2%.

Detection probability, using the directly observed generated-token counts:

| cell | n | 0.05% | 0.10% | 0.28% | 1.0% | 5.0% | 11.2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepinfra/gemma | 200 | 100 | 100 | 100 | 100 | 100 | 100 |
| deepinfra/llama | 194 | 100 | 100 | 100 | 100 | 100 | 100 |
| fireworks/deepseek | 200 | 100 | 100 | 100 | 100 | 100 | 100 |
| fireworks/glm | 200 | 100 | 100 | 100 | 100 | 100 | 100 |
| fireworks/gpt-oss-120b | 200 | 100 | 100 | 100 | 100 | 100 | 100 |
| fireworks/gpt-oss-20b | 197 | 100 | 100 | 100 | 100 | 100 | 100 |
| fireworks/kimi | 200 | 100 | 100 | 100 | 100 | 100 | 100 |

Detection is certain at every tested size down to 0.05%, which is a fifth of
the paper's smallest claimed rate. The reason is structural rather than lucky:
F has exactly zero variance on these cells, so any injected inflation moves the
mean off a point mass and is caught immediately.

### Was inflation available in the first place?

For each cell we construct a genuine alternative tokenization of the same
returned string by encoding it character by character. That is a valid
segmentation, so its length is a strict lower bound on the longest one
available. Median surface runs **+268% to +401%** across all 30 cells with
stored text, and the smallest anywhere is +153% (together/deepseek).

So a provider wanting to inflate these exact bills had room to multiply them
several times over before running out of valid tokenizations. The paper's own
0.28% to 11.2% is the fraction of that surface surviving its top-p plausibility
filter, which needs model logits we do not have. Measuring the unfiltered
surface is the conservative direction: it establishes that the attack was
structurally available on real commercial outputs rather than foreclosed by
them.

### What the two controls establish together

Inflation was available at a scale far exceeding anything the paper claims.
The method would have caught it at a fifth of the paper's smallest rate, with
certainty. It measured exactly zero, on 1,394 of 1,394 verifiable requests.

That is the difference between "we looked and saw nothing" and "we established
that there was something to see, that we would have seen it, and that it is
not there."
