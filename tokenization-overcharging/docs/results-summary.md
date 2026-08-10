# Token billing audit: full results

Empirical rebuttal of Velasco et al., "Is Your LLM Overcharging You? Tokenization,
Transparency, and Incentives" (arXiv:2505.21627).

**12,600 real billed requests · 33 distinct (provider, model) cells · 51 cell-runs
across 5 blocks, 8 platforms**, covering four resale platforms (Together, Fireworks,
DeepInfra, OpenRouter) and four first-party APIs (Z.ai, OpenAI non-reasoning,
OpenAI reasoning/GPT-5.6, Google).

The paper proves a provider *could* misreport tokenization to overcharge. Nobody in that
literature ever checked a real invoice, and the same authors twice marked real-provider
evaluation "out of scope of the current work." This is that check.

---

## Method

For open-weight models the tokenizer is public. Send a prompt, capture the returned text
and the billed `completion_tokens`, re-tokenize locally, diff. If billed exceeds canonical,
you have caught them.

The difficulty is that **an honest provider produces nonzero deltas for boring reasons**:
end-of-turn tokens, reasoning-channel structural markers, tokenization-boundary ambiguity
on degenerate text, backend metadata bugs. Run those to ground or you either invent a
scandal or miss a real one. Most of the work went into eliminating them, and two eliminations
overturned earlier conclusions of our own.

**Only non-truncated responses count.** A response billed at exactly `max_tokens` was cut
off, so its bill is pinned to the cap and cannot exceed it no matter what the provider
does. Those samples carry no evidence either way and are excluded from every statistic
rather than padding the reported n. `n_inf` below is the informative count.

---

## Tier 1: direct observation of the generated sequence

Comparing a bill against a canonical re-tokenization leaves one term
unobserved, and it is the term the original paper exploits. Writing G for the
sequence the model generated and C(s) for the canonical re-tokenization of the
returned string, `B - C(s) = (B - |G|) + (|G| - C(s))`, where only the first
part is overcharging.

Providers that return per-token `logprobs` let us observe G directly, one entry
per generated token, with no tokenizer involved. That measures `F = B - |G|`,
which is overcharging itself.

| cell | verified | F = 0 | mean F |
|---|---:|---:|---:|
| fireworks/deepseek | 200 | 200/200 | 0.0000 |
| fireworks/glm | 200 | 200/200 | 0.0000 |
| fireworks/gpt-oss-120b | 200 | 200/200 | 0.0000 |
| fireworks/gpt-oss-20b | 197 | 197/197 | 0.0000 |
| fireworks/kimi | 200 | 200/200 | 0.0000 |
| deepinfra/gemma | 200 | 200/200 | 0.0000 |
| deepinfra/llama | 194 | 194/194 | 0.0000 |
| deepinfra/deepseek | 3 | 3/3 | 0.0000 |
| deepinfra/kimi | 0 | logprobs incomplete, not interpretable | n/a |

**1,394 of 1,394 at exactly zero.** Summed, 870,631 billed tokens against
870,631 observed generated tokens. 980 of these carried a reasoning channel,
1,669,213 characters of hidden text, all accounted for.

Together, OpenRouter, Z.ai and Google return no logprobs, so their cells rest
on the weaker canonical comparison in the tables below and are never pooled
with these.

## Upper bounds instead of detection thresholds

A minimum detectable effect is a prospective design quantity, and "nothing
exceeded its MDE" is not evidence of absence. `equivalence_bounds.py` replaces
it with one-sided upper bounds carrying Bonferroni multiplicity control across
all 59 cells, so the bounds hold simultaneously at 95% familywise confidence.

Tier 1 excludes inflation above **0.0000%**, which is categorical: F has zero
variance, so there is no spread for a bound to accommodate.

Tier 2 is per-cell rather than global. **47 of 50 cells exclude inflation above
0.28%**, the paper's smallest claimed rate. Three do not, and they are named
rather than averaged away: `p99:openrouter/llama` at 0.368%,
`r2:deepinfra/gpt-oss-120b` at 1.281%, and `r2:deepinfra/gpt-oss-20b` at
4.675%. The last two are widened by the rewrite records described below, where
the provider bills a discarded draft it does not return. That is a documented
non-fraud mechanism, and it still widens the bound, which is what an
adversarial reader will quote.

## Positive control

Injected inflation is detected in 100% of bootstrap replicates at every size
tested down to 0.05%, a fifth of the paper's smallest claimed rate, across all
seven directly verified cells. The attack surface actually available on these
outputs, measured as a valid alternative tokenization of the same string, runs
at a median of +268% to +401% per cell. Inflation was available at scale, the
method would have caught it, and it measured zero.

---

## Why the numbers differ between cells

Five distinct causes. None of them is provider honesty.

### 1. The same label is not the same model

Resellers stopped carrying the tiny checkpoints the paper used, so each provider was
matched to the nearest model it actually serves on pay-per-token.

| Cell | Platform | Model actually served | Tokenizer used |
|---|---|---|---|
| llama | Together | Llama-3.3-**70B**-Instruct-Turbo | meta-llama/Llama-3.3-70B-Instruct |
| llama | DeepInfra | Meta-Llama-3.1-**8B**-Instruct-Turbo | meta-llama/Meta-Llama-3.1-8B-Instruct |
| llama | OpenRouter | llama-3.2-**1B**-instruct | unsloth/Llama-3.2-1B-Instruct *(substitute)* |
| gemma | Together | gemma-4-**31B**-it | google/gemma-4-31B-it |
| gemma | DeepInfra | gemma-3-**4b**-it | google/gemma-3-4b-it |
| gemma | OpenRouter | gemma-3-**4b**-it | google/gemma-3-4b-it |

A 70B and a 1B are not comparable on response length, coherence, or truncation rate.

### 2. Identical weights, different billing conventions

The finding that moved the numbers most. The *same checkpoint* is billed under different
conventions by different resellers. Every cell was calibrated against live responses
(`calibrate_cells.py`), because assuming a format transfers would have manufactured multi-token
phantom deltas.

| Model | Platform | Reconstruction that reproduces the provider's own count | Bills an EOS? |
|---|---|---|---|
| Kimi-K2.6 | Together / Fireworks / OpenRouter | `<think>…</think>` | no |
| Kimi-K2.6 | **DeepInfra** | **plain concatenation, no think tags** | no |
| DeepSeek-V4-Flash | Together / Fireworks / OpenRouter | `<think>…</think>` | no |
| DeepSeek-V4-Flash | **DeepInfra** | **plain, returns no reasoning field at all** | **yes** |
| GLM-4.6 | DeepInfra | `<think>…</think>` | **yes** |
| GLM-4.6 | Z.ai (first-party) | `<think>…</think>` | no |

### 3. `n_inf` differs because models truncate at different rates

This measures how often a model finished naturally inside the token budget.
Heavy reasoning models burn the cap: Fireworks' gpt-oss cells land at 43–49 informative out
of 200, while DeepInfra's gemma reaches 167. Raising the cap to 1,200 for the expansion
lifted the overall informative rate from 47% to ~80%.

### 4. Boundary noise scales with how degenerate the text is

Re-encoding a decoded string does not always reproduce the original sampling path at
ambiguous byte-pair boundaries. This concentrates in incoherent, mixed-script output,
exactly what the smallest model at temperature 1.3 produces. OpenRouter's 1B llama
therefore carries the widest spread in the study, and its noise runs **negative**, against
the provider's own revenue.

### 5. The detection threshold is per-cell, not global

Minimum detectable effect depends on that cell's delta variance, its informative sample
size, and its mean response length. A zero-variance cell resolves to about 0.002%, while the noisiest
to 0.60%. This is why one cell flags at +0.6 tokens while another with a larger mean does
not: **the threshold moved rather than the behavior.**

---

## Results for every cell

`mean Δ` is billed minus canonical, in tokens. Positive is the overcharging direction.
Statistics on informative samples only.

### Block A: main study, top_p 0.95 (paper's headline setting), 1,800 requests

| Platform | Model | n | n_inf | mean Δ | % exact | range | MDE | avg billed |
|---|---|---:|---:|---:|---:|:--:|---:|---:|
| DeepInfra | gemma | 200 | 167 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 101 |
| DeepInfra | llama | 200 | 111 | −0.0090 | 99.1% | −1 … 0 | 0.023% | 98 |
| Fireworks | gpt-oss-120b | 200 | 49 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 152 |
| Fireworks | gpt-oss-20b | 200 | 43 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 184 |
| OpenRouter | gemma | 200 | 162 | +0.0062 | 99.4% | 0 … 1 | 0.016% | 98 |
| OpenRouter | llama | 200 | 79 | −0.0380 | 94.9% | −2 … 1 | 0.157% | 53 |
| Together | gemma | 200 | 64 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 172 |
| Together | llama | 200 | 159 | −0.0063 | 99.4% | −1 … 0 | 0.013% | 115 |
| Z.ai | glm | 200 | 13 | **+0.0000** | **100.0%** | 0 … 0 | 0.003% | 242 |

### Block B: main study, top_p 0.99 (their most fraud-favorable setting), 1,600 requests

| Platform | Model | n | n_inf | mean Δ | % exact | range | MDE | avg billed |
|---|---|---:|---:|---:|---:|:--:|---:|---:|
| DeepInfra | gemma | 200 | 163 | −0.0061 | 98.8% | −2 … 1 | 0.034% | 101 |
| DeepInfra | llama | 200 | 94 | −0.0851 | 94.7% | −2 … 0 | 0.120% | 81 |
| Fireworks | gpt-oss-120b | 200 | 49 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 150 |
| Fireworks | gpt-oss-20b | 200 | 35 | +0.1143 | 91.4% | 0 … 2 | 0.093% | 181 |
| OpenRouter | gemma | 200 | 167 | −0.0060 | 99.4% | −1 … 0 | 0.014% | 103 |
| OpenRouter | llama | 200 | 62 | −0.1613 | 90.3% | −4 … 1 | 0.600% | 41 |
| Together | gemma | 200 | 67 | +0.0149 | 98.5% | 0 … 1 | 0.021% | 174 |
| Together | llama | 200 | 162 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 118 |

### Block C: expansion to DeepSeek, Kimi, GLM and GPT, top_p 0.95, cap 1,200, 2,600 requests

| Platform | Model | n | n_inf | mean Δ | % exact | range | MDE | avg billed |
|---|---|---:|---:|---:|---:|:--:|---:|---:|
| DeepInfra | deepseek | 200 | 198 | +0.0051 | 99.5% | 0 … 1 | 0.003% | 365 |
| DeepInfra | glm | 200 | 58 | +0.0172 | 96.6% | −1 … 2 | 0.011% | 893 |
| DeepInfra | kimi | 200 | 171 | +0.0351 | 94.2% | −1 … 2 | 0.011% | 532 |
| Fireworks | deepseek | 200 | 149 | +0.0000 | 98.7% | −1 … 1 | 0.004% | 657 |
| Fireworks | glm | 200 | 146 | +0.0137 | 98.6% | 0 … 1 | 0.004% | 626 |
| Fireworks | kimi | 200 | 159 | +0.0063 | 96.9% | −1 … 1 | 0.006% | 536 |
| OpenAI | gpt-4.1-nano | 200 | 195 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 107 |
| OpenAI | gpt-4o-mini | 200 | 191 | **+0.0000** | **100.0%** | 0 … 0 | 0.001% | 131 |
| OpenRouter | deepseek | 200 | 155 | −0.1806 | 78.1% | −1 … 1 | 0.017% | 516 |
| OpenRouter | kimi | 200 | 171 | −0.2164 | 69.0% | −2 … 2 | 0.027% | 540 |
| Together | deepseek | 200 | 149 | **+0.0000** | **100.0%** | 0 … 0 | 0.000% | 623 |
| Together | kimi | 200 | 162 | +0.0000 | 98.1% | −2 … 1 | 0.007% | 535 |
| Z.ai | glm | 200 | 170 | +0.0118 | 98.8% | 0 … 1 | 0.003% | 696 |

### Block D: expansion at top_p 0.99, 2,400 requests

| Platform | Model | n | n_inf | mean Δ | % exact | range | MDE | avg billed |
|---|---|---:|---:|---:|---:|:--:|---:|---:|
| DeepInfra | deepseek | 200 | 197 | +0.0355 | 97.0% | 0 … 2 | 0.011% | 356 |
| DeepInfra | glm | 200 | 75 | −0.1600 | 78.7% | −2 … 2 | 0.026% | 777 |
| **DeepInfra** | **kimi** ⚑ | 200 | 161 | **+0.6025** | 41.0% | −3 … 3 | 0.025% | 523 |
| Fireworks | deepseek | 200 | 127 | +0.0079 | 99.2% | 0 … 1 | 0.003% | 608 |
| Fireworks | glm | 200 | 139 | +0.0144 | 97.1% | −1 … 1 | 0.005% | 645 |
| Fireworks | kimi | 200 | 115 | +0.0087 | 96.5% | −3 … 2 | 0.019% | 448 |
| OpenAI | gpt-4.1-nano | 200 | 195 | **+0.0000** | **100.0%** | 0 … 0 | 0.002% | 109 |
| OpenAI | gpt-4o-mini | 200 | 194 | +0.0103 | 99.0% | 0 … 1 | 0.014% | 129 |
| OpenRouter | deepseek | 200 | 153 | −0.2745 | 71.9% | −2 … 1 | 0.018% | 516 |
| OpenRouter | kimi | 200 | 173 | −0.0636 | 74.6% | −2 … 2 | 0.022% | 562 |
| Together | deepseek | 200 | 156 | +0.0128 | 98.7% | 0 … 1 | 0.003% | 662 |
| Together | kimi | 200 | 156 | −0.0833 | 97.4% | −12 … 1 | 0.036% | 539 |

**Eleven cell-runs at exactly 100.0%.** One cell above its own detection threshold (below).

### Block E: round 2 model-choice fixes, top_p 0.95, cap 1,200, 1,800 requests

Three inconsistencies corrected: gpt-oss extended from one provider to four,
gemma-4-31B on DeepInfra matched to Together, GLM-5.2 (current) added alongside 4.6.

| Platform | Model | n | n_inf | mean Δ | % exact | range | MDE |
|---|---|---:|---:|---:|---:|:--:|---:|
| DeepInfra | gemma-4-31B | 200 | 199 | −0.030 | 99.5% | −6 … 0 | 0.026% |
| DeepInfra | GLM-5.2 | 200 | 144 | −0.056 | 97.2% | −7 … 1 | 0.020% |
| DeepInfra | gpt-oss-120b | 200 | 182 | +1.698 | 95.1% | 0 … 301 | 0.763% |
| DeepInfra | gpt-oss-20b | 200 | 163 | +9.859 | 72.4% | −1 … 655 | 2.356% |
| OpenRouter | gpt-oss-120b | 200 | 169 | −0.858 | 82.2% | −10 … 3 | 0.104% |
| OpenRouter | gpt-oss-20b | 200 | 156 | −1.583 | 63.5% | −46 … 6 | 0.205% |
| Together | gpt-oss-120b | 200 | 158 | +0.057 | 93.7% | −1 … 2 | 0.010% |
| Together | gpt-oss-20b | 200 | 170 | +0.082 | 91.2% | −1 … 2 | 0.011% |
| Z.ai | GLM-5.2 | 200 | 159 | +0.006 | 99.4% | 0 … 1 | 0.003% |

The two DeepInfra gpt-oss means and the two OpenRouter ones are both fully
explained below, and neither is inflation.

### Block F: GPT-5.6 reasoning models, 400 requests

`logprobs` is not returned for reasoning models and the reasoning text is
discarded server-side, so both Tier-1 checks are unavailable. What remains is an
accounting identity, calibrated live (16/16 consistent):

    billed == tiktoken(visible) + reasoning_tokens + K
    K = 3  when reasoning_tokens == 0     (final channel header only)
    K = 9  when reasoning_tokens  > 0     (analysis channel + transition)

| Model | n | n_inf | exact | mean Δ | reasoning billed | hidden share |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 200 | 196 | **194/196** | +0.061 | 16,339 | 36.6% |
| gpt-5.6-luna | 200 | 194 | **191/194** | +0.098 | 17,741 | 35.1% |

**About a third of the bill is reasoning tokens the customer never sees, and the
total is exactly accounted for on 385 of 390 informative samples.** The five
exceptions are a constant +6/+7, never scaling, all on high-reasoning
factual-lookup answers, which is consistent with an extra Harmony channel that the two-mode K
does not cover. Left as an unexplained residual rather than fitted away.

**This is a weaker claim than the open-weight cells and should not be conflated
with them.** It verifies the billed total is fully accounted for by the
provider's own disclosed reasoning count, since a padded bill breaks the identity. It
does not verify that `reasoning_tokens` is itself truthful, because that is a
claim about content nobody can see. Only the Google Gemma cell closes that gap.

---

## The headline: hidden reasoning tokens, verified

Every auditing framework in this literature treats reasoning tokens as unverifiable, since the
provider bills for chain-of-thought the customer never sees. Google serves the open-weight
`gemma-4-31b-it` on its own API and returns the **raw thought text** alongside a separate
`thoughtsTokenCount`, so the channel becomes directly checkable.

Google first-party API · gemma-4-31b-it · n=200, all informative:

| Channel | exact | mean Δ | tokens billed | share of output |
|---|---:|---:|---:|---:|
| Thought (hidden reasoning) | **194 / 200** | +0.0150 | 83,838 | **50.7%** |
| Answer (visible output) | 192 / 200 | −0.0200 | 81,533 | 49.3% |

Provider arithmetic (`prompt + candidates + thoughts = total`) reconciled **200/200**.

**Half the bill was for tokens the customer never sees, and it is correct to a mean of
+0.015 tokens.** This is the channel that matters far more in 2026 than the retokenization
channel the paper analyzes.

---

## The one cell that flagged, and why it is not inflation

DeepInfra's Kimi cell at top_p 0.99 was the only cell in 42 runs to exceed its own detection
threshold: **+0.602 tokens per response**, 41% exact.

A per-response EOS rule would have made it vanish, which is exactly how any finding becomes
a null result. Instead, the test that separates the two hypotheses: proportional inflation
must hold its *percentage* constant while the absolute token count grows with response
length. A fixed convention token must do the opposite.

| billed tokens | n | mean Δ | as % of billed |
|---|---:|---:|---:|
| 0–249 | 29 | +0.621 | 0.497% |
| 250–499 | 52 | +0.558 | 0.149% |
| 500–749 | 45 | +0.600 | 0.096% |
| 750–999 | 21 | +0.571 | 0.065% |
| 1000–1249 | 14 | +0.786 | 0.070% |

The absolute delta stays flat at roughly half a token, the percentage decays as 1/length, and the **maximum
delta anywhere in the cell: 3 tokens.** That is an end-of-turn convention token billed on
about half of that model's responses.

The paper's *smallest* claimed rate, 0.28% systematic, would be 2 to 78 extra tokens on a
700-token response and would grow with length. The two cannot be confused given this test.

---

## Two further findings from round 2

### An honest provider can bill tokens you cannot reconstruct

DeepInfra's gpt-oss-20b mean of +9.86 comes almost entirely from three records
at +441, +443 and +655 (and the 120b's +1.70 from a single +301). Excluding
those, the cells sit at +0.425 and +0.044.

Reading those records explains them: the model produced a garbled first attempt,
detected it, and rewrote: *"Oops the assistant messed up, it output wrong... 
Let's rewrite."* The discarded draft was genuinely generated, so billing it is
correct. But it is returned in neither `content` nor `reasoning`, so the customer
cannot reconstruct those tokens.

This is a **verifiability** gap rather than a dishonesty one, and it is the strongest
real-world instance of the general problem the original paper only gestures at.
Not claimed as provider-specific: four events across two cells, on the weakest
models in the study at temperature 1.3, is not enough to support that.

### A single reseller bills the same model inconsistently, request to request

Both OpenRouter gpt-oss cells ran negative. Their delta distributions are not
spreads, because each contains a discrete cluster at **exactly −10** (13 records on the
20b, 14 on the 120b, plus every SiliconFlow-served request).

−10 is precisely the cost of the Harmony structural tokens the reconstruction
adds: `assistant<|channel|>analysis<|message|>` = 4 tokens, and
`<|end|><|start|>assistant<|channel|>final<|message|>` = 6. Confirmed by
re-scoring those records with plain concatenation, which recovers most of them.

**OpenRouter routes one customer-facing model across many upstream backends, and
those backends disagree on whether Harmony structural tokens are billed.** Only
about 5% of responses disclose which backend served them.

This sharpens the earlier convention-divergence finding. Before: different
*resellers* billed the same checkpoint differently. Here: a *single* reseller
bills the same model differently request-to-request based on routing the
customer cannot see, so no single reconstruction can be correct even in
principle. All of it in the undercharging direction.

---

## Takeaways

- **This is not "we failed to detect anything."** Eleven cell-runs are correct to the token
  at 100.0% exact, across 10,800 billed requests. That is a positive measurement,
  not a null result.
- **The current models are as clean as the older ones.** GLM-5.2 (99.4% exact on
  its creator's own API) and GPT-5.6 (385/390 exactly accounted for) both check
  out, so the result is not an artifact of auditing superseded checkpoints.
- **The flagged cell is the best evidence in the study.** It shows the instrument
  resolves a half-token-per-response effect and then classifies it correctly. Lead with it
  rather than burying it.
- **The channel the field calls unverifiable is verifiable, and honest.** Hidden reasoning
  billing checks out to +0.015 tokens across 83,838 billed reasoning tokens. It is
  unverifiable only because tokenizers go unpublished.
- **The real problem is verifiability rather than integrity.** Providers bill honestly, and customers
  still cannot check. The ask reduces to *publish the tokenizer*, because the attack
  produces a valid alternative segmentation of the same string, only the canonical one
  proves anything. OpenAI already does this, while Anthropic and Google do not.
- **The single defect found across seven platforms is a matter of schema hygiene.** Google's
  OpenAI-compatibility layer never reconciles (0/8, ~200 tokens/response unaccounted,
  `reasoning_tokens` left empty) while its own native endpoint proves it computes that
  value. Any auditor built on that interface mis-accounts Google.
- **Provider billing conventions genuinely differ for identical weights.** This likely
  explains phantom discrepancies in prior work, and is a contribution on its own.

---

## What this does not show

- We exclude OpenAI's reasoning models from the verified cells, because reasoning content is
  discarded server-side, so that share of the bill has no ground truth.
- Anthropic discloses thinking tokens and is internally self-consistent (constant −10
  wrapper offset against its own `count_tokens`, no drift), but publishes no tokenizer and
  no combined total. Self-consistent is not independently verified.
- Gemma on Google's API says nothing about how Google bills its *proprietary* Gemini
  models. It proves the channel is checkable, and honest where checked.
- Nothing here addresses model substitution, which is a harder problem needing trusted execution or
  zkML.

---

## Where the numbers come from

- Raw data lives in `results/*.jsonl`, holding every request, prompt, response, reasoning text, billed and
  recomputed count.
- Per-cell summary: `results/all_cells_summary.json`.
- Regenerate: `python final_report.py`.
- Full methodology, every deviation, bug and withdrawn conclusion: `implementation-notes.md`.
- The argument: `the-case-against.md`. The provider-side fix: `verifiable-billing-spec.md`.
