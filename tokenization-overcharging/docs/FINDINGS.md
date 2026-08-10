# Is Your LLM Overcharging You? No, and It Was Never Going To

**An empirical rebuttal of Velasco, Tsirtsis, Okati & Gómez-Rodríguez,
"Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives"
(arXiv:2505.21627).**

12,600 real billed API requests, 33 (provider, model) cells, 8 platforms.
Billed tokens match the directly observed generated sequence exactly on every
request where that sequence can be observed.

---

## 1. The claim, and what is actually at issue

The paper proves that a provider *could* overcharge by reporting a different
but perfectly valid tokenization of the same output string, and that finding
the optimal padding of this kind is NP-hard. **The mathematics is correct and
we do not dispute it.**

We dispute the claim the title sells, which is a live warning that pay-per-token
pricing exposes users to covert overcharging today. That is an empirical claim
about the world, and nobody tested it. Every number in the paper comes from the
authors' own hardware attacking their own downloaded weights, and no invoice
was ever examined, either by them or by any of the follow-up work in this line.
Their own second paper marks real-provider evaluation "out of scope of the
current work."

The experiment was cheap, obvious, and available, because the models they
attack are resold commercially, for real money, by platforms with public
tokenizers. This is that experiment.

---

## 2. Method

For open-weight models the tokenizer is public, so ground truth exists. We send
a prompt, capture the returned text and the billed `completion_tokens`,
re-tokenize the text locally, and diff the two. If the billed count exceeds the
canonical one, that is overcharging.

The entire difficulty is that **an honest provider produces nonzero deltas for
mundane reasons**, including end-of-turn tokens, reasoning-channel structural
markers, tokenization-boundary ambiguity on degenerate text, and backend
metadata bugs. Failing to run those causes to ground yields either a fabricated
scandal or a missed signal. We therefore calibrated every cell below against
live responses before running it, choosing the reconstruction that reproduces
the provider's own count on honest traffic, established empirically per cell
rather than assumed.

Two rules govern the statistics.

**Only non-pinned responses count.** A response billed at exactly `max_tokens`
was truncated, so its bill is pinned to the cap and cannot exceed it whatever
the provider does. There is no evidence in such samples in either direction, so
we exclude them, and `n_inf` reports the informative count. A response billed
*above* the cap was not truncated and is therefore included, as discussed in
section 5.3.

**Detection thresholds are per-cell.** We compute the minimum detectable effect
from each cell's own delta variance, informative sample size, and mean response
length (one-sided, α=0.05, power=0.80). A mean only counts as signal when it
exceeds the MDE of the cell it came from.

---

## 3. Headline result: billed equals generated, exactly

The strongest evidence does not rely on re-tokenization at all.

Comparing a bill against a canonical re-tokenization has a known weakness, and
it is the weakness the original paper exploits. Writing G for the token
sequence the model actually generated, s for the returned string, B for the
billed count and C(s) for the canonical re-tokenization:

    B - C(s) = (B - |G|) + (|G| - C(s))

Only the first term is overcharging. The second is unobserved, and it is
nonzero precisely when the generated sequence is not the canonical one. So
agreement with C(s) does not by itself establish honesty, no matter how many
requests we send.

We therefore stopped inferring G and observed it. Several providers return
per-token `logprobs`, one entry per generated token, each carrying the token
string and id. That is G, measured rather than reconstructed, and it involves
no tokenizer, so it does not depend on trusting a published one either. The
quantity becomes

    F = B - |G|

which is overcharging itself, with nothing unobserved.

**Result: 1,394 of 1,394 requests with a verifiably complete token stream came
back at exactly F = 0.** Summed across them, 870,631 billed tokens against
870,631 observed generated tokens, a difference of zero. The largest deviation
anywhere in the set is zero.

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

980 of these carried a reasoning channel, 1,669,213 characters of
chain-of-thought the customer never sees, and every token of it is accounted
for.

This answers the paper on its own terms. Its attack substitutes a different
valid tokenization of the same string. When the bill equals the observed
generated sequence exactly, there is no room for a substitution to hide in.

**Two guards on this result.** F is only interpretable when the logprob array
really contains every generated token, so we require each parsed field to
appear in the stream and the stream to be no shorter than their sum. That gate
matters: DeepInfra's Kimi produced F values of +5 and +7, the overcharging
direction, and would have been the study's only positive finding. Its stream
runs *shorter* than the parsed text, which is impossible for a complete array,
so the provider omits entries and that F measures missing data. It is reported
as uninterpretable rather than as a result. Separately, Together, OpenRouter,
Z.ai and Google return no logprobs, so those cells rest on the weaker canonical
comparison and are reported as such throughout.

### The same conclusion from the weaker method

Across 10,800 billed requests at both the paper's headline sampling setting and
its most fraud-favorable one, no cell shows a mean overcharge that survives its
own detection threshold once we identify the cause.

This result is stronger than a failure to detect. **Eleven cell-runs are
correct to the token, at 100.0% exact, with zero deviation across every
informative sample.**

| Strongest cells | informative n | result |
|---|---|---|
| OpenAI gpt-4o-mini / gpt-4.1-nano | 386 | exact, by tokenizer and by token stream |
| Together gemma-4-31B, deepseek, llama | 375 | 100.0% exact |
| DeepInfra gemma | 167 | 100.0% exact |
| Fireworks gpt-oss-20b / 120b | 92 | 100.0% exact |

Two checks verify the OpenAI cells and agree perfectly: recomputation against
OpenAI's published `tiktoken` encoding, and the `logprobs` entry count, which
carries one entry per generated token and so needs no tokenizer. We avoid
calling these fully independent, since the logprob array is a second
provider-supplied artifact while only `tiktoken` is an external reference. They
are two checks with one external trust root, which is still what makes the
generated sequence observable.

The current models check out as well as the superseded ones. GLM-5.2 is 99.4%
exact on its creator's own API, GPT-5.6 is exactly accounted for on 385 of 390
informative samples, and gemma-4-31B on DeepInfra is 99.5% exact.

---

## 4. The channel that matters most is verifiable, and honest

Every auditing framework in this literature treats hidden reasoning tokens as
unverifiable, since the provider bills for chain-of-thought the customer never
sees and there is nothing to recompute against. In 2026 this is the channel
that carries real money, far more than the retokenization channel the paper
analyzes.

It turns out to be verifiable. Google serves the open-weight `gemma-4-31b-it`
on its own first-party API, and the native endpoint returns **the raw thought
text** alongside a separate `thoughtsTokenCount`, allowing the hidden channel to
be checked directly against a public tokenizer.

| Channel | exact | mean Δ | tokens billed | share |
|---|---:|---:|---:|---:|
| Thought (hidden reasoning) | **194 / 200** | +0.015 | 83,838 | **50.7%** |
| Answer (visible output) | 192 / 200 | −0.020 | 81,533 | 49.3% |

The provider's own arithmetic (`prompt + candidates + thoughts = total`)
reconciles **200/200**.

**Half the bill was for tokens the customer never sees, and it is correct to a
mean of 0.015 tokens per response.** Hidden reasoning billing is unverifiable
only because tokenizers go unpublished, and there is nothing intrinsic about
the channel that prevents checking it.

GPT-5.6 gives a weaker but consistent second data point. The provider discards
reasoning content server-side and returns no `logprobs`, so only an accounting
identity remains, namely `billed = tiktoken(visible) + reasoning_tokens + K`,
where K is fixed Harmony channel structure. The identity holds on 385 of 390
informative samples across `sol` and `luna`, with roughly 36% of the bill
hidden. This verifies that the total is fully accounted for, and it does not
verify that `reasoning_tokens` is itself truthful. That distinction should not
be blurred: OpenAI here is self-consistent, while the Google open-weight cell is
independently verified.

---

## 5. What we did find

We found no fraud. Four things are still worth reporting, because they are real
and they cost customers either money or auditability. Where we established a
mechanism we say so, and where only the observation is established, we say that
instead.

### 5.1 An honest provider can bill tokens you cannot reconstruct

Four records across DeepInfra's gpt-oss cells were billed 301 to 655 tokens
more than their responses contain. Reading those records explains them, because
the model produced a garbled first attempt, detected it, and rewrote:
*"Oops the assistant messed up, it output wrong... Let's rewrite."*

The model genuinely generated the discarded draft, so billing it is correct.
However, that draft is returned in neither `content` nor `reasoning`, so no
external party can reproduce those tokens from the response. This is the
strongest real-world instance of the general problem the paper only gestures
at, and it arises from ordinary honest behavior.

### 5.2 One reseller bills the same model inconsistently, request to request

Both OpenRouter gpt-oss cells run negative, and their delta distributions
contain a discrete cluster at **exactly −10**, which is precisely the cost of
the Harmony structural tokens: `assistant<|channel|>analysis<|message|>` is 4
tokens and `<|end|><|start|>assistant<|channel|>final<|message|>` is 6.

OpenRouter routes one customer-facing model across many upstream backends, and
those backends disagree about whether the Harmony structural tokens are billed.
Only about 5% of responses disclose which backend served the request.

A customer therefore cannot select a single correct reconstruction even in
principle. All of the resulting error runs in the undercharging direction.

### 5.3 Requested caps were exceeded on one platform (mechanism unresolved)

24 requests were billed **above** the `max_tokens` the client asked for. Every
one of them went to OpenRouter, and no other platform produced any.

| Cell | over cap | median overage | worst |
|---|---:|---:|---:|
| openrouter / kimi | 18 / 400 (4.5%) | 990 tokens | **5,893** |
| openrouter / gpt-oss-20b | 3 / 200 | 222 | 390 |
| openrouter / gpt-oss-120b | 2 / 200 | 88 | 153 |
| openrouter / deepseek | 1 / 400 | 1 | 1 |

In total, 29,328 tokens were billed beyond the requested cap.

**The observation is solid and the mechanism is not yet established.** We state
both precisely, because the distinction matters.

*What is established.* Decomposing the over-cap responses shows that the
visible output stays within the cap in 23 of 24 cases, and that the overage is
almost entirely reasoning. The worst case was billed 7,093 tokens against a
1,200-token cap, of which 1,202 were visible and **5,889 were reasoning**.

*What is not.* A controlled follow-up failed to reproduce the behavior. We
re-ran the same model at a 400-token cap under five parameter variants,
covering no `reasoning` field (as in the study), `reasoning.max_tokens`,
`reasoning.effort`, `reasoning.enabled: false`, and `reasoning.exclude: true`,
and all five produced a billed count exactly equal to the cap. The simple
explanation that reasoning is unbounded by `max_tokens` on this platform is
therefore insufficient, since the behavior is intermittent at 4.5% of that cell
rather than systematic.

Backend attribution cannot settle it either. OpenRouter reports `provider_name`
on only about 3% of responses, and 23 of the 24 over-cap records carry no
attribution at all.

There are two further points of context. OpenRouter does expose a dedicated
reasoning budget through `reasoning.max_tokens` and `reasoning.effort`, which
the study did not set, although leaving it unset did not reproduce the overage.
In addition, OpenRouter's own documentation states that `max_tokens` "must be
strictly higher than the reasoning budget to ensure there are tokens available
for the final response," which implies that reasoning is intended to count
against `max_tokens`. The observed records contradict that intent.

*What a targeted reproduction settled.* We ran 120 fresh requests at the exact
study settings, and they reproduced the behavior at **4.2%** (5 of 120) against
4.5% in the study, showing that it is stable and reproducible rather than an
artifact of one batch. That run also captured backend attribution from the
response body, which is the field the original runs missed, and it revealed the
structural cause without isolating a single culprit.

**OpenRouter served this one customer-facing model from 19 distinct upstream
backends across 120 requests.** The five over-cap events came from four
different backends (SiliconFlow twice, Novita, StreamLake, AtlasCloud), so no
single misbehaving upstream explains them, and at n=5 the attribution carries no
statistical weight anyway.

Therefore the overage is real, stable, and reproducible, it is concentrated on
the one platform that fans out across many upstreams, and it is not attributable
to a specific backend at this sample size. What is structurally clear is that a
customer sending one `max_tokens` value has it interpreted by any of 19
different systems, with no way to know which, while the platform's own
documentation says that reasoning should count against that value.

**What each platform documents.** We surveyed the documentation because a
documented separate reasoning budget would reframe this as a case of using the
wrong parameter.

| Platform | Documented relationship of `max_tokens` to reasoning | Separate budget param | Observed |
|---|---|---|---|
| Together | **Explicit**: max_tokens "cap[s] total output... may truncate reasoning", reasoning "billed as completion tokens" | effort guidance | 0/400 over, matches docs |
| OpenAI | **Explicit**: `max_completion_tokens` includes reasoning | `reasoning_effort` | 0 over, matches docs |
| Fireworks | Silent | `thinking.budget_tokens` (>=1024) | 0/400 over |
| DeepInfra | Silent (documents only a 16,384 hard cap) | not documented | 0/400 over |
| OpenRouter | States `max_tokens` "must be strictly higher than the reasoning budget", implying reasoning counts against it | `reasoning.max_tokens` / `.effort` / `.enabled` / `.exclude` | **18/400 over, contradicts its own docs** |

The platform carrying the over-cap records is thus the one whose observed
behavior contradicts its own documentation. It does offer a dedicated reasoning
budget that we did not set, but leaving that budget unset did not reproduce the
overage under controlled testing, so an unset parameter fails to explain it.

The cross-platform contrast is clean regardless. The same checkpoint at the same
requested cap behaves differently elsewhere.

| Platform (Kimi-K2.6, cap 1200) | n | under cap | pinned at cap | over cap |
|---|---:|---:|---:|---:|
| Together | 400 | 318 | 82 | **0** |
| Fireworks | 400 | 274 | 126 | **0** |
| DeepInfra | 400 | 332 | 68 | **0** |
| OpenRouter | 400 | 344 | 38 | **18** |

Three platforms never exceed the cap across 1,200 requests, while OpenRouter
does so on 18 of 400. OpenAI's API is explicit that `max_completion_tokens`
includes reasoning, and our GPT-5.6 data confirms it, since those responses stop
at the cap.

The tokens are real and billing them is correct, so this is not overcharging.
The problem is that a customer's only spend control did not hold on 4.5% of
requests to one platform, and the response carries no signal that distinguishes
those requests from the rest.

**This is the most financially consequential thing in the entire study, and it
has nothing to do with tokenization.** A 4.5% chance that a request costs
several times its authorized ceiling dwarfs the sub-0.1% effects that the
paper's threat model concerns itself with.

### 5.4 A transparency gap in Google's compatibility layer

Google's native API is fully transparent. Its **OpenAI-compatibility endpoint,
serving the same model**, never reconciles, because `prompt_tokens +
completion_tokens` never equals `total_tokens` across the 8 sampled requests,
leaving a median of roughly 200 tokens per response unaccounted for and
`completion_tokens_details.reasoning_tokens` empty. OpenAI's schema defines that
field, and Google demonstrably computes the value, since the native endpoint
returns it.

This is not fraud, because the tokens do appear in `total_tokens`. However, any
auditing tool built against the OpenAI-compatible interface, which is the
industry norm, will systematically mis-account Google's billing.

---

## 6. The one cell that flagged, and why it is not inflation

`deepinfra/kimi` at top_p 0.99 is the only cell in 51 runs whose mean exceeds
its own detection threshold, at **+0.602 tokens per response**.

A per-response EOS rule would make that mean vanish, and we did not apply one,
because tuning a reconstruction until the residual disappears is how any finding
becomes a null result. Instead we ran the test that distinguishes the two
hypotheses. Proportional inflation must hold its *percentage* constant while the
absolute token count grows with response length, and a fixed convention token
must do the opposite.

| billed tokens | n | mean Δ | as % of bill |
|---|---:|---:|---:|
| 0–249 | 29 | +0.621 | 0.497% |
| 250–499 | 52 | +0.558 | 0.149% |
| 500–749 | 45 | +0.600 | 0.096% |
| 750–999 | 21 | +0.571 | 0.065% |
| 1000–1249 | 14 | +0.786 | 0.070% |

The absolute delta stays flat at roughly half a token, the percentage decays as
1/length, and the maximum delta anywhere in the cell is **3 tokens**. What we
are measuring is an end-of-turn convention token billed on about half of that
model's responses.

The paper's smallest claimed rate of 0.28%, applied systematically, would be 2
to 78 extra tokens on a 700-token response and would grow with length. This test
separates the two cases cleanly.

**This cell is the best evidence in the study.** It demonstrates that the
instrument resolves a half-token-per-response effect and then classifies it
correctly.

---

## 7. Why it was never going to happen

The empirical result is unsurprising rather than lucky, because no rational
provider would deploy this attack.

**The legal alternative strictly dominates.** Raising the posted price by the
same percentage costs one edit to a pricing page, with no NP-hard optimization,
no per-response verification pass, no forensic trail, and no crime. The one
thing covert padding buys over a transparent increase is *concealment*, which
has value only when the market would not bear the price transparently, that is,
precisely when customers have already revealed that they would not pay it.

**The paper's own numbers make the attack computationally irrational.** Its
Theorem 3 proves that optimal padding is NP-hard, and its Algorithm 2 workaround
still requires a verification forward pass per response, which its own Table 1
measures as costing more energy than generating the response. A provider would
spend more compute, on metered infrastructure that leaves its own cost trail, to
capture between 0.28% and 11.2%.

**There is no margin desperation at the layer this attack targets.** API gross
margins are positive and rising, and the losses at frontier labs come from
training and capital expenditure sitting on top of an already-profitable serving
layer. Shaving 2 to 3% off API revenue does not touch that, and it risks a
trust-dependent valuation in the attempt.

**The observed behavior is the legal one.** Providers raise prices in public, on
pricing pages, while inference costs fall.

One retreat position is worth granting explicitly, namely that the paper only
claims the attack is possible rather than that anyone performs it. Granting it
concedes the argument that matters, because the real-world stakes that justified
the title were never supported by anything beyond self-run simulation, and they
now have a counter-experiment against them that the authors twice declined to
run.

---

## 7b. The positive control

A null result is worth nothing unless the attack was available and the method
could have seen it. Both are established, offline, from data already held.

**Would we have caught inflation?** We inject proportional inflation into the
recorded bills, re-run the detector unchanged, and bootstrap 400 replicates per
cell. Detection is certain at every size tested down to **0.05%**, a fifth of
the paper's smallest claimed rate, in all seven directly verified cells. That
follows from the structure of the result rather than from luck: F has exactly
zero variance, so any injected inflation moves the mean off a point mass.

| injected effect | 0.05% | 0.10% | 0.28% | 1.0% | 5.0% | 11.2% |
|---|---:|---:|---:|---:|---:|---:|
| detection rate, all 7 cells | 100% | 100% | 100% | 100% | 100% | 100% |

**Was inflation available?** For each cell we construct a genuine alternative
tokenization of the same returned string, character by character. It is a valid
segmentation, so its length is a strict lower bound on the longest one
available. The median surface runs **+268% to +401%** across all 30 cells with
stored text, and the smallest anywhere is +153%.

A provider wanting to inflate these exact bills could have multiplied them
several times over before running out of valid tokenizations. The paper's own
0.28% to 11.2% is the share of that surface surviving its top-p plausibility
filter, which needs model logits we do not have. Measuring the unfiltered
surface is the conservative direction, since it establishes that the attack was
structurally available on real commercial output rather than foreclosed by it.

Taken together: inflation was available at a scale far beyond anything the
paper claims, the method would have caught it at a fifth of the paper's
smallest rate with certainty, and it measured exactly zero.

---

## 8. Limitations

- We exclude OpenAI's reasoning models from the fully-verified cells, because
  the provider discards reasoning content server-side and that share of the bill
  has no ground truth. The accounting identity in section 4 is a weaker claim.
- Anthropic discloses thinking tokens and is internally self-consistent, showing
  a constant −10 wrapper offset against its own `count_tokens` with no drift
  across response lengths. However, it publishes neither a tokenizer nor a
  combined total, and self-consistency is not independent verification.
- The Google result uses an open-weight model on Google's API. It proves that
  the hidden-reasoning channel is checkable and honest where we checked it, and
  it says nothing about how Google bills its proprietary Gemini models.
- Model substitution, where a provider silently serves a cheaper model, is a
  different and harder channel requiring trusted execution or zkML. Nothing here
  addresses it.
- Section 5.1 rests on four events across two cells, on the weakest models at
  temperature 1.3. We do not claim that the behavior is provider-specific.
- Direct observation of the generated sequence covers 7 cells. Together,
  OpenRouter, Z.ai and Google expose no logprobs, and DeepInfra returns
  incomplete arrays for Kimi and DeepSeek, so those cells rest on canonical
  comparison and carry the weaker claim. We report the two tiers separately and
  never pool them.

---

## 9. Verification and reproduction

- `audit_integrity.py` runs an independent integrity audit that deliberately
  avoids the analysis path it checks, covering duplicate detection, per-cell
  counts, config ambiguity, independent delta recomputation, and prompt-set
  identity. Current status: **0 problems**. All 30 full cells share an identical
  200-prompt set, there are no duplicates, and no record fails to resolve.
- `logprob_audit.py` performs the direct-observation audit, including the
  completeness gate that decides when F is interpretable.
- `positive_control.py` runs both controls: injected-inflation detection rates
  and the available attack surface.
- `final_report.py` produces all per-cell statistics.
- `results/*.jsonl` holds every request, with the prompt, response, reasoning
  text, billed count, recomputed count, and raw usage.
- `calibrate_cells.py` and `calibrate_round2.py` implement the per-cell format
  calibration that every cell passed before being run.
- `verifiable-billing-spec.md` sets out what a provider can ship to *prove* it
  is not overcharging, in four ascending levels.

---

## 10. What this adds up to

Providers bill honestly, and customers still cannot check.

Every failure this study found is a **verifiability** failure rather than a
dishonesty one, covering tokens that cannot be reconstructed, conventions that
vary invisibly per request, caps that fail to hold, and a schema field left
empty. The paper aimed at a mechanism nobody has any reason to use, and in doing
so it missed the real problem sitting next to it.

That problem has a cheap fix. Because the attack produces a valid *alternative*
segmentation of the same string, only the **canonical** segmentation proves
anything, which means the verifier needs the tokenizer. The ask therefore
reduces to one line: **publish the tokenizer**, though not the weights. OpenAI
already does, which is why its billing is the most verifiable in this study,
while Anthropic and Google do not.
