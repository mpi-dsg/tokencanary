# Is Your LLM Overcharging You? No, and It Was Never Going To

A response to Velasco, Tsirtsis, Okati, and Gomez-Rodriguez, "Is Your LLM
Overcharging You? Tokenization, Transparency, and Incentives," arXiv:2505.21627.

This document does not rebut the theorem, which is correct once you accept its
restrictions. It argues instead that the thing the paper actually sells, a live
warning that pay-per-token pricing creates a standing financial incentive users
should worry about, was never going to appear, for reasons that hold whether or
not anyone catches a provider doing it. We then show that we checked anyway,
with real money, on real infrastructure, at a precision tighter than the
paper's own headline numbers, and found nothing.

We keep the two layers separate so that attacking one cannot dismiss the other.

- **Layer 1, a priori:** no rational provider would deploy this, for structural
  reasons that hold whether or not we ever ran a single query.
- **Layer 2, empirical:** we ran the queries anyway. 12,600 real billed requests
  across eight platforms, at both the headline sampling setting and the single
  setting most favorable to fraud. No cell shows a positive-direction signal, at
  a stated statistical precision tighter than the paper's own lowest claimed
  effect size.

Layer 1 makes Layer 2 unsurprising rather than lucky, and Layer 2 keeps Layer 1
from being a just-so story. Together they close off the retreat.

---

## Layer 1: it would never make sense in the first place

### 1. Revealed preference: the legal alternative strictly dominates

A provider that wants more revenue per token has a completely legal, zero-risk,
unlimited-upside option available at all times, which is to change the sticker
price. This market does so constantly and openly. We found documented API price
movements of 40 to 60% since mid-2025, and gross margins moving from -94% to
+60% at Anthropic and toward a stated 52% target at OpenAI over roughly the same
window, achieved entirely through pricing and efficiency.

Compare the two paths to the same extra dollar of revenue.

| | Raise the price | Run Algorithm 2 |
|---|---|---|
| Legal exposure | None | Wire fraud, FTC Section 5, breach of contract, securities fraud if the false billing data touches revenue recognition at a company raising money on these numbers |
| Engineering cost | A config change | Build and maintain a heuristic that must run *in addition to* generation |
| Detection risk | Zero, since it is disclosed | Nonzero and, per this project, now trivially checkable by any customer with a laptop |
| Upside ceiling | Unlimited | Bounded by the paper's own numbers to low single digits at realistic settings |

A rational profit-maximizing firm does not choose the strictly dominated option.
The claim here concerns firms leaving free, legal money on the table in favor of
a worse and illegal version of the same trade, and it says nothing about
corporate virtue.

There is a sharper version of this point worth stating precisely. The one thing
hidden fraud buys over a transparent price increase is *concealment*, and
concealment only has value when the market would not bear the price
transparently. The fraud therefore works only in the scenario where customers
have already revealed, through ordinary price sensitivity, that they will not
pay this much. That amounts to defeating a functioning competitive price signal
by force, and it remains a worse trade than a transparent increase on every axis
except concealment, which itself only matters when a firm is already trying to
extract money the market has told it that customers will not give.

### 2. Solving an NP-hard problem to make 2-3% is not a rational firm's move

Be concrete about what deploying this actually costs, beyond what it legally
risks. The paper's own Theorem 3 proves that finding the *optimal* plausible
padding is NP-hard. Their Algorithm 2 workaround still requires, for every
response, an extra forward pass to verify top-p plausibility, and their own
Table 1 and Appendix C.3 show that this verification pass costs more energy than
generating the response itself, with measured c_o/c_v ratios of 0.17 to 0.51, or
2 to 6 times the generation cost. It also requires careful tuning of an
iteration count that their own Figure 2 shows has a narrow, unimodal profitable
range per model and per sampling setting. Finally, it requires permanent
integration into a production billing pipeline that feeds revenue recognition,
survives internal code review, and would surface in any future financial audit
or litigation discovery.

All of that is real, ongoing engineering and compute expense, and it leaves its
own internal cost trail on the provider's own infrastructure bills before any
external customer notices anything. A provider would build and maintain it
indefinitely to capture, by the paper's own numbers, **0.28% to 11.2%**, where
the top of that range is reachable only at sampling settings (temperature 1.3,
top_p 0.99) that essentially nobody runs by default.

Meanwhile, raising the posted price by the same percentage costs one edit to a
pricing page, with no verification pass, no NP-hard optimization, no forensic
trail, no crime, no engineering team, and no audit exposure. There is no framing
under which the NP-hard path is the rational one.

### 3. There is no margin desperation at the layer this fraud would target

The standard mental model, which holds that AI labs are burning cash and would
therefore cheat, is directionally wrong at the layer that matters here,
including for the provider most often assumed to be under the most pressure.
OpenAI's adjusted API gross margin, the exact line that pay-per-token pricing
touches, sits at 33 to 52% and is rising toward a stated target. The company's
roughly $14B annual loss comes from training compute and infrastructure capital
expenditure sitting on top of that already-profitable serving layer, and the
serving layer itself is not losing money.

Shaving 2 to 3% off API revenue through a scheme that risks the entire company's
trust-dependent valuation does not touch the actual and much larger loss. It is
a rounding error against the wrong expense line, taken on for catastrophic tail
risk. A firm bleeding cash on GPU clusters and training runs does not fix that
by defrauding customers on token counts. It fixes that by raising prices, which
is already happening publicly along the path 33% to 39% to a 52% target, or by
cutting compute cost, which is also already happening publicly through reported
inference-cost halving in software alone. Neither of those is fraud, and both
are the observed behavior.

The regime argument goes further. The industry is currently in a land-grab
phase, subsidizing usage below fully-loaded cost specifically to buy market
share and the growth metrics that justify the next valuation. A firm doing that
does not simultaneously nickel-and-dime the same users on token counts, which
would undermine the growth story the subsidy exists to build. Anthropic's margin
trajectory, from -94% in 2024 to roughly 60% in 2026, is the observed proof that
this regime resolves through pricing and efficiency, in public, already.

That argument also survives a hypothetical future where margins genuinely
tighten and subsidies end. In that world the dominant move is still the
transparent one, since raising the price carries zero legal risk and unlimited
upside. The NP-hard engineering cost and the catastrophic tail risk of fraud do
not shrink because margins got tighter, and a margin-constrained firm has less
slack to absorb a fraud scandal, so it has less reason to risk one. We can
identify no regime, current or plausible, in which fraud dominates the legal
alternative that achieves the same or greater revenue.

### 4. This is a bright legal line, in an industry whose entire product is trust

The paper's own language, with its talk of misreporting and strategizing, is
game-theory euphemism for billing a customer for compute that was not delivered.
Quietly serving a quantized model is a genuine gray area open to arguments about
specification compliance, and routing to a cheaper backend under load is an
operational judgment call. Fabricating a token count on an invoice is
unambiguous. For a company whose product is "trust this black box," and whose
executives carry personal liability exposure at exactly the moment these
companies are raising capital at valuations that depend on that trust, this is
the worst kind of scandal to be caught in, offering small dollar upside,
existential downside, and no gray area to hide behind once discovered.

### 5. The paper's historical analogies work against it once you look at detectability

The paper invokes Google's second-price auction scandal, Volkswagen, Wells
Fargo, and LIBOR to argue that big, trusted firms do commit fraud. That is true.
However, every one of those cases required an external institution with subpoena
power, whether the EPA, Congress, or federal regulators, to grind through years
of internal documents before the fraud surfaced, and in each case the harm was
diffuse and hard for any individual victim to verify independently.

The fraud this paper describes requires re-encoding a string with a tokenizer
that is, for the exact models the paper studies, public. We built the
counter-check in an afternoon and ran it for about the price of a coffee.
Comparing a fraud that needs a federal subpoena to a fraud any customer can
check with three lines of Python describes a different threat model, and the
paper's analogy section inadvertently proves the point, because the examples it
reaches for are exactly the kind of fraud that this one is not.

### 6. Repeated-game market structure favors honesty

Switching costs are low, since an API call is an API call and OpenRouter exists
specifically to make switching trivial. Providers interact repeatedly with the
same customers at volume, and there is now a public, free auditing tool. All of
this pushes the market toward the cooperative equilibrium of a repeated game
rather than the one-shot-defection equilibrium the paper implicitly models.
Enterprise customers spending enough to matter already run FinOps reconciliation
against these bills as a matter of course, which makes them the hardest targets
to defraud, and they are exactly the revenue that matters.

---

## Layer 2: and then we checked anyway

None of Layer 1 requires an audit to be true. We ran one because a structural
argument alone invites the question of how we know. Full methodology,
deviations, and every dead end are recorded in `implementation-notes.md`, and
the summary follows.

- **Billed equals generated, exactly, wherever the generated sequence can be
  observed.** Comparing a bill to a canonical re-tokenization leaves one term
  unobserved, and it is the term the paper exploits. Providers returning
  per-token `logprobs` remove it: one entry per generated token, no tokenizer
  involved, so the measured quantity is overcharging itself. Across
  **1,394 of 1,394** such requests the result is exactly zero, 870,631 billed
  tokens against 870,631 observed. A positive control confirms this is
  meaningful: injected inflation is caught in 100% of replicates down to 0.05%,
  and the attack surface actually available on these outputs runs +268% to
  +401%.
- **8 real commercial platforms**, comprising four resellers (Together,
  Fireworks, DeepInfra, OpenRouter) and four first-party APIs (Z.ai, OpenAI
  non-reasoning, OpenAI reasoning, and Google). The first-party cells matter
  more than the resale ones, because they test whether the labs that actually
  train and price these models bill their own customers honestly. These are real
  invoices and real money, rather than a simulation on the authors' own
  hardware, which is what the original paper's entire empirical section consists
  of. We confirmed that by reading the paper directly, and again by reading
  every follow-up paper in this line of work, including the same authors' second
  paper, which explicitly marks real-provider evaluation as "out of scope of the
  current work."
- **The hidden-reasoning channel is verified rather than assumed.** Every
  auditing framework in this literature treats reasoning tokens as unverifiable,
  since the provider bills for chain-of-thought that the customer never sees.
  Google serves the open-weight `gemma-4-31b-it` on its own API and returns the
  raw thought text alongside a separate `thoughtsTokenCount`, which makes that
  channel directly checkable. Across 200 requests, **50.7% of billed output
  tokens were hidden reasoning**, amounting to 83,838 tokens, and the billed
  count matched the re-tokenized thought text on **194 of 200 responses**, at a
  mean error of +0.015 tokens. The provider's own arithmetic reconciled 200/200.
  This is the channel that matters in 2026, and it is honest.
- **12,600 real billed requests across 33 distinct (provider, model) cells**,
  spanning seven model families (Llama, Gemma, gpt-oss, DeepSeek, Kimi, GLM, and
  GPT) at the paper's own headline sampling setting of temperature 1.3 and top_p
  0.95, plus a full second pass at **top_p 0.99, their single most
  favorable-to-fraud setting**, which is where their highest reported number of
  11.2% comes from. Moving to their most extreme setting changed nothing.
- **Exactly one cell in 51 runs exceeded its own detection threshold, and it is
  demonstrably not inflation.** `deepinfra/kimi` at p=0.99 came in at +0.602
  tokens per response. A per-response EOS rule could have recalibrated that away,
  and doing so is how any finding gets turned into a null result, so instead we
  applied the test that separates the two hypotheses. Proportional inflation must
  hold its *percentage* constant while the absolute token count grows with
  response length, and a fixed convention token must do the opposite. Measured
  across length buckets, the absolute delta stays flat at roughly half a token,
  between +0.56 and +0.79, while the percentage decays from 0.497% to 0.070%,
  and the maximum delta anywhere in the cell is **3 tokens**. What we are
  measuring is an end-of-turn convention token billed on about half of that
  model's responses. The paper's smallest claimed rate of 0.28%, applied
  systematically, would be 2 to 78 extra tokens on a 700-token response and would
  grow with length, so this test separates the two cases cleanly.
- **No cell, at any setting, shows a mean overcharge exceeding its own minimum
  detectable effect.** Eleven cell-runs sit at exactly zero, matching the
  provider's billed count on 100.0% of informative requests. Across the
  expansion block, 25 of 1,397 informative samples, or 1.8%, run in the
  overcharging direction, and every one of them does so by 1 or 2 tokens against
  a comparable tail of the same size in the undercharging direction. That is a
  symmetric one-token noise floor. The paper's threat model predicts systematic
  inflation of 0.28% to 11.2%, which on a 300-token response is roughly 1 to 33
  extra tokens on every request.
- **Only non-truncated responses are counted.** A response billed at exactly
  `max_tokens` was cut off, so its billed count is pinned to the cap and cannot
  exceed it whatever the provider does. Such samples carry no evidence in either
  direction, and we exclude them from every statistic rather than padding the
  reported n. The full breakdown is in `final_report.py`.
- **Every nonzero delta is traced to a specific, verified cause.** The model's
  own EOS token was being correctly billed and wrongly excluded from our
  first-pass ground truth, which we fixed. OpenAI's Harmony reasoning format and
  Google's Gemma thinking-mode format both bill structural tokens that our naive
  text concatenation missed, which we fixed and verified against each format's
  real vocabulary and, where available, its official documentation. One backend,
  Cloudflare, mislabels max-token-truncated responses as "stop" instead of
  "length," which we found, fixed, and applied defensively everywhere. What
  remains is ordinary BPE re-tokenization noise on degenerate high-temperature
  text, the same tokenization multiplicity the literature already documents,
  confirmed by reading the actual garbled text rather than inferred.
- **Statistical power is stated rather than implied.** The minimum detectable
  effect ranges from 0.01% to 0.21% of response length, holding at both sampling
  settings. Even our noisiest cell, OpenRouter-llama at 0.21% with p=0.99, has
  more power than the original paper's smallest claimed inflation rate of 0.28%
  at p=0.90. We are not saying that we failed to notice anything. We are saying
  that we could have detected an effect smaller than their most conservative
  number, in every cell, at both their headline and most extreme sampling
  settings, and we did not. The computation is in `final_report.py`.
- **We checked for selective fraud as well as average fraud.** If a provider ran
  Algorithm 2 on a subset of traffic, the delta distribution should be bimodal,
  showing an honest cluster alongside a padded one rather than unimodal noise.
  Every cell's histogram shows one mode with a smooth decay around it. We also
  report the formal bimodality coefficient even where it is spuriously high on
  near-degenerate discrete data, and explain why that statistic misleads here,
  rather than omitting an inconvenient number.
- **Direction matters.** Where noise exists it runs negative, meaning the
  canonical recount meets or exceeds the billed count, and never the reverse at
  any meaningful scale. On this data, providers are marginally undercharging
  relative to a strict re-tokenization.

---

## What this does and does not claim

**We do not claim** that the theorem is false, that NP-hardness is not
NP-hardness, or that no provider anywhere has ever produced a mismatched token
count under any sampling condition on any model. No finite audit proves a
universal negative, and pretending otherwise would hand back exactly the kind of
overclaim this project was built to avoid.

**We do claim** that the paper's title and motivating framing, which present a
live and economically rational vulnerability that users should worry about
today, is false on both counts available to check it against. Structurally, no
rational actor would run this scheme given the legal alternative, the
verification-cost economics, the margin picture, and the reputational asymmetry.
Empirically, across real commercial infrastructure, real money, the paper's own
most extreme sampling setting, and a stated detection precision tighter than
their smallest reported number, it is not happening.

One retreat position is worth naming precisely, because it is the only one that
survives contact with everything above. Mathematical possibility cannot be
disproven, because irrational firms sometimes do irrational things, and
Volkswagen and Wells Fargo establish that much on their own. That is true and
needs no arguing away. What it does not establish is the paper's implicit
justification for why a rational profit-maximizer would deploy this, namely that
the cost of the algorithm is lower than the additional revenue and that rational
actors therefore do it. Layer 1 shows that justification is wrong on the cost
side, where it ignores the NP-hard engineering burden, the verification-cost
economics, and the legal exposure entirely, and wrong on the regime side, where
it assumes a margin desperation that does not match the current subsidized-growth
reality, and where even a hypothetical future margin-constrained regime leaves
the legal move dominant, which is the move we can already observe in public.

Nobody needs to prove a universal negative to say that. It takes showing that
the specific mechanism the paper offers for why a rational firm would do this
fails under every regime we can identify, and then checking, with real money on
real infrastructure, that it is not happening anyway.
