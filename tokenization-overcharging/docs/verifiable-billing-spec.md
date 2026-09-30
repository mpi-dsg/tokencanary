# Verifiable Token Billing: what a provider can ship to prove it is not overcharging

This is a companion to *"Is Your LLM Overcharging You? No, and It Was Never
Going To."* That document argues that the threat is neither occurring nor
rational, and this one addresses the question that survives it, namely that a
customer still cannot check.

Those are two different problems, and the original literature conflates them.
Our audit found honest billing everywhere we could measure it. However, "we
measured it and it was fine" is not a property that a customer can verify on
their own invoice, on a Tuesday, without running a research project. What is
missing is verifiability rather than integrity.

The useful part is that **the fix is nearly free, and one major lab has already
shipped most of it.**

## What actually has to be proven

The naive proposal, which is to return the token boundaries so that customers
can count them, does not work, and the reason is worth stating precisely
because it comes from the original paper's own attack.

Padding produces a **different but perfectly valid** segmentation of the **same
output string**. Disclosed boundaries would still concatenate to exactly the
text the customer received, every boundary would be a real token in the
vocabulary, and nothing in the receipt would look wrong.

The property that must be established is therefore stronger than "these tokens
spell the output." It is this:

> **the billed segmentation is the one this model's tokenizer actually produces
> for this string**, which is to say the canonical segmentation rather than
> merely a valid one.

Checking that requires the verifier to hold the tokenizer, which leads to the
entire recommendation:

> **Publish the tokenizer**, though not the weights. The tokenizer is not the
> valuable intellectual property, and shipping it converts token billing from
> unverifiable into independently checkable.

## Four levels, in increasing order of provider effort

### Level 0: publish the tokenizer (cost: approximately zero)

Ship the tokenizer as a downloadable artifact, versioned per model.

**OpenAI already does this**, through `tiktoken` and `o200k_base`, while
Anthropic and Google do not for their proprietary models. This single step is
what allowed us to verify OpenAI's billing to the token, at 386 of 386
informative samples exact.

Nothing else on this list matters as much. Levels 1 through 3 are refinements,
and Level 0 is what separates a checkable bill from an uncheckable one.

### Level 1: populate the reasoning-token field you already compute

When a model spends tokens on hidden reasoning, disclose that count in the
documented field, which is `completion_tokens_details.reasoning_tokens` in the
OpenAI schema and `output_tokens_details.thinking_tokens` in Anthropic's.

This is not hypothetical housekeeping, because we found a live instance.
**Google's OpenAI-compatibility endpoint leaves `reasoning_tokens` empty**, so
`prompt_tokens + completion_tokens` never equals `total_tokens` across the 8
sampled requests, leaving a median of roughly 200 tokens per response
unaccounted for. Google's native endpoint returns `thoughtsTokenCount` and
reconciles exactly, so the provider already computes the number and simply does
not surface it on that interface.

The consequence is concrete. Any auditing tool built against the
OpenAI-compatible interface, which is the industry norm, will systematically
mis-account Google's billing.

### Level 2: emit a billing receipt

For each response, return the token boundary offsets over the generated string.

```json
"billing_receipt": {
  "tokenizer": "o200k_base",
  "tokenizer_sha256": "…",
  "segments": [[0, 5], [5, 11], [11, 12]],
  "reasoning_segments": 118,
  "billed_tokens": 141
}
```

With Level 0 in place, a customer can verify three things offline and without
trusting the provider: that the segments concatenate to the received text, that
the segment count equals the billed count, and that re-tokenizing the text with
the published tokenizer reproduces exactly these boundaries. That last check is
the one that closes the paper's attack, and it is possible only because of
Level 0.

For reasoning models the receipt should cover the reasoning stream as a count
even where the content is withheld. The customer cannot check the reasoning
text, and the totals must still reconcile.

### Level 3: sign the receipt

Sign the receipt with a published key, so that it becomes non-repudiable
evidence rather than a courtesy. This is the only level that survives a
genuinely adversarial provider, and the only one that requires real
engineering. It is also the only level worth arguing about, since Levels 0
through 2 disclose values the provider already holds.

## What the verifier looks like

We have already built it, in this repository.

- `audit.py` and `canonical_tokenizer.py` implement Level 0 verification,
  recomputing the canonical count from the returned text and diffing it against
  the bill. They work today against any provider with a published tokenizer,
  including OpenAI.
- `closed_api_accounting.py` implements Level 1 verification, checking whether
  the provider's own published numbers reconcile and whether the reasoning
  field is populated. It needs no tokenizer at all, so it also works against
  Anthropic and Gemini's proprietary models.
- `google_native_audit.py` demonstrates that Levels 0 and 1 together are
  sufficient. Because Google serves an open-weight model and discloses thought
  counts, hidden-reasoning billing became fully checkable, at 194 of 200 exact
  across 83,838 billed reasoning tokens.

That last result is the existence proof worth leading with. This literature
treats hidden reasoning-token billing as unverifiable, and it is unverifiable
only because tokenizers go unpublished. Where one is published, the check is
trivial, and the answer came back clean.

## Why a provider would adopt this

There are three reasons, and none of them is altruism.

First, it is the cheap half of a trust problem these providers already have.
Levels 0 through 2 disclose values the provider already computes, and
enterprise procurement increasingly asks for billing auditability that nobody
can currently supply.

Second, it forecloses the accusation permanently. This paper got attention
because the claim is unfalsifiable from outside, and a provider at Level 2 can
point at the receipt instead of at a press release.

Finally, one competitor is already most of the way there. OpenAI ships the
tokenizer and populates the reasoning field, which is a differentiator today
and will be a baseline expectation shortly.

## The honest limitation

None of Levels 0 through 2 stops a determined adversary, since a provider
willing to lie about token counts can lie about receipts as well. What they do
is convert an undetectable scheme into a detectable one, and detection is what
changes incentives, for the same reason that audited financial statements
matter despite audits being imperfect.

Level 3 raises the bar to non-repudiation. Genuine attestation of which model
actually ran, which is the model-substitution channel, is a separate and much
harder problem requiring TEE or zkML, and nothing here addresses it.
