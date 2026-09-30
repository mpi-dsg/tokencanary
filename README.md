# Checking the Token Bill: Retokenization Overbilling Is Smaller Than Reported, and Auditable

This repository contains the code and data for the paper.

Language-model APIs charge by the token, but the tokens never leave the provider. The customer
sees only text and a count. One text has many tokenizations. A model that writes `tokenizer`
emits `[token][izer]`, but a provider could report `[t][o][k][e][n][i][z][e][r]` and bill four and
a half times as much, and the response the customer receives would be identical. Prior work
([Velasco et al., arXiv:2505.21627](https://arxiv.org/abs/2505.21627)) presents this as an
opportunity for undetectable billing fraud of up to 11.2%.

The paper treats this as an auditing problem. It asks what a customer can verify, with what access,
and whether any provider overbills today.

## Results

- **An inflated split is improbable under the provider's own model.** A provider's model gives a
  padded tokenization a much lower probability than the tokenization it actually generated.
  Scoring the billed split against the canonical one, with a threshold calibrated on honest
  responses, rejects every greedy report and all but 4 of 878 published-heuristic reports. It
  does this at a nominal 1% false-positive rate. The held-out false-positive rate is 0.8%.
- **No misreporting in production.** On 10,286 requests to 9 endpoints from 4 providers in 5
  languages, the billed count equals the reported generated count everywhere. On the 3 endpoints
  that can be scored independently of the provider that bills, likelihood rejections stay near
  the nominal rate.
- **Bounded even against a test-aware provider.** A provider that knows the threshold can pad
  only by keeping every response above it. That hides 0.00–1.36% of generated tokens per
  response. Accumulating evidence across a stream bounds undetected additions at a median 1.151%
  of generated tokens over 1,000 requests, with no assumption about the provider's strategy.
- **Smaller than reported.** Raw prompts reproduce the published scale: 11.77% against the
  published 11.2%. Under the chat template that serving APIs actually apply, the same metric
  gives 1.14–1.75%.
- **Billing by the canonical tokenization removes the incentive.** Charging for the tokenizer's
  own split of the returned text is incentive-compatible for string-preserving reports.
- **An earlier billing audit agrees.** It covers 12,600 billed requests on 33 provider–model
  pairs. Wherever per-token log-probabilities expose the generated sequence, the bill equals the
  generated length on 1,394 of 1,394 requests. Elsewhere, simultaneous upper bounds exclude excess
  above 0.28% of the bill in 47 of 50 settings. The audit also found 24 requests billed above the
  client's `max_tokens`, all on one gateway.

What a customer can check depends on access. The tokenizer alone gives the count checks. Per-token
log-probabilities make the billed split scorable. Catching a provider that falsifies those reports
needs a third party that holds the same weights.

## Repository layout

| Directory | Contents | License |
|---|---|---|
| [`token-billing/`](token-billing/) | `tokencanary`, the auditor that applies the paper's checks to live API traffic | Apache-2.0 |
| [`tokenization-overcharging/`](tokenization-overcharging/) | The earlier billing audit: collection and analysis code, and every request as JSONL | MIT |

The two directories are independent. Each has its own README and dependencies.

## tokencanary

`tokencanary` audits every chat completion a client receives. It runs no model locally. It
replaces the HTTP transport of the OpenAI client, so application code does not change:

```bash
cd token-billing
uv sync
```

```python
import tokencanary
from openai import OpenAI, AsyncOpenAI

client = OpenAI(http_client=tokencanary.http_client())
aclient = AsyncOpenAI(http_client=tokencanary.async_http_client())
```

Clients in other languages can use the local proxy:

```bash
uv run tokencanary proxy --upstream https://api.openai.com
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

The transport requests per-token log-probabilities, removes them from the response your
application sees, and passes the request and response to the auditor. Every audit is appended to
`~/.tokencanary/audit.jsonl`, and `tokencanary report` summarizes the log per endpoint.

| Check | Requires | Detects | Alerts |
|---|---|---|---|
| Output limit | usage, requested limit | a bill or report above the limit | per response |
| Token count | log-probabilities covering the generation | a bill that differs from the reported tokens | per response |
| Canonical count | tokenizer | nothing; records the canonical bill | never |
| Count-only rules | tokenizer, counts from a trusted period | excess billed − canonical without log-probabilities | per endpoint |
| Likelihood test | log-probabilities, tokenizer, scoring provider, calibration | a padded report that matches the bill | per endpoint |

The count checks need only the tokenizer and add no API calls. The likelihood test scores the
reported and canonical token ids through a second provider that serves the same weights, so it
also catches a provider whose padded report matches its own bill. About α of honest responses are
rejected by design. Alerts are therefore raised per endpoint, by a sequential test on the
rejection rate.

The paper's aggregate stream test is not included. It needs the model's full next-token
distribution at every position, which scoring APIs do not return. Endpoints without per-token
log-probabilities, and closed models that only one provider serves, get only the count checks.

The [tokencanary README](token-billing/README.md) covers configuration, tokenizer resolution and
calibration. Run the tests with `uv run pytest`.

## Reproducing the earlier billing audit

The analysis reads only the committed `results/*.jsonl` and needs no API keys:

```bash
cd tokenization-overcharging
make setup     # venv with pinned dependencies (uv, Python 3.13)
make report    # per-cell statistics
make audit     # independent integrity audit of the dataset
make bounds    # simultaneous upper bounds on excess over canonical billing
make control   # positive control: detection rates and attack surface
```

Collecting new data requires API keys, listed in
[`.env.example`](tokenization-overcharging/.env.example). The
[study README](tokenization-overcharging/README.md) describes each script and data file.

## Scope

The audit covers unchanged text and verifiable quantities: retokenization, output counts, and
billing past the output limit. It does not cover the following:

- Hidden-reasoning counts, cache-hit charges and model substitution need separate audits.
- Scoring shows that a split is improbable under a reference model. It cannot show which weights
  actually served the request.
- Agreement between counts alone cannot rule out a provider that falsifies its token reports to
  match its bills. Catching that needs independent scoring.
