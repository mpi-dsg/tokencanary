# tokencanary

Checks that the output tokens you are billed for match the text you received.

The same text can be split into tokens many ways, so a provider can report a longer
split (`[D][a][m][a][s][c][u][s]` instead of `[Dam][ascus]`) and bill for it.
tokencanary audits every Chat Completions response and alerts when the bill is inconsistent.
It runs no model locally. The likelihood test scores tokens through a second provider that
serves the same weights.

## Use

Swap the HTTP transport of the OpenAI client:

```python
import tokencanary
from openai import OpenAI, AsyncOpenAI

client = OpenAI(http_client=tokencanary.http_client())
aclient = AsyncOpenAI(http_client=tokencanary.async_http_client())
```

Or, for non-Python clients, run a localhost proxy:

```bash
tokencanary proxy --upstream https://api.openai.com
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

Every audit is appended to `~/.tokencanary/audit.jsonl`; `tokencanary report` summarizes it.
Alerts go to the `tokencanary` logger and to `Auditor(on_alert=f)`, called as `f(kind, detail)`
with kind `"response"`, `"likelihood"` or `"count"`.

## Checks

| Check | Needs | Detects | Alerts |
|---|---|---|---|
| Output limit | usage | billed or reported tokens above `max_tokens` | every violation |
| Token count | per-token log-probs | billed differs from the number of returned tokens | every mismatch |
| Canonical count | tokenizer | nothing on its own; the bill under canonical billing | never, reported only |
| Count rules | tokenizer, a trusted period of the endpoint | unusually large billed − canonical, for responses without log-probs | per endpoint |
| Likelihood | per-token log-probs, tokenizer, scoring provider, calibration | a padded token list that matches the bill | per endpoint |

The transport adds `logprobs: true` (and usage for streams) to requests and strips it
from the response your code sees. If a model rejects that, the original request is resent.

Some providers return log-probs for the whole generation, including reasoning and channel
tokens; then the token count covers everything billed and the likelihood test uses the part
that spells the answer. Others return log-probs for the answer only while billing hidden
reasoning or channel tokens (gpt-oss); then the token count is skipped. Token strings that stand for part of a multi-byte
character (shown as `�`) are aligned to the text in every possible way, and the reading most
favorable to the provider is used. Responses cut at the output limit are counted but not
likelihood-tested, in calibration as in audits.

## False alarms

Honest models sometimes emit unusual splits, so about `alpha` of honest responses fail the
likelihood test by design. A single rejection is logged, never alerted. Alerts are per
endpoint: a sequential test that the rejection rate exceeds `tolerance × alpha` alerts when
its evidence reaches `1 / confidence` and the worst-case overcharge (every rejection counted
as padding) exceeds `min_overcharge`. The count rules feed the same test. Evidence is kept in
`~/.tokencanary/state.json` across restarts.

The remaining risk is calibration that does not match the endpoint. A calibration file records
the tokenizer, scorer and sampling settings it was computed with. Responses whose tokenizer,
scorer, temperature or top-p differ are not tested; a different system prompt gives a warning.
tokencanary also warns when a calibration set is too small for the chosen `alpha` and
`tolerance`, and when the scoring provider is the audited provider.

## Configuration

`~/.tokencanary/config.toml`, `$TOKENCANARY_CONFIG`, or `Auditor(config=path)`. Keyword
arguments to `Auditor` override the file. See [examples/config.toml](examples/config.toml):

```toml
alpha = 0.01
tolerance = 3
confidence = 1e-6
min_overcharge = 0.001
calibration = "~/.tokencanary/calibration.json"

[scorers."openai/gpt-oss-120b"]
model = "accounts/fireworks/models/gpt-oss-120b"

[tokenizers]
"my-finetune" = "hf:Qwen/Qwen2.5-7B-Instruct"

[models."meta-llama/*"]
tolerance = 5
```

**Tokenizers** are resolved per model name: user `[tokenizers]` patterns first, then the
bundled [registry](src/tokencanary/registry.toml) (OpenAI families, common provider names),
then `hf:<name>` for `org/model` names. Patterns are case-insensitive globs; the most specific
one wins. If returned tokens are missing from the chosen vocabulary, the response is marked
unverifiable and tokencanary warns that the entry is probably wrong.

## Likelihood test

The scorer sends token ids to a second provider that serves the same weights and reads back
their log-probs (`echo`). It needs an endpoint that accepts token-id prompts; Fireworks does,
OpenRouter does not. The key is read from `FIREWORKS_API_KEY`, or the variable named by
`api_key_env`. Each span where the reported and canonical tokenizations differ is scored in a
local window, three times, and the median is used.

Calibrate per language with the endpoint's system prompt and sampling settings. The command
samples honest responses from the scoring provider and scores them like audited ones:

```bash
tokencanary calibrate --served-model openai/gpt-oss-120b \
  --scorer-model accounts/fireworks/models/gpt-oss-120b \
  --prompts prompts_de.txt --language de --top-p 0.95 --out ~/.tokencanary/calibration.json
```

Calibrate the count rules on a period in which you trust the audited endpoint, from its audit log:

```bash
tokencanary calibrate-counts --model my-model --host api.example.com \
  --since 2026-09-01 --until 2026-09-14 --out ~/.tokencanary/calibration.json
```

The likelihood test scores local windows around divergent spans (the paper's cross-provider
statistic), not the full prompt: a customer cannot see how the endpoint serializes the prompt.

Pass the language per request with `extra_headers={"x-tokencanary-language": "de"}`. Without it,
responses are keyed by their dominant script (latin, cjk, arabic, ...). When several token ids
share the same bytes, the reading the scoring model finds most likely is used.

Live check with your keys: `uv run python examples/live_check.py`.

## Not implemented

- The paper's aggregate stream test and its bound on undetected additions. They need the model's
  full next-token distributions at every step, which scoring APIs do not return.
- Padding that stays above the threshold on every response therefore passes. In the paper's
  evaluation a test-aware provider hid up to 1.4% of generated tokens this way; that figure is
  a lower bound from an incomplete search.
- Closed models served by a single provider cannot be scored independently.
- Endpoints without per-token log-probs (e.g. Claude, Gemini) get only the output-limit check,
  the canonical count and the count rules.
- Hidden reasoning tokens, cache accounting, and model substitution are not checked.
