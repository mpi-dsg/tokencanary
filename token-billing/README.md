# tokencanary

Checks that the output tokens you are billed for match the text you received.

The same text can be split into tokens many ways, so a provider can report a longer
split (`[D][a][m][a][s][c][u][s]` instead of `[Dam][ascus]`) and bill for it.
tokencanary audits every Chat Completions response and alerts when the bill is inconsistent.

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
Alerts go to the `tokencanary` logger and to `Auditor(on_alert=f)`, called as `f(kind, detail)` with kind `"response"` or `"provider"`.

## Checks

| | Needs | Check | Alerts |
|---|---|---|---|
| Cap | nothing | billed ≤ `max_tokens` | every violation |
| Count | nothing | billed = number of per-token log-probs, when they spell the returned text | every mismatch |
| Shadow bill | tokenizer | canonical token count of the text, the bill under canonical billing | never, reported only |
| Likelihood | reference weights + calibration | `log p(reported) − log p(canonical)` below the calibrated threshold | provider-level only |

The transport adds `logprobs: true` (and usage for streams) to requests and strips it
from the response your code sees. If a model rejects that, the original request is resent.

## False alarms

Honest models sometimes emit unusual splits, so about `alpha` of honest responses fail the
likelihood test by design. A single rejection is logged, never alerted. The alert is on the
provider: an anytime-valid test that its rejection rate exceeds `tolerance × alpha`, which
fires when the evidence reaches `1 / confidence` and the worst-case overcharge (every
rejection counted as padding) exceeds `min_overcharge`. An honest provider whose true
rejection rate is at most `tolerance × alpha` is ever flagged with probability at most
`confidence`, however long it is monitored. Evidence is kept in `~/.tokencanary/state.json`
across restarts.

The remaining risk is calibration that does not match the endpoint. tokencanary warns when
a calibration set is too small for the chosen `alpha` and `tolerance`.

## Configuration

`~/.tokencanary/config.toml`, `$TOKENCANARY_CONFIG`, or `Auditor(config=path)`. Keyword
arguments to `Auditor` override the file. See [examples/config.toml](examples/config.toml):

```toml
alpha = 0.01
tolerance = 3
confidence = 1e-6
min_overcharge = 0.001

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

Scores use local weights, which must be the weights the provider serves. Calibrate per
language with the endpoint's system prompt and sampling settings:

```bash
tokencanary calibrate --model-id Qwen/Qwen2.5-7B-Instruct --served-model my-model \
  --prompts prompts_de.txt --language de --top-p 0.95 --out ~/.tokencanary/calibration.json
```

```python
from tokencanary.scoring import LocalHFScorer

auditor = tokencanary.Auditor(scorers={"my-model": LocalHFScorer("Qwen/Qwen2.5-7B-Instruct")},
                              calibration="~/.tokencanary/calibration.json")
client = OpenAI(http_client=tokencanary.http_client(auditor))
client.chat.completions.create(..., extra_headers={"x-tokencanary-language": "de"})
```

Without a language header, responses are keyed by their dominant script (latin, cjk, arabic, ...).
When several token ids share the same bytes, the reading the model finds most likely is used.

Demo with a padding provider, no API key: `uv run --extra local python examples/local_attack_demo.py`.
Live check with your keys: `uv run python examples/smoke_test.py`.

## Limits

- Without the likelihood test, a provider that fakes a log-prob array matching its bill passes.
- Padding that stays above the threshold passes. That is a small share of tokens, but not zero.
- Endpoints without per-token log-probs (e.g. Claude, Gemini) get only the cap check and the shadow bill.
- Hidden reasoning tokens, cache accounting, and model substitution are not checked.
- A constant billed − returned offset on one endpoint can come from structural tokens the log-probs omit.
