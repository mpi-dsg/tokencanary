# Token billing audit

An empirical rebuttal of Velasco et al., *"Is Your LLM Overcharging You?
Tokenization, Transparency, and Incentives"* (arXiv:2505.21627).

**12,600 real billed API requests, 33 (provider, model) cells, 8 platforms.**
On every request where the generated token sequence can be observed directly,
the billed count matches it exactly.

Start with **[docs/FINDINGS.md](docs/FINDINGS.md)**, which carries the argument
and every headline number.

## The result in one paragraph

The paper proves that a provider could overcharge by reporting a different but
valid tokenization of the same output string. It never tested whether anyone
does, and neither did any follow-up work in that line. We tested it. Where the
generated sequence is observable through per-token `logprobs`, billed equals
generated on 1,394 of 1,394 requests, exactly. A control confirms the attack was
available on those outputs at a surface of +268% to +401%, and that injected
inflation would have been caught at a fifth of the paper's smallest claimed
rate with certainty. What we did find, instead of fraud, is that customers
mostly cannot verify their bills at all.

## Layout

```
docs/        the argument and all reported numbers
audit/       collection, calibration, and analysis code
results/     every request, as JSONL
investigations/  targeted diagnostics, excluded from all statistics
```

## Documents

| File | What it is |
|---|---|
| **[FINDINGS.md](docs/FINDINGS.md)** | **The argument.** Claim, method, results, what we found, why we expected it, and the limits. Read this first. |
| [results-summary.md](docs/results-summary.md) | Every per-cell number, and why cells differ from one another. |
| [verifiable-billing-spec.md](docs/verifiable-billing-spec.md) | What a provider can ship to prove it is not overcharging, in four levels. |
| [the-case-against.md](docs/the-case-against.md) | The longer structural argument: economics, incentives, legal exposure. |
| [implementation-notes.md](docs/implementation-notes.md) | The lab record. Every deviation, dead end, bug, and withdrawn conclusion, in order. This is the audit trail rather than the argument. |

## Reproducing the numbers

```bash
make setup      # venv + pinned dependencies
make report     # per-cell statistics
make audit      # independent integrity audit
make bounds     # simultaneous upper bounds on inflation
make control    # positive control
```

Every one of those reads only the committed `results/*.jsonl` and **needs no
API keys**, so a reader can re-derive each number offline. Keys are required
only to collect new data.

## Code

Run scripts from the repository root, as `python audit/<script>.py`.

**Analysis**

| Script | Purpose |
|---|---|
| `final_report.py` | Per-cell statistics, the source of every table in the docs. |
| `audit_integrity.py` | Independent integrity audit: duplicates, counts, config ambiguity, delta recomputation, prompt-set identity, and the completeness gate on the direct-observation block. It avoids importing the analysis path it checks. Current status: 0 problems. |
| `equivalence_bounds.py` | One-sided upper bounds on inflation with Bonferroni multiplicity control, replacing the earlier minimum-detectable-effect argument. |
| `positive_control.py` | Detection rates for injected inflation, and the attack surface actually available on real outputs. |
| `power_and_shape_analysis.py` | Distribution-shape diagnostics. |

**Collection**

| Script | Purpose |
|---|---|
| `logprob_audit.py` | Direct observation of the generated sequence, including the completeness gate that decides when the measurement is interpretable. This produces the strongest evidence in the study. |
| `audit.py` | Main runner for OpenAI-compatible platforms, with content-matched resume and per-cell token caps. |
| `google_native_audit.py` | Google's native API, where hidden reasoning becomes checkable. |
| `openai_reasoning_audit.py` | GPT-5.6, where reasoning content is withheld and only an accounting identity is available. |
| `closed_api_accounting.py` | Cross-provider usage reconciliation. Needs no tokenizer, so it works against Anthropic and Gemini too. |
| `calibrate_cells.py`, `calibrate_round2.py` | Per-cell format calibration. Every cell passed this before running, because reconstruction formats do not transfer between providers. |
| `probe_new_models.py`, `discover.py` | Pre-flight servability checks. Catalog presence does not mean a model is servable. |

**Configuration**

`config.py` holds every cell definition. `canonical_tokenizer.py` does
ground-truth counting, including the reasoning-format reconstructions and three
non-standard tokenizer loaders. `providers.py` is a thin OpenAI-compatible
client. `prompts.py` and `env.py` handle prompt sourcing and key loading.

Superseded but working: `analyze.py`, `recompute_analysis.py`,
`investigate_openrouter_llama.py`. Nothing in them is missing from
`final_report.py`.

## Data

| Path | Contents |
|---|---|
| `results/lp_*.jsonl` | Direct observation of the generated sequence (1,800) |
| `results/full_*.jsonl` | Main study, top_p 0.95 (1,800) |
| `results/p99_*.jsonl` | Main study, top_p 0.99 (1,600) |
| `results/exp_*.jsonl` | DeepSeek/Kimi/GLM/GPT expansion, top_p 0.95 (2,600) |
| `results/exp99_*.jsonl` | The same expansion at top_p 0.99 (2,400) |
| `results/r2_*.jsonl` | Model-choice fixes (1,800) |
| `results/oai_*.jsonl` | GPT-5.6 reasoning models (400) |
| `results/gnative_*.jsonl` | Google native, hidden-reasoning verification (200) |
| `investigations/` | Diagnostics, excluded from every statistic |

Every record carries the prompt, response text, reasoning text, billed count,
recomputed count, the `max_tokens` in force, the finish reason, and raw usage,
so any claim can be re-derived from source rather than taken on trust.

## Keys

`.env` is not in this repository and must never be committed, because it holds
live keys for seven paid accounts. `.env.example` lists the variable names, and
the Z.ai key is read from `~/.zai` when that file exists.

Collecting new data needs accounts at Together, Fireworks, DeepInfra,
OpenRouter, OpenAI, Google AI Studio, and Z.ai, plus a Hugging Face token with
gated-repo access for the Llama and Gemma tokenizers.
