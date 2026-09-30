"""Tier-2: billing-accounting transparency test for closed / frontier APIs.

For open-weight models we can recompute a canonical token count and compare it
to the bill. For closed models we usually cannot. But there is a check that
needs no tokenizer at all and that no existing auditing framework in this
literature performs: **does the provider's own usage report internally
reconcile?**

Specifically:
  1. Reconciliation -- do the disclosed parts sum to the disclosed total?
     A total that exceeds prompt + completion means tokens are being counted
     somewhere the customer cannot see from those two fields.
  2. Reasoning disclosure -- when a model spends tokens on hidden reasoning,
     is that count surfaced in the documented field
     (`completion_tokens_details.reasoning_tokens` in the OpenAI schema), or
     silently folded into a total?
  3. Logprob cross-check (OpenAI only) -- `logprobs.content` carries one entry
     per generated token, so `len(logprobs.content)` vs `completion_tokens` is
     an internal-contradiction test requiring no external ground truth.

None of these can prove a provider honest. All of them can catch a provider
whose own numbers do not add up, which is a strictly weaker but genuinely
falsifiable claim -- and the point of the tool is to be explicit about which
of the two it is delivering.
"""
import json
import os
import statistics
import sys

import requests

import env

env.ensure_loaded()

from prompts import load_prompts  # noqa: E402

N = int(os.environ.get("N_ACCOUNTING", "8"))
MAX_TOKENS = 800


def _post(url, **kw):
    kw.setdefault("timeout", 240)
    return requests.post(url, **kw)


def openai_probe(model, prompt, want_logprobs):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MAX_TOKENS,
    }
    if want_logprobs:
        body["logprobs"] = True
    r = _post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
        json=body,
    )
    d = r.json()
    if "usage" not in d:
        return {"error": json.dumps(d)[:160]}
    u = d["usage"]
    det = u.get("completion_tokens_details") or {}
    lp = (d["choices"][0].get("logprobs") or {}).get("content")
    return {
        "prompt": u.get("prompt_tokens", 0),
        "completion": u.get("completion_tokens", 0),
        "total": u.get("total_tokens", 0),
        "reasoning_disclosed": det.get("reasoning_tokens"),
        "logprob_entries": len(lp) if lp else None,
    }


def gemini_compat_probe(model, prompt, _):
    r = _post(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["GOOGLE_API_KEY"]},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": MAX_TOKENS},
    )
    d = r.json()
    if "usage" not in d:
        return {"error": json.dumps(d)[:160]}
    u = d["usage"]
    det = u.get("completion_tokens_details") or {}
    return {
        "prompt": u.get("prompt_tokens", 0),
        "completion": u.get("completion_tokens", 0),
        "total": u.get("total_tokens", 0),
        "reasoning_disclosed": det.get("reasoning_tokens"),
        "logprob_entries": None,
    }


def gemini_native_probe(model, prompt, _):
    r = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": os.environ["GOOGLE_API_KEY"]},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"maxOutputTokens": MAX_TOKENS}},
    )
    d = r.json()
    u = d.get("usageMetadata")
    if not u:
        return {"error": json.dumps(d)[:160]}
    return {
        "prompt": u.get("promptTokenCount", 0),
        "completion": u.get("candidatesTokenCount", 0),
        "total": u.get("totalTokenCount", 0),
        "reasoning_disclosed": u.get("thoughtsTokenCount"),
        "logprob_entries": None,
    }


def anthropic_probe(model, prompt, _):
    r = _post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"},
        # Extended thinking must be explicitly enabled, or the model simply
        # does not think and `output_tokens_details.thinking_tokens` is absent
        # -- which reads as "no disclosure" when it actually means "nothing to
        # disclose". A first version of this script missed that and unfairly
        # recorded Anthropic as not disclosing reasoning tokens.
        json={"model": model, "max_tokens": 2048,
              # Anthropic requires budget_tokens >= 1024 AND
              # max_tokens > budget_tokens, so this cell cannot run at the
              # shared MAX_TOKENS=800. Overridden here rather than raising the
              # global value, which would change every other target's regime.
              "thinking": {"type": "enabled", "budget_tokens": 1024},
              "messages": [{"role": "user", "content": prompt}]},
    )
    d = r.json()
    u = d.get("usage")
    if not u:
        return {"error": json.dumps(d)[:160]}
    out = u.get("output_tokens", 0)
    return {
        "prompt": u.get("input_tokens", 0),
        "completion": out,
        # Anthropic publishes NO combined total. We synthesize one as
        # input+output purely so the row renders -- which makes the
        # reconciliation column vacuous here: it is arithmetic on our own
        # construction, not a test of anything Anthropic published. Flagged
        # below so it cannot display as a pass.
        "total": u.get("input_tokens", 0) + out,
        "no_total_published": True,
        "reasoning_disclosed": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
        "logprob_entries": None,
    }


TARGETS = [
    ("openai/gpt-4o-mini",       openai_probe,        "gpt-4o-mini",     True),
    ("openai/gpt-5-nano",        openai_probe,        "gpt-5-nano",      False),
    ("google-compat/gemma-4-31b", gemini_compat_probe, "gemma-4-31b-it", False),
    ("google-native/gemma-4-31b", gemini_native_probe, "gemma-4-31b-it", False),
    ("google-native/gemini-2.5-flash", gemini_native_probe, "gemini-2.5-flash", False),
    ("anthropic/claude-haiku-4-5", anthropic_probe,   "claude-haiku-4-5-20251001", False),
]


def main():
    prompts, source = load_prompts(N)
    print(f"prompts: {len(prompts)} from {source}\n")
    print(f"{'target':<32} {'n':>3} {'reconciles':>11} {'hidden(med)':>12} "
          f"{'reasoning field':>16} {'logprob check':>14}")
    print("-" * 96)
    rows = {}
    for label, fn, model, want_lp in TARGETS:
        recs = []
        for p in prompts:
            try:
                res = fn(model, p, want_lp)
            except Exception as e:  # noqa: BLE001
                res = {"error": str(e)[:120]}
            if "error" in res:
                continue
            recs.append(res)
        if not recs:
            print(f"{label:<32} {'--':>3}  NO DATA")
            continue
        # Reasoning-token placement differs by provider and must not be
        # conflated. OpenAI nests reasoning_tokens INSIDE completion_tokens,
        # so completion already covers it. Gemini reports thoughtsTokenCount
        # ALONGSIDE candidatesTokenCount, so it must be added before the sum
        # can balance. Getting this wrong makes an honest, fully-disclosing
        # provider look like it is hiding tokens -- exactly the error class
        # this whole project exists to avoid.
        reasoning_is_additive = label.startswith("google-native")
        hidden = []
        for r in recs:
            extra = (r["reasoning_disclosed"] or 0) if reasoning_is_additive else 0
            hidden.append(r["total"] - r["prompt"] - r["completion"] - extra)
        n_rec = sum(1 for h in hidden if h == 0)
        disclosed = [r["reasoning_disclosed"] for r in recs]
        if all(d is None for d in disclosed):
            dfield = "absent"
        elif any(d for d in disclosed):
            dfield = "present/nonzero"
        else:
            dfield = "present/zero"
        lpc = [r for r in recs if r["logprob_entries"] is not None]
        if not lpc:
            lps = "n/a"
        else:
            agree = sum(1 for r in lpc if r["logprob_entries"] == r["completion"])
            lps = f"{agree}/{len(lpc)} match"
        if recs[0].get("no_total_published"):
            rec_col, hid_col = "n/a (no total)", "n/a"
        else:
            rec_col = f"{n_rec}/{len(recs)}"
            hid_col = f"{statistics.median(hidden):.0f}"
        print(f"{label:<32} {len(recs):>3} {rec_col:>11} "
              f"{hid_col:>12} {dfield:>16} {lps:>14}")
        rows[label] = {"n": len(recs), "reconciles": n_rec,
                       "median_hidden": statistics.median(hidden),
                       "reasoning_field": dfield, "logprob": lps}
    with open("results/closed_api_accounting.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote results/closed_api_accounting.json")


if __name__ == "__main__":
    sys.exit(main())
