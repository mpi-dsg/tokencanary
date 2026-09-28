# Live check: calibrate on Fireworks, audit gpt-oss-120b traffic from OpenRouter, then the same traffic padded.
# uv run python examples/live_check.py   (reads OPENROUTER_API_KEY and FIREWORKS_API_KEY from .env or the environment)
import copy
import os
import random

import httpx
from openai import OpenAI

import tokencanary
from tokencanary.commission import calibrate

SERVED = "openai/gpt-oss-120b"
SCORER = "accounts/fireworks/models/gpt-oss-120b"
OPENROUTER = "https://openrouter.ai/api/v1"
ROUTE = {"provider": {"order": ["AkashML"], "allow_fallbacks": False}, "reasoning": {"effort": "low"}}
TOPICS = ["photosynthesis", "black holes", "compound interest", "vaccines", "plate tectonics", "the water cycle",
          "neural networks", "the Roman empire", "inflation", "DNA replication", "volcanoes", "the immune system",
          "the Renaissance", "gravity", "antibiotics", "coral reefs", "tides", "nuclear fission", "the Silk Road",
          "rainbows", "jet engines", "sleep", "glaciers", "earthquakes", "the human heart", "solar panels", "chess",
          "lightning", "penicillin", "jazz"]
TEMPLATES = ["Explain {} in two sentences.", "Give me two short facts about {}."]
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 400}


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def pad(response, tok, k, rng):
    """Split up to k content tokens into two tokens with the same bytes, and bill the extra tokens."""
    out = copy.deepcopy(response)
    entries = out["choices"][0]["logprobs"]["content"]
    new, splits = [], 0
    for e in entries:
        b = bytes(e["bytes"])
        cuts = [c for c in range(1, len(b)) if tok.ids_for_bytes(b[:c]) and tok.ids_for_bytes(b[c:])]
        if splits < k and cuts and rng.random() < 0.5:
            c = rng.choice(cuts)
            new += [{"token": b[:c].decode(errors="replace"), "bytes": list(b[:c]), "logprob": -0.1},
                    {"token": b[c:].decode(errors="replace"), "bytes": list(b[c:]), "logprob": -0.1}]
            splits += 1
        else:
            new.append(e)
    out["choices"][0]["logprobs"]["content"] = new
    out["usage"]["completion_tokens"] += splits
    return out


def main():
    load_env()
    rng = random.Random(0)
    prompts = [[{"role": "user", "content": t.format(x)}] for x in TOPICS for t in TEMPLATES]
    rng.shuffle(prompts)
    cal_prompts, test_prompts = prompts[:40], prompts[40:60]

    scorer = tokencanary.EchoScorer(SCORER)
    tok = tokencanary.load_tokenizer("tiktoken:o200k_harmony")
    cal = tokencanary.Calibration()
    kept = calibrate(scorer, SERVED, tok, cal_prompts, cal, SAMPLING["temperature"], SAMPLING["top_p"], SAMPLING["max_tokens"])
    key = f"{SERVED}|latin"
    print(f"calibration: kept {kept}/40, non-canonical {sum(s != 0 for s in cal.scores[key])}, "
          f"tau(0.05) = {cal.threshold(key, 0.05):.2f}, noise margin {cal.margins.get(SERVED, 0):.2f}")

    auditor = tokencanary.Auditor(scorers={SERVED: scorer}, calibration=cal, alpha=0.05, log=None, state=None, background=False)
    client = OpenAI(base_url=OPENROUTER, api_key=os.environ["OPENROUTER_API_KEY"], http_client=tokencanary.http_client(auditor))
    reply = client.chat.completions.create(model=SERVED, messages=test_prompts[0], extra_body=ROUTE, **SAMPLING)
    stream = client.chat.completions.create(model=SERVED, messages=test_prompts[1], extra_body=ROUTE, stream=True, **SAMPLING)
    streamed = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    print("transport: logprobs hidden from caller:", reply.choices[0].logprobs is None, "| streamed chars:", len(streamed))
    for r in auditor.records:
        print(f"  {r.response_id}: billed {r.billed}, returned {r.reported}, count check: {r.count_skipped or 'done'}, "
              f"verdict {r.choices[0].verdict}")

    http = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    raw = []
    for messages in test_prompts[2:]:
        body = {"model": SERVED, "messages": messages, "logprobs": True, **ROUTE, **SAMPLING}
        resp = http.post(f"{OPENROUTER}/chat/completions", json=body).json()
        if resp.get("choices") and resp["choices"][0].get("finish_reason") == "stop":
            raw.append((body, resp))
    print(f"\n{'mode':<10}{'responses':>10}{'non-canonical':>15}{'rejected':>10}")
    for k in (0, 1, 3, 10):
        start = len(auditor.records)
        for body, resp in raw:
            auditor.audit(body, pad(resp, tok, k, rng) if k else resp, "openrouter.ai")
        recs = auditor.records[start:]
        noncanon = sum(bool(c.noncanonical) for r in recs for c in r.choices)
        rejected = sum("likelihood_reject" in r.findings for r in recs)
        print(f"{f'{k} splits':<10}{len(recs):>10}{noncanon:>15}{rejected:>10}")
    print("\n" + auditor.report())


if __name__ == "__main__":
    main()
