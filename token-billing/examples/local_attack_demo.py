import argparse
import json
import random
import time

import httpx
import openai

from tokencanary import Auditor, AuditTransport, Calibration, HFTokenizer
from tokencanary.scoring import LocalHFScorer, commission

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SERVED = "qwen-0.5b"
TOPICS = [
    "photosynthesis", "black holes", "the French revolution", "compound interest", "vaccines", "plate tectonics",
    "the water cycle", "neural networks", "the Roman empire", "inflation", "DNA replication", "climate change",
    "supply and demand", "the printing press", "quantum entanglement", "volcanoes", "the immune system", "blockchains",
    "the Renaissance", "gravity", "antibiotics", "the internet", "coral reefs", "the stock market", "electric cars",
    "the moon landing", "honeybees", "tides", "nuclear fission", "the Silk Road", "democracy", "rainbows",
    "jet engines", "sleep", "the periodic table", "glaciers", "coffee", "earthquakes", "the human heart",
    "solar panels", "chess", "the Great Wall of China", "recycling", "lightning", "penicillin", "the Olympic games",
    "rain forests", "public key cryptography", "the Nile river", "jazz",
]
TEMPLATES = ["Explain {} in two sentences.", "Give me three short facts about {}."]


def pad(ids, tok, k, rng):
    """Split up to k random tokens into two vocabulary tokens with the same bytes."""
    out = [[t] for t in ids]
    order = list(range(len(ids)))
    rng.shuffle(order)
    for i in order:
        if k == 0:
            break
        b = tok.token_bytes(ids[i])
        cuts = [c for c in range(1, len(b)) if tok.ids_for_bytes(b[:c]) and tok.ids_for_bytes(b[c:])]
        if cuts:
            c = rng.choice(cuts)
            out[i] = [tok.ids_for_bytes(b[:c])[0], tok.ids_for_bytes(b[c:])[0]]
            k -= 1
    return [t for group in out for t in group]


class Provider:
    """Mock endpoint serving local generations; with attack_k > 0 it pads the bill and fakes a matching log-prob array."""

    def __init__(self, scorer, tok):
        self.scorer, self.tok = scorer, tok
        self.attack_k = 0
        self.rng = random.Random(0)
        self.honest = self.billed = 0

    def __call__(self, request):
        body = json.loads(request.content)
        _, gen = self.scorer.sample(body["messages"], body["temperature"], body["top_p"], body["max_tokens"])
        ids = pad(gen, self.tok, self.attack_k, self.rng)
        self.honest += len(gen)
        self.billed += len(ids)
        entries = [{"token": "", "logprob": 0.0, "bytes": list(self.tok.token_bytes(t))} for t in ids]
        text = b"".join(map(self.tok.token_bytes, ids)).decode(errors="replace")
        return httpx.Response(200, json={
            "id": f"demo-{time.time_ns()}", "model": SERVED,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "logprobs": {"content": entries}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(ids), "total_tokens": len(ids)},
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-prompts", type=int, default=60)
    ap.add_argument("--test-prompts", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    sampling = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 128}

    prompts = [[{"role": "user", "content": t.format(x)}] for x in TOPICS for t in TEMPLATES]
    random.Random(0).shuffle(prompts)
    cal_prompts = prompts[: args.calibration_prompts]
    test_prompts = prompts[args.calibration_prompts : args.calibration_prompts + args.test_prompts]

    scorer = LocalHFScorer(MODEL_ID)
    tok = HFTokenizer.from_pretrained(MODEL_ID)
    cal = Calibration()
    commission(scorer, SERVED, cal_prompts, cal, sampling["temperature"], sampling["top_p"], sampling["max_tokens"])
    for key, scores in cal.scores.items():
        print(f"calibrated {key}: n={len(scores)}, non-canonical={sum(s != 0 for s in scores)}, tau={cal.threshold(key, args.alpha):.2f}")

    provider = Provider(scorer, tok)
    auditor = Auditor(tokenizers={SERVED: tok}, scorers={SERVED: scorer}, calibration=cal, alpha=args.alpha,
                      log=None, state=None, background=False)
    client = openai.OpenAI(api_key="demo", base_url="https://provider.invalid/v1",
                           http_client=httpx.Client(transport=AuditTransport(auditor, httpx.MockTransport(provider))))

    print(f"\n{'mode':<10}{'padded':>8}{'count alerts':>14}{'likelihood rejects':>20}")
    for k in (0, 1, 3, 10):
        provider.attack_k, provider.honest, provider.billed = k, 0, 0
        start = len(auditor.records)
        for messages in test_prompts:
            client.chat.completions.create(model=SERVED, messages=messages, **sampling)
        recs = auditor.records[start:]
        counts = sum("count_mismatch" in r.findings for r in recs)
        rejects = sum("likelihood_reject" in r.findings for r in recs)
        padded = 100 * (provider.billed - provider.honest) / provider.honest
        print(f"{f'{k} splits':<10}{padded:>7.1f}%{counts:>14}{rejects:>15} / {len(recs)}")

    print("\n" + auditor.report())


if __name__ == "__main__":
    main()
