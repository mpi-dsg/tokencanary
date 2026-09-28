from __future__ import annotations

import concurrent.futures
import os
import statistics

import httpx

from .tokenizer import Tokenizer, Unverifiable

FIREWORKS = "https://api.fireworks.ai/inference/v1"


def divergent_spans(token_bytes, a: list[int], b: list[int]) -> list[tuple[int, int, int, int]]:
    """Index ranges (a_lo, a_hi, b_lo, b_hi) where two tokenizations of the same bytes differ."""

    def cuts(ids):
        out, pos = {0: 0}, 0
        for i, t in enumerate(ids, 1):
            pos += len(token_bytes(t))
            out[pos] = i
        return out

    ca, cb = cuts(a), cuts(b)
    shared = sorted(set(ca) & set(cb))
    return [
        (ca[lo], ca[hi], cb[lo], cb[hi])
        for lo, hi in zip(shared, shared[1:])
        if ca[hi] - ca[lo] != 1 or cb[hi] - cb[lo] != 1 or a[ca[lo]] != b[cb[lo]]
    ]


class EchoScorer:
    """Scores token ids through a provider that echoes prompt log-probs, in windows around divergent spans."""

    def __init__(self, model: str, base_url: str = FIREWORKS, api_key: str | None = None,
                 api_key_env: str = "FIREWORKS_API_KEY", repeats: int = 3, context: int = 32, suffix: int = 8,
                 workers: int = 8, http: httpx.Client | None = None):
        self.model = model
        self.url = base_url.rstrip("/")
        self.repeats, self.context, self.suffix = repeats, context, suffix
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        key = api_key or os.environ.get(api_key_env)
        self.http = http or httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"} if key else {})

    def fingerprint(self) -> dict:
        """Settings that calibration and audit must share."""
        return {"model": self.model, "base_url": self.url, "context": self.context, "suffix": self.suffix,
                "repeats": self.repeats}

    def prompt_logprobs(self, ids: list[int]) -> list[float | None]:
        r = self.http.post(f"{self.url}/completions", json={
            "model": self.model, "prompt": ids, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 1.0})
        r.raise_for_status()
        lp = r.json()["choices"][0]["logprobs"]
        if lp.get("token_ids", ids)[: len(ids)] != ids:
            raise Unverifiable("scoring provider did not echo the token ids it was sent")
        return lp["token_logprobs"][: len(ids)]

    def windows(self, pairs: list[tuple[list[int], list[int]]], repeats: int | None = None) -> list[float]:
        """log p(body | ctx) for each (ctx, body), median over repeated calls, all calls in parallel."""
        n = repeats or self.repeats
        calls = [self.pool.submit(self.prompt_logprobs, ctx + body) for ctx, body in pairs for _ in range(n)]
        sums = [sum(c.result()[len(pairs[i // n][0]):]) for i, c in enumerate(calls)]
        return [statistics.median(sums[i * n:(i + 1) * n]) for i in range(len(pairs))]

    def _context(self, ids: list[int], lo: int, tok: Tokenizer) -> list[int]:
        return ids[max(0, lo - self.context):lo] or tok.canonical(b"\n")

    def llr(self, reported: list[int], canonical: list[int], tok: Tokenizer) -> tuple[float, int]:
        """Sum over divergent spans of log p(reported window) - log p(canonical window), and the span count."""
        spans = divergent_spans(tok.token_bytes, reported, canonical)
        pairs = []
        for r_lo, r_hi, c_lo, c_hi in spans:
            ctx = self._context(canonical, c_lo, tok)
            tail = canonical[c_hi:c_hi + self.suffix]
            pairs += [(ctx, reported[r_lo:r_hi] + tail), (ctx, canonical[c_lo:c_hi] + tail)]
        scores = self.windows(pairs)
        return sum(scores[0::2]) - sum(scores[1::2]), len(spans)

    def noise(self, reported: list[int], canonical: list[int], tok: Tokenizer) -> float:
        """Largest difference between two independent window scores of the same report."""
        pairs = []
        for r_lo, r_hi, c_lo, c_hi in divergent_spans(tok.token_bytes, reported, canonical):
            body = reported[r_lo:r_hi] + canonical[c_hi:c_hi + self.suffix]
            pairs += [(self._context(canonical, c_lo, tok), body)] * 2
        scores = self.windows(pairs)
        return max([0.0] + [abs(a - b) for a, b in zip(scores[0::2], scores[1::2])])

    def resolve(self, candidates: list[list[int]], tok: Tokenizer, limit: int = 8) -> list[int] | None:
        """At each ambiguous position, the id the model finds most likely."""
        ids = [cs[0] for cs in candidates]
        ambiguous = [i for i, cs in enumerate(candidates) if len(cs) > 1]
        if len(ambiguous) > limit:
            return None
        for i in ambiguous:
            ctx = self._context(ids, i, tok)
            tail = ids[i + 1:i + 1 + self.suffix]
            scores = self.windows([(ctx, [t] + tail) for t in candidates[i]], repeats=1)
            ids[i] = candidates[i][scores.index(max(scores))]
        return ids

    def generate(self, messages: list[dict], temperature: float, top_p: float, max_tokens: int) -> dict:
        """One honest chat completion with per-token log-probs, for calibration."""
        r = self.http.post(f"{self.url}/chat/completions", json={
            "model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p,
            "max_tokens": max_tokens, "logprobs": True})
        r.raise_for_status()
        return r.json()
