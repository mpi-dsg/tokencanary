from __future__ import annotations

import json
import math
import os
import threading

_SCRIPTS = {
    "cjk": ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF)),
    "devanagari": ((0x0900, 0x097F),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "cyrillic": ((0x0400, 0x04FF),),
}


def script_bucket(text: str) -> str:
    """Dominant script of the text; the calibration key when no language is given."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            cp = ord(ch)
            name = next((s for s, rs in _SCRIPTS.items() if any(lo <= cp <= hi for lo, hi in rs)), "latin")
            counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), summed directly so tiny tails stay accurate."""
    if k <= 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def order_index(n: int, alpha: float) -> int:
    return math.floor(alpha * (n + 1) + 1e-9)


def conformal_threshold(scores: list[float], alpha: float) -> float:
    """k-th smallest honest score, k = floor(alpha (n+1)); -inf if k = 0."""
    k = order_index(len(scores), alpha)
    return sorted(scores)[k - 1] if k else -math.inf


def excess_risk(n: int, alpha: float, rate: float) -> float:
    """Probability over calibration draws that the threshold's true honest rejection rate exceeds `rate`.

    That rate is Beta(k, n+1-k) distributed, and P(Beta(k, n+1-k) > x) = P(Binomial(n, x) <= k-1).
    """
    k = order_index(n, alpha)
    return binom_cdf(k - 1, n, rate) if k else 0.0


class Calibration:
    """Honest likelihood scores per `model|language` key, stored as JSON."""

    def __init__(self, path: str | None = None):
        self.path = os.path.expanduser(path) if path else None
        self.scores: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        if self.path and os.path.exists(self.path):
            with open(self.path) as f:
                self.scores = json.load(f)["scores"]

    @staticmethod
    def key(model: str, language: str) -> str:
        return f"{model}|{language}"

    def threshold(self, key: str, alpha: float) -> float | None:
        scores = self.scores.get(key)
        return conformal_threshold(scores, alpha) if scores else None

    def add(self, key: str, score: float) -> None:
        with self._lock:
            self.scores.setdefault(key, []).append(score)

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("Calibration.save needs a path")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._lock, open(path, "w") as f:
            json.dump({"scores": self.scores}, f)
