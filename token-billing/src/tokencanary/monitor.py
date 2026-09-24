from __future__ import annotations

import json
import math
import os
import threading

# Bet sizes as fractions of the largest safe bet 1/p0.
GRID = [0.5 * 2.0**-j for j in range(8)]


class Monitor:
    """Anytime-valid test per endpoint that the likelihood-rejection rate exceeds p0.

    Each tested response multiplies the evidence for bet b by 1 + (b / p0) (x - p0), x = 1 if rejected.
    If the true rejection rate is at most p0, the averaged evidence is a nonnegative supermartingale, so by
    Ville's inequality it ever reaches 1 / delta with probability at most delta, however long we monitor.
    State persists across restarts; restarting from zero would give an honest provider fresh chances.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.state: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path) as f:
                self.state = json.load(f)

    def update(self, key: str, p0: float, delta: float, min_overcharge: float, rejected: bool, extra: int, tokens: int) -> dict:
        """Add one tested response. Sets `flagged` once evidence >= 1/delta and worst-case overcharge >= min_overcharge."""
        with self._lock:
            st = self.state.get(key)
            if st is None or st["p0"] != p0:
                st = self.state[key] = {"p0": p0, "log_m": [0.0] * len(GRID), "tested": 0, "rejected": 0,
                                        "extra": 0, "tokens": 0, "log_evidence": 0.0, "flagged": False}
            x = 1.0 if rejected else 0.0
            st["log_m"] = [lm + math.log1p(b / p0 * (x - p0)) for lm, b in zip(st["log_m"], GRID)]
            st["tested"] += 1
            st["rejected"] += int(rejected)
            st["extra"] += extra
            st["tokens"] += tokens
            top = max(st["log_m"])
            st["log_evidence"] = top + math.log(sum(math.exp(lm - top) for lm in st["log_m"]) / len(GRID))
            overcharge = st["extra"] / max(st["tokens"], 1)
            crossed = st["log_evidence"] >= -math.log(delta)
            st["newly_flagged"] = not st["flagged"] and crossed and overcharge >= min_overcharge
            st["flagged"] |= st["newly_flagged"]
            if self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "w") as f:
                    json.dump(self.state, f)
            return dict(st)
