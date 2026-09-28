from __future__ import annotations

import random
from statistics import fmean


def request_reject(gap: int, calibration: list[list[int]]) -> bool:
    """Per-request rule: billed - canonical above every honest calibration gap."""
    return gap > max(g for g, _ in calibration)


def request_level(calibration: list[list[int]]) -> float:
    """False-alarm probability of the per-request rule under exchangeability."""
    return 1 / (len(calibration) + 1)


def batch_pvalue(batch: list[list[int]], calibration: list[list[int]], margin: float,
                 permutations: int = 2000, seed: int = 0) -> float:
    """One-sided permutation test that the batch's gaps, minus `margin` of each canonical count,
    exceed the calibration gaps on average."""
    adjusted = [g - margin * c for g, c in batch]
    reference = [float(g) for g, _ in calibration]
    observed = fmean(adjusted) - fmean(reference)
    pooled, n = adjusted + reference, len(adjusted)
    rng = random.Random(seed)
    hits = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        hits += fmean(pooled[:n]) - fmean(pooled[n:]) >= observed
    return (hits + 1) / (permutations + 1)
