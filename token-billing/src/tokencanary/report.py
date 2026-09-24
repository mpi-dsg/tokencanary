from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

from .calibration import binom_sf
from .config import Config


def load_log(path: str | None) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_state(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def summarize(records: list[dict], config: Config, state: dict) -> str:
    groups = defaultdict(list)
    for r in records:
        groups[r["host"], r["model"]].append(r)
    if not groups:
        return "tokencanary: no audited requests."
    lines = []
    for (host, model), recs in sorted(groups.items()):
        cfg = config.for_model(model)
        found = Counter(f for r in recs for f in r["findings"])
        skipped = Counter(r["count_skipped"] for r in recs if r["count_skipped"])
        offsets = Counter(r["billed_minus_reported"] for r in recs if r["billed_minus_reported"] is not None)
        lines.append(f"== {model} @ {host}: {len(recs)} requests")
        lines.append(f"  billed > max_tokens: {found['over_cap']}")
        lines.append(f"  billed != returned tokens: {len(offsets)} of {len(recs) - sum(skipped.values())} checked"
                     + (f", offsets {dict(offsets)}" if offsets else "") + (f"; skipped {dict(skipped)}" if skipped else ""))

        shadow = [r for r in recs if r["billed"] is not None and r["canonical"] is not None and not r["reasoning_tokens"]]
        if shadow:
            b, c = sum(r["billed"] for r in shadow), sum(r["canonical"] for r in shadow)
            lines.append(f"  shadow bill: billed {b}, canonical {c} ({b - c:+d}, {100 * (b - c) / max(b, 1):+.2f}%)")

        choices = [c for r in recs for c in r["choices"]]
        tested = [c for c in choices if c["verdict"] in ("pass", "reject")]
        rejected = [c for c in tested if c["verdict"] == "reject"]
        uncalibrated = sum(c["verdict"] == "uncalibrated" for c in choices)
        if tested or uncalibrated:
            n, k = len(tested), len(rejected)
            p = binom_sf(k, n, cfg.alpha)
            line = f"  likelihood test: {k} of {n} rejected (expected {cfg.alpha * n:.1f} at alpha={cfg.alpha}; P(>= {k} | honest) = {p:.2g})"
            if uncalibrated:
                line += f"; {uncalibrated} uncalibrated"
            lines.append(line)
            if tested:
                extra = sum(max(0, c["reported"] - c["canonical"]) for c in rejected)
                lines.append(f"    if every rejection were padding: overcharge <= {extra / sum(c['reported'] for c in tested):.3%} of tokens")
        if st := state.get(f"{host}|{model}"):
            verdict = "FLAGGED" if st["flagged"] else "not flagged"
            lines.append(f"  provider test (rate > {st['p0']:.1%}): evidence e^{st['log_evidence']:.1f}, "
                         f"alert at e^{-math.log(cfg.confidence):.1f}; {verdict}")
    return "\n".join(lines)
