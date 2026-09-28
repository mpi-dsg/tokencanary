from __future__ import annotations

import logging

from .auditor import best_reading, reading_ids, sampling, system_hash, text_slice
from .calibration import Calibration, script_bucket
from .scoring import EchoScorer
from .tokenizer import Tokenizer, Unverifiable

log = logging.getLogger("tokencanary")


def calibrate(scorer: EchoScorer, served_model: str, tok: Tokenizer, prompts: list[list[dict]], calibration: Calibration,
              temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 512, language: str | None = None) -> int:
    """Score honest responses from the scoring provider as the auditor would; returns the number kept."""
    kept = 0
    for messages in prompts:
        choice = scorer.generate(messages, temperature, top_p, max_tokens)["choices"][0]
        if choice.get("finish_reason") == "length":
            continue
        text = (choice["message"].get("content") or "").encode()
        entries = (choice.get("logprobs") or {}).get("content") or []
        kind, lo, hi, byte_readings = text_slice(entries, text, tok)
        try:
            if kind is None or byte_readings is None:
                raise Unverifiable("per-token array does not spell the returned text")
            canonical = tok.canonical(text)
            readings = reading_ids(entries[lo:hi], byte_readings, tok)
            if any(len(r) == len(canonical) and all(c in cs for c, cs in zip(canonical, r)) for r in readings):
                score = 0.0
            else:
                score, _ = best_reading(scorer, readings, canonical, tok)
                calibration.raise_margin(served_model, scorer.noise(scorer.resolve(readings[0], tok), canonical, tok))
        except Unverifiable as exc:
            log.info("skipping calibration response: %s", exc)
            continue
        key = Calibration.key(served_model, language or script_bucket(text.decode(errors="replace")))
        request = {"messages": messages, "temperature": temperature, "top_p": top_p}
        t, p = sampling(request)
        calibration.set_meta(key, {"tokenizer": tok.name, "scorer": scorer.fingerprint(), "temperature": t, "top_p": p,
                                   "system": system_hash(request)})
        calibration.add(key, score)
        kept += 1
    return kept


def count_gaps(records: list[dict], model: str, calibration: Calibration, host: str | None = None,
               since: float | None = None, until: float | None = None) -> int:
    """Count gaps from audit records of a trusted period of the endpoint; returns the number added."""
    added = 0
    for r in records:
        if r["model"] != model or (host and r["host"] != host) or not r["shadow"] or len(r["choices"]) != 1:
            continue
        if r["choices"][0]["capped"] or r["status"] == "alert":
            continue
        if (since is not None and r["time"] < since) or (until is not None and r["time"] > until):
            continue
        calibration.add_gap(Calibration.key(model, r["choices"][0]["language"]), r["billed"] - r["canonical"], r["canonical"])
        added += 1
    return added
