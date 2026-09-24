from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import threading
import time
from typing import Callable

from .calibration import Calibration, excess_risk, order_index, script_bucket
from .config import Config, load_config
from .monitor import Monitor
from .tokenizer import Tokenizer, TokenizerRegistry, Unverifiable

log = logging.getLogger("tokencanary")

MAX_AMBIGUOUS = 8  # for scorers without `resolve`: positions resolved by rescoring


@dataclasses.dataclass
class ChoiceResult:
    index: int
    reported: int | None = None  # tokens in the returned per-token array
    canonical: int | None = None  # canonical token count of the text
    aligned: bool | None = None  # array spells the returned text
    noncanonical: bool | None = None
    language: str | None = None  # calibration key: given language or dominant script
    llr: float | None = None
    tau: float | None = None
    verdict: str | None = None  # pass | reject | uncalibrated | unverifiable
    reason: str | None = None


@dataclasses.dataclass
class Record:
    time: float
    host: str
    model: str
    response_id: str | None
    billed: int | None
    max_tokens: int | None
    n: int
    reasoning_tokens: int = 0
    reported: int | None = None
    canonical: int | None = None
    billed_minus_reported: int | None = None
    count_skipped: str | None = None
    findings: list[str] = dataclasses.field(default_factory=list)
    status: str = "ok"  # ok | alert | unverifiable
    choices: list[ChoiceResult] = dataclasses.field(default_factory=list)


_HEX = re.compile(rb"\\x([0-9a-fA-F]{2})")


def entry_bytes(entry: dict) -> bytes:
    if entry.get("bytes") is not None:
        return bytes(entry["bytes"])
    token = entry.get("token", "")
    if token.startswith("bytes:"):
        return _HEX.sub(lambda m: bytes([int(m.group(1), 16)]), token[6:].encode("latin-1", "backslashreplace"))
    return token.encode()


def entry_ids(entry: dict, tok: Tokenizer) -> list[int]:
    """Token ids the entry may stand for; several when distinct ids share the same bytes."""
    token = entry.get("token", "")
    if token.startswith("token_id:"):
        return [int(token[9:])]
    b = entry_bytes(entry)
    if ids := tok.ids_for_bytes(b):
        if len(ids) > 1:  # the encoder's own id first; it is almost always the one generated
            try:
                own = tok.canonical(b)
                ids = sorted(ids, key=lambda t: [t] != own)
            except Unverifiable:
                pass
        return ids
    if (sid := tok.special_id(token)) is not None:
        return [sid]
    raise Unverifiable(f"returned token {token!r} is not in the {tok.name} vocabulary")


class Auditor:
    """Checks Chat Completions responses against their bill. Never raises into the request path.

    config: a Config, a TOML path, or None for the default config file. Keyword overrides replace config fields.
    scorers: model -> reference scorer; enables the likelihood test for that model.
    calibration: Calibration or JSON path; defaults to `config.calibration`.
    """

    def __init__(
        self,
        config: Config | str | None = None,
        scorers: dict | None = None,
        calibration: Calibration | str | None = None,
        background: bool = True,
        on_alert: Callable[[str, dict], None] | None = None,
        **overrides,
    ):
        cfg = config if isinstance(config, Config) else load_config(config)
        self.config = cfg.replace(**overrides) if overrides else cfg
        self.tokenizers = TokenizerRegistry(self.config.tokenizers, self.config.guess_hf)
        self.scorers = dict(scorers or {})
        calibration = calibration or self.config.calibration
        self.calibration = calibration if isinstance(calibration, Calibration) else Calibration(calibration)
        self.monitor = Monitor(self.config.state)
        self.on_alert = on_alert
        self.records: list[Record] = []
        self._lock = threading.Lock()
        self._warned: set[str] = set()
        self._pool = concurrent.futures.ThreadPoolExecutor(1, "tokencanary") if background else None
        self._pending: list[concurrent.futures.Future] = []

    def audit(self, request: dict, response: dict, host: str = "", language: str | None = None) -> Record | None:
        try:
            rec, jobs = self._check(request, response, host, language)
        except Exception:
            log.exception("tokencanary: audit failed")
            return None
        if jobs and self._pool:
            self._pending.append(self._pool.submit(self._finish, rec, request, jobs))
        else:
            self._finish(rec, request, jobs)
        return rec

    def flush(self) -> None:
        pending, self._pending = self._pending, []
        concurrent.futures.wait(pending)

    def report(self) -> str:
        from .report import summarize

        self.flush()
        return summarize([dataclasses.asdict(r) for r in self.records], self.config, self.monitor.state)

    def _check(self, request: dict, response: dict, host: str, language: str | None):
        model = request.get("model") or response.get("model") or "?"
        cfg = self.config.for_model(model)
        usage = response.get("usage") or {}
        choices = response.get("choices") or []
        cap = request.get("max_completion_tokens") or request.get("max_tokens")
        rec = Record(
            time=time.time(), host=host, model=model, response_id=response.get("id"),
            billed=usage.get("completion_tokens"), max_tokens=cap, n=max(len(choices), 1),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
        )
        tok = self.tokenizers.get(model)
        scorer = self.scorers.get(model)

        if cfg.over_cap and rec.billed is not None and cap is not None and rec.billed > cap * rec.n:
            rec.findings.append("over_cap")

        jobs = []
        for ch in choices:
            cr, ids = self._choice(ch, tok, language)
            rec.choices.append(cr)
            if cr.reason and model not in self._warned:
                self._warned.add(model)
                log.warning("tokencanary [%s]: %s; is the tokenizer entry for this model right?", model, cr.reason)
            if ids and scorer:
                jobs.append((cr, *ids))

        crs = rec.choices
        if crs and all(c.reported is not None for c in crs):
            rec.reported = sum(c.reported for c in crs)
        if crs and all(c.canonical is not None for c in crs):
            rec.canonical = sum(c.canonical for c in crs)

        if rec.reasoning_tokens:
            rec.count_skipped = "hidden_reasoning"
        elif rec.reported is None:
            rec.count_skipped = "no_logprobs"
        elif not all(c.aligned for c in crs):
            rec.count_skipped = "unaligned"
        elif rec.billed is None:
            rec.count_skipped = "no_usage"
        elif rec.billed != rec.reported:
            rec.billed_minus_reported = rec.billed - rec.reported
            if cfg.count_mismatch:
                rec.findings.append("count_mismatch")
        return rec, jobs

    def _choice(self, ch: dict, tok: Tokenizer | None, language: str | None):
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        data = text.encode()
        entries = (ch.get("logprobs") or {}).get("content")
        cr = ChoiceResult(index=ch.get("index", 0), language=language or script_bucket(text))
        canonical = None
        if tok is not None:
            try:
                canonical = tok.canonical(data)
                cr.canonical = len(canonical)
            except Unverifiable:
                pass
        if entries is None:
            return cr, None
        cr.reported = len(entries)
        cr.aligned = not msg.get("tool_calls") and b"".join(map(entry_bytes, entries)) == data
        if not cr.aligned or canonical is None:
            cr.verdict = "unverifiable"
            return cr, None
        try:
            candidates = [entry_ids(e, tok) for e in entries]
        except Unverifiable as exc:
            cr.verdict, cr.reason = "unverifiable", str(exc)
            return cr, None
        # Provider's favor: if the canonical split is one possible reading, the report is canonical.
        cr.noncanonical = len(candidates) != len(canonical) or any(c not in cs for c, cs in zip(canonical, candidates))
        return cr, (candidates, canonical)

    def _finish(self, rec: Record, request: dict, jobs) -> None:
        try:
            for cr, candidates, canonical in jobs:
                self._likelihood(rec, request, cr, candidates, canonical)
        except Exception:
            log.exception("tokencanary: likelihood test failed")
        if any(c.verdict == "reject" for c in rec.choices):
            rec.findings.append("likelihood_reject")
        if "over_cap" in rec.findings or "count_mismatch" in rec.findings:
            rec.status = "alert"
        elif rec.count_skipped and not any(c.verdict in ("pass", "reject") for c in rec.choices):
            rec.status = "unverifiable"
        with self._lock:
            self.records.append(rec)
            if self.config.log:
                os.makedirs(os.path.dirname(os.path.abspath(self.config.log)), exist_ok=True)
                with open(self.config.log, "a") as f:
                    f.write(json.dumps(dataclasses.asdict(rec), ensure_ascii=False) + "\n")
        if rec.status == "alert":
            self._alert_response(rec)
        self._monitor(rec)

    def _likelihood(self, rec: Record, request: dict, cr: ChoiceResult, candidates: list[list[int]], canonical: list[int]) -> None:
        """Reject when log p(reported) - log p(canonical) < tau; canonical reports score 0 without scoring."""
        cfg = self.config.for_model(rec.model)
        key = Calibration.key(rec.model, cr.language)
        cr.tau = self.calibration.threshold(key, cfg.alpha)
        if cr.tau is None:
            cr.verdict = "uncalibrated"
            return
        self._check_calibration(key, cfg)
        if not cr.noncanonical:
            cr.llr = 0.0
        else:
            scorer = self.scorers[rec.model]
            reported = self._resolve(scorer, request, candidates)
            if reported is None:
                cr.verdict, cr.reason = "unverifiable", "too many ambiguous token ids"
                return
            cr.llr = scorer.logprob(request, reported) - scorer.logprob(request, canonical)
        cr.verdict = "reject" if cr.llr < cr.tau else "pass"

    @staticmethod
    def _resolve(scorer, request: dict, candidates: list[list[int]]) -> list[int] | None:
        """Provider's favor: at each ambiguous position, the id the reference model finds most likely."""
        if hasattr(scorer, "resolve"):
            return scorer.resolve(request, candidates)
        ambiguous = [i for i, cs in enumerate(candidates) if len(cs) > 1]
        if len(ambiguous) > MAX_AMBIGUOUS:
            return None
        ids = [cs[0] for cs in candidates]
        for i in ambiguous:
            ids[i] = max(candidates[i], key=lambda t: scorer.logprob(request, ids[:i] + [t] + ids[i + 1:]))
        return ids

    def _check_calibration(self, key: str, cfg: Config) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        n = len(self.calibration.scores[key])
        if order_index(n, cfg.alpha) == 0:
            log.warning("tokencanary [%s]: %d calibration responses cannot reach alpha=%g; the test never rejects", key, n, cfg.alpha)
        elif (risk := excess_risk(n, cfg.alpha, cfg.tolerance * cfg.alpha)) > 0.01:
            log.warning("tokencanary [%s]: with %d calibration responses, P(honest rejection rate > %g) = %.3f; add calibration data",
                        key, n, cfg.tolerance * cfg.alpha, risk)

    def _monitor(self, rec: Record) -> None:
        cfg = self.config.for_model(rec.model)
        for c in rec.choices:
            if c.verdict not in ("pass", "reject"):
                continue
            rejected = c.verdict == "reject"
            extra = max(0, c.reported - c.canonical) if rejected else 0
            st = self.monitor.update(f"{rec.host}|{rec.model}", min(cfg.tolerance * cfg.alpha, 0.5), cfg.confidence,
                                     cfg.min_overcharge, rejected, extra, c.reported)
            if st["newly_flagged"]:
                msg = (f"tokencanary [{rec.host} {rec.model}]: likelihood rejections {st['rejected']}/{st['tested']} exceed "
                       f"{st['p0']:.1%} with evidence e^{st['log_evidence']:.1f}; if all were padding, "
                       f"overcharge <= {st['extra'] / st['tokens']:.2%} of tokens")
                self._alert("provider", msg, st)

    def _alert_response(self, rec: Record) -> None:
        parts = []
        if "over_cap" in rec.findings:
            parts.append(f"billed {rec.billed} tokens, above max_tokens={rec.max_tokens} x {rec.n}")
        if "count_mismatch" in rec.findings:
            parts.append(f"billed {rec.billed} tokens but returned {rec.reported} ({rec.billed_minus_reported:+d})")
        self._alert("response", f"tokencanary [{rec.host} {rec.model} {rec.response_id}]: " + "; ".join(parts), dataclasses.asdict(rec))

    def _alert(self, kind: str, message: str, detail: dict) -> None:
        log.warning(message)
        if self.on_alert:
            try:
                self.on_alert(kind, detail)
            except Exception:
                log.exception("tokencanary: on_alert failed")
