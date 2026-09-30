from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import logging
import os
import re
import threading
import time
from statistics import fmean
from typing import Callable
from urllib.parse import urlparse

from . import counts
from .calibration import Calibration, excess_risk, order_index, script_bucket
from .config import Config, best_match, load_config
from .monitor import Monitor
from .scoring import EchoScorer
from .tokenizer import TokenMismatch, Tokenizer, TokenizerRegistry, Unverifiable

log = logging.getLogger("tokencanary")

PLACEHOLDER = "�"  # how providers render a token that holds part of a multi-byte character
MAX_READINGS = 16  # alignments of placeholder tokens tried before a response counts as unverifiable


@dataclasses.dataclass
class ChoiceResult:
    index: int
    reported: int | None = None  # entries in the returned per-token array
    answer: int | None = None  # entries that spell the returned text
    canonical: int | None = None  # canonical token count of the text
    array: str | None = None  # "content": array spells the text; "full": text is its final part; None: unaligned
    reasoning: bool = False  # message carries reasoning text
    capped: bool = False  # generation stopped at the output limit
    noncanonical: bool | None = None
    language: str | None = None  # calibration key: given language or dominant script
    llr: float | None = None
    tau: float | None = None
    verdict: str | None = None  # pass | reject | uncalibrated | unverifiable | capped
    reason: str | None = None
    token_mismatch: bool = False  # returned tokens do not fit the configured tokenizer


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
    shadow: bool = False  # billed and canonical counts cover the same text
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


def entry_pattern(entry: dict) -> list[bytes | None]:
    """Byte pattern of an entry. None stands for one to three bytes of a partial character."""
    token = entry.get("token", "")
    if entry.get("bytes") is not None or token.startswith("bytes:") or PLACEHOLDER not in token:
        return [entry_bytes(entry)]
    out: list[bytes | None] = []
    for i, part in enumerate(token.split(PLACEHOLDER)):
        if i:
            out.append(None)
        if part:
            out.append(part.encode())
    return out


def _ends(pattern: list[bytes | None], data: bytes, pos: int) -> set[int]:
    ends = {pos}
    for seg in pattern:
        nxt = set()
        for p in ends:
            if seg is None:
                nxt |= {p + k for k in (1, 2, 3) if p + k <= len(data) and all(b >= 0x80 for b in data[p:p + k])}
            elif data.startswith(seg, p):
                nxt.add(p + len(seg))
        ends = nxt
    return ends


def align(entries: list[dict], data: bytes, limit: int = MAX_READINGS) -> list[list[bytes]] | None:
    """All ways the entries can spell `data`, as per-entry byte strings; None if more than `limit`."""
    back: list[dict[int, set[int]]] = []
    cur = {0}
    for entry in entries:
        nxt: dict[int, set[int]] = {}
        pattern = entry_pattern(entry)
        for pos in cur:
            for end in _ends(pattern, data, pos):
                nxt.setdefault(end, set()).add(pos)
        if not nxt:
            return []
        back.append(nxt)
        cur = set(nxt)
    if len(data) not in cur:
        return []
    paths = [[len(data)]]
    for step in reversed(back):
        paths = [[start, *p] for p in paths for start in sorted(step[p[0]])]
        if len(paths) > limit:
            return None
    return [[data[p[i]:p[i + 1]] for i in range(len(entries))] for p in paths]


def entry_ids(entry: dict, tok: Tokenizer, b: bytes | None = None) -> list[int]:
    """Token ids the entry may stand for, given its bytes `b`; several when distinct ids share the same bytes."""
    b = entry_bytes(entry) if b is None else b
    tid = entry.get("token_id")
    if tid is None and entry.get("token", "").startswith("token_id:"):
        tid = int(entry["token"][9:])
    if tid is not None:
        if tok.token_bytes(tid) != b:
            raise TokenMismatch(f"token id {tid} does not decode to the entry's bytes {b!r}")
        return [tid]
    if ids := tok.ids_for_bytes(b):
        if len(ids) > 1:  # the encoder's own id first; it is almost always the one generated
            try:
                own = tok.canonical(b)
                ids = sorted(ids, key=lambda t: [t] != own)
            except Unverifiable:
                pass
        return ids
    if (sid := tok.special_id(entry.get("token", ""))) is not None:
        return [sid]
    raise TokenMismatch(f"returned token {entry.get('token')!r} is not in the {tok.name} vocabulary")


def reading_ids(entries: list[dict], byte_readings: list[list[bytes]], tok: Tokenizer) -> list[list[list[int]]]:
    """Candidate ids per entry for each byte reading; readings that map to no tokens are dropped."""
    readings, error = [], None
    for reading in byte_readings:
        try:
            readings.append([entry_ids(e, tok, b) for e, b in zip(entries, reading)])
        except Unverifiable as exc:
            error = exc
    if not readings:
        raise error or Unverifiable("per-token array does not spell the returned text")
    return readings


def _special(entry: dict, tok: Tokenizer | None) -> bool:
    if entry.get("token_id") is not None and tok is not None:
        return tok.is_special(entry["token_id"])
    return tok is not None and tok.special_id(entry.get("token", "")) is not None


def _length_range(entry: dict) -> tuple[int, int]:
    pattern = entry_pattern(entry)
    fixed = sum(len(s) for s in pattern if s is not None)
    holes = sum(s is None for s in pattern)
    return fixed + holes, fixed + 3 * holes


def text_slice(entries: list[dict], data: bytes, tok: Tokenizer | None) -> tuple[str | None, int, int, list | None]:
    """(kind, lo, hi, readings): "content" if the entries spell the text, "full" if it is their final part."""
    readings = align(entries, data)
    if readings is None or readings:
        return "content", 0, len(entries), readings
    end = len(entries)
    while end and _special(entries[end - 1], tok):
        end -= 1
    low = high = 0
    for start in range(end - 1, -1, -1):
        lo_len, hi_len = _length_range(entries[start])
        low, high = low + lo_len, high + hi_len
        if low > len(data):
            break
        if data and high >= len(data):
            readings = align(entries[start:end], data)
            if readings is None or readings:
                return "full", start, end, readings
    return None, 0, 0, None


def system_hash(request: dict) -> str | None:
    text = "\n".join(str(m.get("content")) for m in request.get("messages") or [] if m.get("role") in ("system", "developer"))
    return hashlib.sha256(text.encode()).hexdigest()[:16] if text else None


def sampling(request: dict) -> tuple[float, float]:
    t, p = request.get("temperature"), request.get("top_p")
    return (1.0 if t is None else float(t), 1.0 if p is None else float(p))


def best_reading(scorer, readings: list[list[list[int]]], canonical: list[int], tok: Tokenizer) -> tuple[float, int]:
    """The highest likelihood ratio over readings, with its divergent span count."""
    best = None
    for candidates in readings:
        reported = scorer.resolve(candidates, tok)
        if reported is None:
            raise Unverifiable("too many ambiguous token ids")
        result = scorer.llr(reported, canonical, tok)
        if best is None or result[0] > best[0]:
            best = result
    return best


class Auditor:
    """Checks Chat Completions responses against their bill. Never raises into the request path.

    config: a Config, a TOML path, or None for the default config file. Keyword overrides replace config fields.
    scorers: model -> scorer, overriding `config.scorers`; enables the likelihood test for that model.
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
        self._batches: dict[str, list[list[int]]] = {}
        self._lock = threading.Lock()
        self._warned: set[str] = set()
        self._pool = concurrent.futures.ThreadPoolExecutor(1, "tokencanary") if background else None
        self._pending: list[concurrent.futures.Future] = []

    def scorer(self, model: str):
        if model not in self.scorers:
            spec = best_match(model, self.config.scorers)
            self.scorers[model] = EchoScorer(**spec) if spec else None
        return self.scorers[model]

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

    def _warn_once(self, key: str, message: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message, *args)

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
        scorer = self.scorer(model)
        if scorer is not None and host and urlparse(getattr(scorer, "url", "")).hostname == host:
            self._warn_once(f"independence|{model}", "tokencanary [%s]: the scoring provider is the audited provider, "
                            "so likelihood scores are not independent of the bill", model)

        jobs = []
        for ch in choices:
            cr, job = self._choice(ch, tok, language)
            rec.choices.append(cr)
            if cr.token_mismatch:
                self._warn_once(f"tokenizer|{model}", "tokencanary [%s]: %s; is the tokenizer entry for this model right?",
                                model, cr.reason)
            if job and scorer is not None:
                jobs.append((cr, *job))

        crs = rec.choices
        if cfg.over_cap and cap is not None and (
            (rec.billed is not None and rec.billed > cap * rec.n) or any((c.reported or 0) > cap for c in crs)
        ):
            rec.findings.append("over_cap")
        if crs and all(c.reported is not None for c in crs):
            rec.reported = sum(c.reported for c in crs)
        if crs and all(c.canonical is not None for c in crs):
            rec.canonical = sum(c.canonical for c in crs)
        channels = tok is not None and tok.special_id("<|channel|>") is not None
        hidden = rec.reasoning_tokens or channels or any(c.reasoning for c in crs)  # billed but not in the text
        rec.shadow = rec.billed is not None and rec.canonical is not None and not hidden

        if rec.reported is None:
            rec.count_skipped = "no_logprobs"
        elif not all(c.array for c in crs):
            rec.count_skipped = "unaligned"
        elif hidden and any(c.array == "content" for c in crs):
            rec.count_skipped = "billed_tokens_outside_logprobs"
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
        cr = ChoiceResult(index=ch.get("index", 0), language=language or script_bucket(text),
                          reasoning=bool(msg.get("reasoning") or msg.get("reasoning_content")),
                          capped=ch.get("finish_reason") == "length")
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
        if msg.get("tool_calls"):
            cr.verdict = "unverifiable"
            return cr, None
        cr.array, lo, hi, byte_readings = text_slice(entries, data, tok)
        if cr.array is None:
            cr.verdict = "unverifiable"
            return cr, None
        cr.answer = hi - lo
        if byte_readings is None:
            cr.verdict, cr.reason = "unverifiable", "too many alignments of partial-character tokens"
            return cr, None
        if canonical is None:
            cr.verdict = "unverifiable"
            return cr, None
        if cr.capped:
            cr.verdict = "capped"
            return cr, None
        try:
            readings = reading_ids(entries[lo:hi], byte_readings, tok)
        except Unverifiable as exc:
            cr.verdict, cr.reason, cr.token_mismatch = "unverifiable", str(exc), isinstance(exc, TokenMismatch)
            return cr, None
        # Provider's favor: if the canonical split is one possible reading, the report is canonical.
        cr.noncanonical = not any(
            len(r) == len(canonical) and all(c in cs for c, cs in zip(canonical, r)) for r in readings)
        return cr, (readings, canonical, tok)

    def _finish(self, rec: Record, request: dict, jobs) -> None:
        try:
            for cr, readings, canonical, tok in jobs:
                self._likelihood(rec, request, cr, readings, canonical, tok)
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
        self._monitor_likelihood(rec)
        self._count_rules(rec)

    def _likelihood(self, rec: Record, request: dict, cr: ChoiceResult, readings, canonical, tok: Tokenizer) -> None:
        """Reject when the window LLR falls below min(tau, 0) minus the noise margin per divergent span."""
        cfg = self.config.for_model(rec.model)
        key = Calibration.key(rec.model, cr.language)
        cr.tau = self.calibration.threshold(key, cfg.alpha)
        if cr.tau is None:
            cr.verdict = "uncalibrated"
            return
        scorer = self.scorer(rec.model)
        if mismatch := self._mismatch(key, request, tok, scorer):
            cr.verdict, cr.reason = "uncalibrated", mismatch
            return
        self._check_calibration(key, cfg)
        spans = 0
        if not cr.noncanonical:
            cr.llr = 0.0
        else:
            try:
                cr.llr, spans = best_reading(scorer, readings, canonical, tok)
            except Unverifiable as exc:
                cr.verdict, cr.reason = "unverifiable", str(exc)
                return
        margin = self.calibration.margins.get(rec.model, 0.0)
        cut = min(cr.tau, 0.0) - spans * margin
        cr.verdict = "reject" if cr.llr < cut else "pass"

    def _mismatch(self, key: str, request: dict, tok: Tokenizer, scorer) -> str | None:
        """Why the calibration does not apply to this request, or None."""
        meta = self.calibration.meta.get(key)
        if meta is None:
            self._warn_once(f"meta|{key}", "tokencanary [%s]: calibration records no settings; cannot check they match", key)
            return None
        fingerprint = scorer.fingerprint() if hasattr(scorer, "fingerprint") else None
        if meta.get("tokenizer") != tok.name:
            reason = f"calibrated with tokenizer {meta.get('tokenizer')}, auditing with {tok.name}"
        elif fingerprint is not None and meta.get("scorer") != fingerprint:
            reason = f"calibrated with scorer {meta.get('scorer')}, auditing with {fingerprint}"
        elif sampling(request) != (meta.get("temperature"), meta.get("top_p")):
            t, p = sampling(request)
            reason = f"request samples at temperature {t}, top_p {p}; calibration at {meta.get('temperature')}, {meta.get('top_p')}"
        else:
            if system_hash(request) != meta.get("system"):
                self._warn_once(f"system|{key}", "tokencanary [%s]: system prompt differs from calibration", key)
            return None
        self._warn_once(f"mismatch|{key}|{reason}", "tokencanary [%s]: not tested, %s", key, reason)
        return reason

    def _check_calibration(self, key: str, cfg: Config) -> None:
        n = len(self.calibration.scores[key])
        if order_index(n, cfg.alpha) == 0:
            self._warn_once(f"size|{key}", "tokencanary [%s]: %d calibration responses cannot reach alpha=%g; the test never rejects",
                            key, n, cfg.alpha)
        elif (risk := excess_risk(n, cfg.alpha, cfg.tolerance * cfg.alpha)) > 0.01:
            self._warn_once(f"size|{key}", "tokencanary [%s]: with %d calibration responses, P(honest rejection rate > %g) = %.3f; "
                            "add calibration data", key, n, cfg.tolerance * cfg.alpha, risk)

    def _update(self, kind: str, key: str, level: float, rejected: bool, extra: int, tokens: int, cfg: Config) -> None:
        st = self.monitor.update(key, min(cfg.tolerance * level, 0.5), cfg.confidence, cfg.min_overcharge, rejected, extra, tokens)
        if st["newly_flagged"]:
            msg = (f"tokencanary [{key}]: {kind} rejections {st['rejected']}/{st['tested']} exceed {st['p0']:.1%} "
                   f"with evidence e^{st['log_evidence']:.1f}; if all were padding, overcharge <= "
                   f"{st['extra'] / st['tokens']:.2%} of tokens")
            self._alert(kind, msg, st)

    def _monitor_likelihood(self, rec: Record) -> None:
        cfg = self.config.for_model(rec.model)
        for c in rec.choices:
            if c.verdict in ("pass", "reject"):
                rejected = c.verdict == "reject"
                extra = max(0, c.answer - c.canonical) if rejected else 0
                self._update("likelihood", f"{rec.host}|{rec.model}", cfg.alpha, rejected, extra, c.answer, cfg)

    def _count_rules(self, rec: Record) -> None:
        """Count-only rules for responses without log-probs."""
        cfg = self.config.for_model(rec.model)
        if not (cfg.count_rules and rec.count_skipped == "no_logprobs" and rec.shadow and len(rec.choices) == 1
                and not rec.choices[0].capped):
            return
        key = Calibration.key(rec.model, rec.choices[0].language)
        calibration = self.calibration.gaps.get(key)
        if not calibration:
            return
        gap = rec.billed - rec.canonical
        rejected = counts.request_reject(gap, calibration)
        extra = max(0, gap - max(g for g, _ in calibration)) if rejected else 0
        self._update("count", f"request|{rec.host}|{key}", counts.request_level(calibration), rejected, extra, rec.billed, cfg)
        batch = self._batches.setdefault(key, [])
        batch.append([gap, rec.canonical])
        if len(batch) == cfg.batch_size:
            self._batches[key] = []
            rejected = counts.batch_pvalue(batch, calibration, cfg.batch_margin) < cfg.batch_alpha
            excess = sum(g for g, _ in batch) - len(batch) * fmean(g for g, _ in calibration)
            tokens = sum(g + c for g, c in batch)
            self._update("count", f"batch|{rec.host}|{key}", cfg.batch_alpha, rejected,
                         max(0, round(excess)) if rejected else 0, tokens, cfg)

    def _alert_response(self, rec: Record) -> None:
        parts = []
        if "over_cap" in rec.findings:
            parts.append(f"billed {rec.billed}, reported {rec.reported} tokens; max_tokens={rec.max_tokens} x {rec.n}")
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
