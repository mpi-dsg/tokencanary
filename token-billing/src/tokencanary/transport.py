from __future__ import annotations

import dataclasses
import json
import logging

import httpx

from .auditor import Auditor

log = logging.getLogger("tokencanary")

_HOP = {"content-length", "content-encoding", "transfer-encoding", "connection"}
LANGUAGE_HEADER = "x-tokencanary-language"  # optional per-request calibration language; not forwarded


@dataclasses.dataclass
class _Ctx:
    body: dict
    host: str
    stream: bool
    language: str | None = None
    added_logprobs: bool = False
    added_usage: bool = False


class _Core:
    """Shared by the sync and async transports: request rewriting and response auditing."""

    def __init__(self, auditor: Auditor, inject: bool):
        self.auditor = auditor
        self.inject = inject
        self.no_inject: set[str] = set()

    def prepare(self, request: httpx.Request, content: bytes, inject: bool = True):
        """Add `logprobs` (and usage for streams) to chat completion requests that lack them."""
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            return request, None
        try:
            body = json.loads(content)
        except ValueError:
            return request, None
        if not isinstance(body, dict):
            return request, None
        ctx = _Ctx(body, request.url.host, bool(body.get("stream")), request.headers.get(LANGUAGE_HEADER))
        new = dict(body)
        if inject and self.inject and body.get("model") not in self.no_inject:
            if not body.get("logprobs"):
                new["logprobs"] = ctx.added_logprobs = True
            opts = body.get("stream_options") or {}
            if ctx.stream and not opts.get("include_usage"):
                new["stream_options"] = {**opts, "include_usage": True}
                ctx.added_usage = True
        if new == body and ctx.language is None:
            return request, ctx
        drop = {b"content-length", LANGUAGE_HEADER.encode()}
        headers = [(k, v) for k, v in request.headers.raw if k.lower() not in drop]
        return httpx.Request(request.method, request.url, headers=headers, content=json.dumps(new).encode(), extensions=request.extensions), ctx

    def injection_rejected(self, ctx: _Ctx, resp: httpx.Response, body: bytes) -> bool:
        """A 400 mentioning our additions: remember the model and resend the original request."""
        if resp.status_code != 400 or not (ctx.added_logprobs or ctx.added_usage):
            return False
        if not any(w in body.lower() for w in (b"logprobs", b"stream_options", b"include_usage")):
            return False
        self.no_inject.add(ctx.body.get("model"))
        return True

    def complete(self, ctx: _Ctx, resp: httpx.Response, request: httpx.Request, body: bytes) -> httpx.Response:
        if resp.status_code < 400:
            try:
                obj = json.loads(body)
                self.auditor.audit(ctx.body, obj, ctx.host, ctx.language)
                if ctx.added_logprobs:
                    for ch in obj.get("choices") or []:
                        ch["logprobs"] = None
                    body = json.dumps(obj).encode()
            except Exception:
                log.exception("tokencanary: audit failed")
        return _response(resp, request, content=body)

    def streaming(self, ctx: _Ctx, resp: httpx.Response, request: httpx.Request, stream_cls) -> httpx.Response | None:
        if not (ctx.stream and resp.headers.get("content-type", "").startswith("text/event-stream")):
            return None
        return _response(resp, request, stream=stream_cls(resp, _Stream(self.auditor, ctx)))


def _response(resp: httpx.Response, request: httpx.Request, **kw) -> httpx.Response:
    headers = [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in _HOP]
    return httpx.Response(resp.status_code, headers=headers, request=request, extensions=resp.extensions, **kw)


class _Stream:
    """Passes SSE events through (minus our additions) and audits the reassembled response."""

    def __init__(self, auditor: Auditor, ctx: _Ctx):
        self.auditor, self.ctx = auditor, ctx
        self.buf = b""
        self.meta: dict = {}
        self.usage = None
        self.choices: dict[int, dict] = {}
        self.finished = self.audited = False

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buf = (self.buf + chunk).replace(b"\r\n", b"\n")
        out = []
        while b"\n\n" in self.buf:
            event, self.buf = self.buf.split(b"\n\n", 1)
            if (ev := self._event(event)) is not None:
                out.append(ev + b"\n\n")
        return out

    def tail(self) -> list[bytes]:
        rest, self.buf = self.buf, b""
        ev = self._event(rest) if rest.strip() else rest
        return [ev] if ev else []

    def _event(self, raw: bytes) -> bytes | None:
        data = b"\n".join(ln[5:].lstrip() for ln in raw.split(b"\n") if ln.startswith(b"data:"))
        if data.strip() == b"[DONE]":
            self.finalize()  # clients usually stop reading here
            return raw
        try:
            obj = json.loads(data)
        except ValueError:
            return raw
        if not isinstance(obj, dict):
            return raw
        self._absorb(obj)
        if self.ctx.added_usage and obj.get("usage") and not obj.get("choices"):
            return None
        changed = False
        if self.ctx.added_usage and obj.get("usage") is not None:
            obj["usage"], changed = None, True
        if self.ctx.added_logprobs:
            for ch in obj.get("choices") or []:
                if ch.get("logprobs") is not None:
                    ch["logprobs"], changed = None, True
        return b"data: " + json.dumps(obj).encode() if changed else raw

    def _absorb(self, obj: dict) -> None:
        if not self.meta:
            self.meta = {k: obj.get(k) for k in ("id", "model")}
        if obj.get("usage"):
            self.usage = obj["usage"]
        for ch in obj.get("choices") or []:
            st = self.choices.setdefault(ch.get("index", 0), {"text": [], "logprobs": None, "tools": False})
            delta = ch.get("delta") or {}
            st["text"].append(delta.get("content") or "")
            st["tools"] |= bool(delta.get("tool_calls"))
            if (lp := (ch.get("logprobs") or {}).get("content")) is not None:
                st["logprobs"] = st["logprobs"] or []
                st["logprobs"].extend(lp)
            if ch.get("finish_reason"):
                self.finished = True

    def finalize(self) -> None:
        if not self.finished or self.audited:
            return
        self.audited = True
        choices = [
            {
                "index": i,
                "message": {"content": "".join(st["text"]), "tool_calls": [{}] if st["tools"] else None},
                "logprobs": {"content": st["logprobs"]} if st["logprobs"] is not None else None,
            }
            for i, st in sorted(self.choices.items())
        ]
        try:
            response = {**self.meta, "choices": choices, "usage": self.usage}
            self.auditor.audit(self.ctx.body, response, self.ctx.host, self.ctx.language)
        except Exception:
            log.exception("tokencanary: stream audit failed")


class _SyncStream(httpx.SyncByteStream):
    def __init__(self, resp, state: _Stream):
        self.resp, self.state = resp, state

    def __iter__(self):
        for chunk in self.resp.iter_bytes():
            yield from self.state.feed(chunk)
        yield from self.state.tail()
        self.state.finalize()

    def close(self):
        self.state.finalize()
        self.resp.close()


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, resp, state: _Stream):
        self.resp, self.state = resp, state

    async def __aiter__(self):
        async for chunk in self.resp.aiter_bytes():
            for ev in self.state.feed(chunk):
                yield ev
        for ev in self.state.tail():
            yield ev
        self.state.finalize()

    async def aclose(self):
        self.state.finalize()
        await self.resp.aclose()


class AuditTransport(httpx.BaseTransport):
    """Drop-in httpx transport: `OpenAI(http_client=httpx.Client(transport=AuditTransport()))`."""

    def __init__(self, auditor: Auditor | None = None, inner: httpx.BaseTransport | None = None, inject_logprobs: bool = True):
        self.auditor = auditor or Auditor()
        self.core = _Core(self.auditor, inject_logprobs)
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        content = request.read()
        req, ctx = self.core.prepare(request, content)
        if ctx is None:
            return self.inner.handle_request(request)
        resp = self.inner.handle_request(req)
        if resp.status_code == 400:
            body = resp.read()
            if not self.core.injection_rejected(ctx, resp, body):
                return _response(resp, request, content=body)
            req, ctx = self.core.prepare(request, content, inject=False)
            resp = self.inner.handle_request(req)
        return self.core.streaming(ctx, resp, request, _SyncStream) or self.core.complete(ctx, resp, request, resp.read())

    def close(self):
        self.inner.close()


class AsyncAuditTransport(httpx.AsyncBaseTransport):
    """Drop-in async transport for `AsyncOpenAI`."""

    def __init__(self, auditor: Auditor | None = None, inner: httpx.AsyncBaseTransport | None = None, inject_logprobs: bool = True):
        self.auditor = auditor or Auditor()
        self.core = _Core(self.auditor, inject_logprobs)
        self.inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        content = await request.aread()
        req, ctx = self.core.prepare(request, content)
        if ctx is None:
            return await self.inner.handle_async_request(request)
        resp = await self.inner.handle_async_request(req)
        if resp.status_code == 400:
            body = await resp.aread()
            if not self.core.injection_rejected(ctx, resp, body):
                return _response(resp, request, content=body)
            req, ctx = self.core.prepare(request, content, inject=False)
            resp = await self.inner.handle_async_request(req)
        return self.core.streaming(ctx, resp, request, _AsyncStream) or self.core.complete(ctx, resp, request, await resp.aread())

    async def aclose(self):
        await self.inner.aclose()


def http_client(auditor: Auditor | None = None, **kw) -> httpx.Client:
    """`OpenAI(http_client=tokencanary.http_client())`; keyword arguments go to Auditor."""
    from openai import DefaultHttpxClient

    return DefaultHttpxClient(transport=AuditTransport(auditor or Auditor(**kw)))


def async_http_client(auditor: Auditor | None = None, **kw) -> httpx.AsyncClient:
    """`AsyncOpenAI(http_client=tokencanary.async_http_client())`; keyword arguments go to Auditor."""
    from openai import DefaultAsyncHttpxClient

    return DefaultAsyncHttpxClient(transport=AsyncAuditTransport(auditor or Auditor(**kw)))
