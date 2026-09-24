import json

import httpx
import openai

from tokencanary import Auditor, AuditTransport, TiktokenTokenizer

TOK = TiktokenTokenizer("o200k_base")
MODEL = "gpt-4o-mini"
TEXT = "Damascus is often called the oldest continuously inhabited city in the world."


def client_for(upstream, **auditor_kwargs):
    """OpenAI client whose transport audits into a fresh, in-memory Auditor and talks to `upstream`."""
    auditor = Auditor(**{"log": None, "state": None, "background": False, **auditor_kwargs})
    http = httpx.Client(transport=AuditTransport(auditor, inner=httpx.MockTransport(upstream)))
    return openai.OpenAI(api_key="test", base_url="https://api.example.com/v1", http_client=http), auditor


def ask(client, **kw):
    return client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Oldest city?"}], **kw)


def entries(ids):
    out = []
    for t in ids:
        b = TOK.token_bytes(t)
        out.append({"token": b.decode("utf-8", "replace"), "logprob": -0.1, "bytes": list(b), "top_logprobs": []})
    return out


def split_once(ids, times=1):
    """Replace the first `times` splittable tokens by two tokens with the same bytes."""
    out, done = [], 0
    for t in ids:
        b = TOK.token_bytes(t)
        if done < times and len(b) >= 2:
            for i in range(1, len(b)):
                left, right = TOK.ids_for_bytes(b[:i]), TOK.ids_for_bytes(b[i:])
                if left and right:
                    out += [left[0], right[0]]
                    done += 1
                    break
            else:
                out.append(t)
        else:
            out.append(t)
    assert done == times, "could not split"
    return out


def completion(text, ids=None, billed=None, logprobs=True):
    ids = TOK.canonical(text.encode()) if ids is None else ids
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": {"content": entries(ids)} if logprobs else None,
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(ids) if billed is None else billed, "total_tokens": 0},
    }


def sse(text, ids=None, billed=None):
    ids = TOK.canonical(text.encode()) if ids is None else ids
    events = []
    base = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0, "model": MODEL}
    events.append({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}]})
    for t, e in zip(ids, entries(ids)):
        events.append({**base, "choices": [{"index": 0, "delta": {"content": TOK.token_bytes(t).decode("utf-8", "replace")}, "logprobs": {"content": [e]}, "finish_reason": None}]})
    events.append({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}]})
    events.append({**base, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": len(ids) if billed is None else billed, "total_tokens": 0}})
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


class FakeUpstream:
    """Mock OpenAI endpoint: returns whatever `respond(body)` builds and records requests."""

    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append(body)
        out = self.respond(body)
        if isinstance(out, httpx.Response):
            return out
        if isinstance(out, bytes):
            return httpx.Response(200, content=out, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=out)


class FakeScorer:
    """Every token costs `per_token` nats, so each extra token lowers the LLR by that much."""

    def __init__(self, per_token=2.0):
        self.per_token = per_token
        self.calls = 0

    def logprob(self, request, ids):
        self.calls += 1
        return -self.per_token * len(ids)
