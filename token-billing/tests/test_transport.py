import asyncio
import json
import threading

import httpx
import openai
import pytest

from tokencanary import AsyncAuditTransport, Auditor, Calibration
from tokencanary.proxy import make_server

from helpers import MODEL, TEXT, TOK, FakeScorer, FakeUpstream, ask, client_for, completion, split_once, sse


def test_honest_nonstream_is_transparent():
    up = FakeUpstream(lambda body: completion(TEXT, logprobs=body.get("logprobs", False)))
    client, auditor = client_for(up)
    resp = ask(client, max_tokens=100)
    assert up.requests[0]["logprobs"] is True  # injected
    assert resp.choices[0].message.content == TEXT
    assert resp.choices[0].logprobs is None  # stripped again for the caller
    rec = auditor.records[0]
    assert rec.status == "ok" and rec.findings == []
    assert rec.billed == rec.reported == rec.canonical
    assert rec.choices[0].noncanonical is False


def test_user_requested_logprobs_are_kept():
    up = FakeUpstream(lambda body: completion(TEXT))
    client, _ = client_for(up)
    resp = ask(client, logprobs=True)
    assert resp.choices[0].logprobs is not None


def test_over_cap():
    ids = TOK.canonical(TEXT.encode())
    up = FakeUpstream(lambda body: completion(TEXT, billed=len(ids) + 50))
    client, auditor = client_for(up)
    ask(client, max_tokens=len(ids))
    rec = auditor.records[0]
    assert "over_cap" in rec.findings and rec.status == "alert"


def test_count_mismatch():
    ids = TOK.canonical(TEXT.encode())
    up = FakeUpstream(lambda body: completion(TEXT, billed=len(ids) + 3))
    client, auditor = client_for(up)
    ask(client)
    rec = auditor.records[0]
    assert rec.findings == ["count_mismatch"]
    assert rec.billed_minus_reported == 3


def test_no_logprobs_is_unverifiable_but_shadow_bill_works():
    up = FakeUpstream(lambda body: completion(TEXT, logprobs=False, billed=40))
    client, auditor = client_for(up)
    ask(client)
    rec = auditor.records[0]
    assert rec.status == "unverifiable"
    assert rec.canonical == len(TOK.canonical(TEXT.encode()))


def test_injection_rejected_falls_back():
    def respond(body):
        if body.get("logprobs"):
            return httpx.Response(400, json={"error": {"message": "logprobs is not supported for this model"}})
        return completion(TEXT, logprobs=False)

    up = FakeUpstream(respond)
    client, auditor = client_for(up)
    ask(client)
    ask(client)
    assert [r.get("logprobs") for r in up.requests] == [True, None, None]  # remembers the model
    assert auditor.records[0].status == "unverifiable"


def test_real_400_passes_through():
    up = FakeUpstream(lambda body: httpx.Response(400, json={"error": {"message": "bad messages"}}))
    client, auditor = client_for(up)
    with pytest.raises(openai.BadRequestError):
        ask(client)
    assert auditor.records == []


def test_stream_drops_injected_usage_and_audits():
    ids = TOK.canonical(TEXT.encode())
    up = FakeUpstream(lambda body: sse(TEXT, billed=len(ids) + 2))
    client, auditor = client_for(up)
    chunks = list(ask(client, stream=True))
    assert up.requests[0]["stream_options"] == {"include_usage": True}
    assert all(c.choices for c in chunks)  # caller never sees the usage-only chunk
    assert all(c.usage is None for c in chunks)
    assert all(c.choices[0].logprobs is None for c in chunks)
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == TEXT
    rec = auditor.records[0]
    assert rec.findings == ["count_mismatch"] and rec.reported == len(ids)


def test_stream_with_user_usage_keeps_usage_chunk():
    up = FakeUpstream(lambda body: sse(TEXT))
    client, auditor = client_for(up)
    chunks = list(ask(client, stream=True, stream_options={"include_usage": True}))
    assert chunks[-1].usage is not None
    assert auditor.records[0].status == "ok"


def _calibrated(scores):
    cal = Calibration()
    for s in scores:
        cal.add(f"{MODEL}|latin", s)
    return cal


def test_likelihood_rejects_padded_report():
    ids = TOK.canonical(TEXT.encode())
    padded = split_once(ids, times=3)
    scorer = FakeScorer(per_token=2.0)
    cal = _calibrated([0.0] * 17 + [-2.0, -2.0, -4.0])  # n=20, alpha=.05 -> tau=-4
    up = FakeUpstream(lambda body: completion(TEXT, ids=padded))
    client, auditor = client_for(up, scorers={MODEL: scorer}, calibration=cal, alpha=0.05)
    ask(client)
    rec = auditor.records[0]
    c = rec.choices[0]
    assert rec.findings == ["likelihood_reject"]  # counts agree: coordinated report
    assert rec.status == "ok"  # a single rejection is evidence, not an alert
    assert c.noncanonical and c.llr == -6.0 and c.tau == -4.0


def test_likelihood_passes_small_honest_deviation():
    ids = TOK.canonical(TEXT.encode())
    up = FakeUpstream(lambda body: completion(TEXT, ids=split_once(ids, 1)))
    client, auditor = client_for(up, scorers={MODEL: FakeScorer(2.0)}, calibration=_calibrated([0.0] * 17 + [-2.0, -2.0, -4.0]), alpha=0.05)
    ask(client)
    assert auditor.records[0].choices[0].verdict == "pass"


def test_canonical_report_needs_no_scoring():
    scorer = FakeScorer()
    up = FakeUpstream(lambda body: completion(TEXT))
    client, auditor = client_for(up, scorers={MODEL: scorer}, calibration=_calibrated([0.0] * 20), alpha=0.05)
    ask(client)
    assert scorer.calls == 0 and auditor.records[0].choices[0].verdict == "pass"


def test_likelihood_off_without_scorer_and_uncalibrated_without_calibration():
    ids = TOK.canonical(TEXT.encode())
    up = FakeUpstream(lambda body: completion(TEXT, ids=split_once(ids, 1)))
    client, auditor = client_for(up)
    ask(client)
    assert auditor.records[0].choices[0].verdict is None
    client, auditor = client_for(up, scorers={MODEL: FakeScorer()})
    ask(client)
    assert auditor.records[0].choices[0].verdict == "uncalibrated"


def test_background_scoring_and_log(tmp_path):
    ids = TOK.canonical(TEXT.encode())
    log = tmp_path / "audit.jsonl"
    up = FakeUpstream(lambda body: completion(TEXT, ids=split_once(ids, 3)))
    client, auditor = client_for(
        up, scorers={MODEL: FakeScorer(2.0)}, calibration=_calibrated([0.0] * 18 + [-1.0, -2.0]), background=True, log=str(log), alpha=0.05
    )
    ask(client)
    auditor.flush()
    line = json.loads(log.read_text().splitlines()[0])
    assert line["findings"] == ["likelihood_reject"]
    report = auditor.report()
    assert "1 of 1 rejected" in report and "shadow bill" in report


def test_async_transport():
    up = FakeUpstream(lambda body: sse(TEXT) if body.get("stream") else completion(TEXT))
    auditor = Auditor(log=None, state=None, background=False)
    http = httpx.AsyncClient(transport=AsyncAuditTransport(auditor, inner=httpx.MockTransport(up)))
    client = openai.AsyncOpenAI(api_key="t", base_url="https://api.example.com/v1", http_client=http)

    async def run():
        await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "x"}])
        stream = await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "x"}], stream=True)
        return [c async for c in stream]

    chunks = asyncio.run(run())
    assert all(c.choices for c in chunks)
    assert [r.status for r in auditor.records] == ["ok", "ok"]


def test_proxy_end_to_end():
    ids = TOK.canonical(TEXT.encode())

    def respond(body):
        return sse(TEXT, billed=len(ids) + 1) if body.get("stream") else completion(TEXT, billed=len(ids) + 1)

    up = FakeUpstream(respond)
    auditor = Auditor(log=None, state=None, background=False)
    server = make_server("https://api.example.com", auditor, port=0, inner=httpx.MockTransport(up))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        client = openai.OpenAI(api_key="t", base_url=f"http://127.0.0.1:{port}/v1")
        resp = ask(client)
        assert resp.choices[0].message.content == TEXT
        chunks = list(ask(client, stream=True))
        assert "".join(c.choices[0].delta.content or "" for c in chunks) == TEXT
    finally:
        server.shutdown()
        server.server_close()
    assert [r.findings for r in auditor.records] == [["count_mismatch"], ["count_mismatch"]]
