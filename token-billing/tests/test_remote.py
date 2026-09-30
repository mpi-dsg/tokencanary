import json

import httpx
import pytest

from tokencanary import Calibration, EchoScorer, TiktokenTokenizer
from tokencanary.auditor import text_slice
from tokencanary.commission import calibrate
from tokencanary.config import Config
from tokencanary.counts import batch_pvalue, request_reject
from tokencanary.scoring import divergent_spans
from tokencanary.tokenizer import Unverifiable

from helpers import MODEL, TEXT, TOK, FakeScorer, FakeUpstream, ask, client_for, completion, entries, split_once

FAKE = 10**6  # ids at or above this are duplicates that the mock model finds unlikely


class FakeFireworks:
    """Mock scoring provider: echoes token-id prompts with log-prob -1 per token (-10 for fake ids)
    and serves queued chat completions."""

    def __init__(self, chats=(), echo_ok=True):
        self.chats, self.echo_ok, self.calls = list(chats), echo_ok, 0

    def __call__(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=self.chats.pop(0))
        self.calls += 1
        ids = body["prompt"]
        echoed = ids if self.echo_ok else ids[::-1]
        lps = [None] + [-10.0 if t >= FAKE else -1.0 for t in ids[1:]]
        return httpx.Response(200, json={"choices": [{"logprobs": {"token_ids": echoed + [0], "token_logprobs": lps + [-1.0]}}]})


def scorer_for(fw, **kw):
    return EchoScorer("fw-model", base_url="https://fw.test/v1", http=httpx.Client(transport=httpx.MockTransport(fw)), **kw)


def test_divergent_spans_are_index_ranges():
    ids = TOK.canonical(TEXT.encode())
    padded = split_once(ids, 2)
    spans = divergent_spans(TOK.token_bytes, padded, ids)
    assert len(spans) == 2
    for r_lo, r_hi, c_lo, c_hi in spans:
        assert (r_hi - r_lo, c_hi - c_lo) == (2, 1)
        assert b"".join(map(TOK.token_bytes, padded[r_lo:r_hi])) == b"".join(map(TOK.token_bytes, ids[c_lo:c_hi]))


def test_echo_scorer_scores_divergent_windows():
    fw = FakeFireworks()
    ids = TOK.canonical(TEXT.encode())
    llr, spans = scorer_for(fw).llr(split_once(ids, 3), ids, TOK)
    assert (llr, spans) == (-3.0, 3)  # one extra token per span at -1 nats
    assert fw.calls == 2 * 3 * 3  # two windows per span, median of three calls each


def test_echo_mismatch_is_unverifiable():
    ids = TOK.canonical(TEXT.encode())
    with pytest.raises(Unverifiable):
        scorer_for(FakeFireworks(echo_ok=False)).llr(split_once(ids, 1), ids, TOK)


def test_resolve_prefers_ids_the_model_finds_likely():
    ids = TOK.canonical(TEXT.encode())
    candidates = [[FAKE + t, t] if i in (1, 4) else [t] for i, t in enumerate(ids)]
    assert scorer_for(FakeFireworks()).resolve(candidates, TOK) == ids


def chat(content_ids, reasoning_ids=(), finish="stop"):
    """Chat completion whose log-prob array covers reasoning, channel tokens and content, like Fireworks gpt-oss."""
    special = TOK.special_id("<|endoftext|>")
    full = [*reasoning_ids, *content_ids]
    lp = entries(full) + ([{"token": "<|endoftext|>", "token_id": special, "bytes": []}] if reasoning_ids else [])
    text = b"".join(map(TOK.token_bytes, content_ids)).decode()
    message = {"role": "assistant", "content": text}
    if reasoning_ids:
        message["reasoning_content"] = b"".join(map(TOK.token_bytes, reasoning_ids)).decode()
    return {"choices": [{"message": message, "logprobs": {"content": lp}, "finish_reason": finish}],
            "usage": {"completion_tokens": len(lp)}}


def test_calibrate_from_scoring_provider():
    ids = TOK.canonical(TEXT.encode())
    thinking = TOK.canonical(b"Let me think.")
    fw = FakeFireworks(chats=[chat(ids), chat(split_once(ids, 2), thinking), chat(ids, finish="length")])
    cal = Calibration()
    kept = calibrate(scorer_for(fw), MODEL, TOK, [[{"role": "user", "content": "q"}]] * 3, cal, language="en")
    assert kept == 2  # the capped response is skipped
    assert cal.scores[f"{MODEL}|en"] == [0.0, -2.0]
    assert cal.gaps == {}  # count rules calibrate on the audited endpoint, not the scoring provider
    assert cal.margins[MODEL] == 0.0  # the mock scores deterministically
    meta = cal.meta[f"{MODEL}|en"]
    assert meta["tokenizer"] == TOK.name and meta["scorer"]["model"] == "fw-model" and (meta["temperature"], meta["top_p"]) == (1.0, 1.0)


def test_text_slice_finds_content_inside_full_generation():
    ids = TOK.canonical(TEXT.encode())
    lp = chat(ids, TOK.canonical(b"Thinking first."))["choices"][0]["logprobs"]["content"]
    kind, lo, hi, readings = text_slice(lp, TEXT.encode(), TOK)
    assert kind == "full" and hi - lo == len(ids) and hi == len(lp) - 1 and len(readings) == 1
    assert text_slice(entries(ids), TEXT.encode(), TOK)[0] == "content"
    assert text_slice(entries(ids[:-1]), TEXT.encode(), TOK)[0] is None


def test_full_generation_array_is_counted_and_scored():
    ids = TOK.canonical(TEXT.encode())
    response = chat(split_once(ids, 1), TOK.canonical(b"Thinking first."))
    response["usage"]["completion_tokens_details"] = {"reasoning_tokens": 4}
    cal = Calibration()
    for s in [0.0] * 19 + [-5.0]:
        cal.add(f"{MODEL}|latin", s)
    client, auditor = client_for(FakeUpstream(lambda body: response), scorers={MODEL: FakeScorer(2.0)}, calibration=cal, alpha=0.05)
    ask(client)
    rec = auditor.records[0]
    assert rec.count_skipped is None and rec.findings == []  # billed equals the full array
    assert rec.choices[0].array == "full" and rec.choices[0].llr == -2.0 and rec.choices[0].verdict == "pass"
    assert not rec.shadow  # billed tokens include reasoning, so no shadow bill


def test_hidden_reasoning_outside_logprobs_is_not_a_mismatch():
    # OpenRouter -> AkashML gpt-oss-120b: 14 tokens billed, only the answer's 2 entries returned,
    # reasoning text present and reasoning_tokens reported as 0.
    ids = TOK.canonical(b"Paris.")
    response = completion("Paris.", billed=14)
    response["choices"][0]["message"]["reasoning"] = "Paris."
    response["usage"]["completion_tokens_details"] = {"reasoning_tokens": 0}
    client, auditor = client_for(FakeUpstream(lambda body: response))
    ask(client)
    rec = auditor.records[0]
    assert rec.reported == len(ids) == 2
    assert rec.findings == [] and rec.count_skipped == "billed_tokens_outside_logprobs" and rec.status == "unverifiable"


def test_scorer_from_config():
    client, auditor = client_for(FakeUpstream(lambda body: completion(TEXT)),
                                 scorers=None, config=Config(scorers={"gpt-4o*": {"model": "fw-model", "base_url": "https://fw.test/v1"}}))
    assert isinstance(auditor.scorer(MODEL), EchoScorer) and auditor.scorer(MODEL).model == "fw-model"
    assert auditor.scorer("other-model") is None


def test_count_rules():
    calibration = [[0, 100]] * 90 + [[1, 100]] * 10
    assert not request_reject(1, calibration) and request_reject(2, calibration)
    assert batch_pvalue([[0, 100]] * 90 + [[1, 100]] * 10, calibration, margin=0.0005) > 0.3
    assert batch_pvalue([[3, 100]] * 100, calibration, margin=0.0005) < 0.01


def test_count_rules_flag_inflation_without_logprobs():
    n = len(TOK.canonical(TEXT.encode()))
    cal = Calibration()
    for gap in [0] * 90 + [1] * 10:
        cal.add_gap(f"{MODEL}|latin", gap, n)
    alerts = []
    honest = FakeUpstream(lambda body: completion(TEXT, logprobs=False, billed=n))
    client, auditor = client_for(honest, calibration=cal, on_alert=lambda kind, d: alerts.append(kind))
    for _ in range(100):
        ask(client)
    assert alerts == []
    honest.respond = lambda body: completion(TEXT, logprobs=False, billed=n + 3)
    for _ in range(100):
        ask(client)
    assert "count" in alerts


def test_channel_tokens_outside_logprobs_are_not_a_mismatch():
    # OpenRouter -> AkashML gpt-oss-120b without reasoning text: channel tokens billed (+15), answer-only array.
    harmony = TiktokenTokenizer("o200k_harmony")
    response = completion(TEXT, billed=len(TOK.canonical(TEXT.encode())) + 15)
    client, auditor = client_for(FakeUpstream(lambda body: response), tokenizers={MODEL: harmony})
    ask(client)
    rec = auditor.records[0]
    assert rec.findings == [] and rec.count_skipped == "billed_tokens_outside_logprobs" and not rec.shadow
