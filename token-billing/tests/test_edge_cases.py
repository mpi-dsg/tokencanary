import logging

from tokencanary import Calibration
from tokencanary.auditor import align, text_slice
from tokencanary.commission import count_gaps

from helpers import MODEL, TEXT, TOK, FakeScorer, FakeUpstream, ask, client_for, completion, entries, split_once, sse
from test_remote import chat


def calibration(scores, meta=None, key=f"{MODEL}|latin"):
    cal = Calibration()
    for s in scores:
        cal.add(key, s)
    if meta:
        cal.set_meta(key, meta)
    return cal


META = {"tokenizer": TOK.name, "scorer": None, "temperature": 1.0, "top_p": 1.0, "system": None}


def test_overcharge_counts_answer_tokens_only():
    ids = TOK.canonical(TEXT.encode())
    response = chat(split_once(ids, 3), TOK.canonical(b"A long stretch of reasoning before the answer."))
    client, auditor = client_for(FakeUpstream(lambda body: response), scorers={MODEL: FakeScorer(2.0)},
                                 calibration=calibration([0.0] * 19 + [-2.0]), alpha=0.05)
    ask(client)
    c = auditor.records[0].choices[0]
    assert c.verdict == "reject" and c.answer == len(ids) + 3 and c.reported > c.answer
    st = auditor.monitor.state[f"api.example.com|{MODEL}"]
    assert st["extra"] == 3 and st["tokens"] == c.answer


def test_token_id_must_decode_to_entry_bytes():
    response = completion(TEXT)
    lp = response["choices"][0]["logprobs"]["content"]
    lp[0]["token_id"] = TOK.canonical(b" unrelated")[0]
    client, auditor = client_for(FakeUpstream(lambda body: response), scorers={MODEL: FakeScorer()},
                                 calibration=calibration([0.0] * 20), alpha=0.05)
    ask(client)
    c = auditor.records[0].choices[0]
    assert c.verdict == "unverifiable" and "does not decode" in c.reason


def test_capped_responses_are_not_likelihood_tested():
    ids = TOK.canonical(TEXT.encode())
    capped = completion(TEXT, ids=split_once(ids, 2))
    capped["choices"][0]["finish_reason"] = "length"
    client, auditor = client_for(FakeUpstream(lambda body: capped), scorers={MODEL: FakeScorer()},
                                 calibration=calibration([0.0] * 20), alpha=0.05)
    ask(client)
    rec = auditor.records[0]
    assert rec.choices[0].verdict == "capped" and rec.count_skipped is None  # the count check still runs
    client, auditor = client_for(FakeUpstream(lambda body: sse(TEXT, finish="length")))
    list(ask(client, stream=True))
    assert auditor.records[0].choices[0].capped


def test_calibration_settings_must_match():
    ids = TOK.canonical(TEXT.encode())
    up = FakeUpstream(lambda body: completion(TEXT, ids=split_once(ids, 1)))
    client, auditor = client_for(up, scorers={MODEL: FakeScorer()}, alpha=0.05,
                                 calibration=calibration([0.0] * 20, {**META, "tokenizer": "tiktoken:cl100k_base"}))
    ask(client)
    assert auditor.records[0].choices[0].verdict == "uncalibrated" and "tokenizer" in auditor.records[0].choices[0].reason
    client, auditor = client_for(up, scorers={MODEL: FakeScorer()}, alpha=0.05, calibration=calibration([0.0] * 19 + [-5.0], META))
    ask(client, temperature=0.7)
    assert "temperature 0.7" in auditor.records[0].choices[0].reason
    ask(client)
    assert auditor.records[1].choices[0].verdict == "pass"


def test_calibration_refuses_mixed_settings():
    cal = calibration([0.0], META)
    try:
        cal.set_meta(f"{MODEL}|latin", {**META, "top_p": 0.95})
    except ValueError:
        return
    raise AssertionError("mixed calibration settings were accepted")


def test_system_prompt_difference_only_warns(caplog):
    client, auditor = client_for(FakeUpstream(lambda body: completion(TEXT)), scorers={MODEL: FakeScorer()},
                                 alpha=0.05, calibration=calibration([0.0] * 20, META))
    with caplog.at_level(logging.WARNING, "tokencanary"):
        client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": "Be brief."},
                                                              {"role": "user", "content": "q"}])
    assert auditor.records[0].choices[0].verdict == "pass" and "system prompt differs" in caplog.text


def test_scoring_the_audited_provider_warns(caplog):
    scorer = FakeScorer()
    scorer.url = "https://api.example.com/v1"
    client, _ = client_for(FakeUpstream(lambda body: completion(TEXT)), scorers={MODEL: scorer})
    with caplog.at_level(logging.WARNING, "tokencanary"):
        ask(client)
    assert "not independent" in caplog.text


def test_count_gaps_from_a_trusted_period(tmp_path):
    n = len(TOK.canonical(TEXT.encode()))
    log = str(tmp_path / "audit.jsonl")
    replies = iter([completion(TEXT, logprobs=False, billed=n)] * 3 + [completion(TEXT, logprobs=False, billed=n + 1)])
    client, auditor = client_for(FakeUpstream(lambda body: next(replies)), log=log)
    for _ in range(4):
        ask(client)
    cal = Calibration()
    records = [r.__dict__ | {"choices": [c.__dict__ for c in r.choices]} for r in auditor.records]
    assert count_gaps(records, MODEL, cal, host="api.example.com", until=auditor.records[2].time) == 3
    assert cal.gaps[f"{MODEL}|latin"] == [[0, n]] * 3


def test_placeholder_tokens_align_to_partial_characters():
    data = "中文".encode()
    placeholders = [{"token": "�"}, {"token": "�"}]
    assert align(placeholders, "中".encode()) == [[b"\xe4", b"\xb8\xad"], [b"\xe4\xb8", b"\xad"]]
    byte_ids = [TOK.ids_for_bytes(bytes([b]))[0] for b in "中".encode()]
    lp = [{"token": "�"}] * 3 + entries(TOK.canonical("文".encode()))
    kind, lo, hi, readings = text_slice(lp, data, TOK)
    assert kind == "content" and readings == [[b"\xe4", b"\xb8", b"\xad", "文".encode()]]
    response = completion("中文", ids=byte_ids + TOK.canonical("文".encode()))
    for e in response["choices"][0]["logprobs"]["content"][:3]:
        del e["bytes"]
        e["token"] = "�"
    client, auditor = client_for(FakeUpstream(lambda body: response), scorers={MODEL: FakeScorer(2.0)},
                                 calibration=calibration([0.0] * 20, key=f"{MODEL}|cjk"), alpha=0.05)
    ask(client)
    c = auditor.records[0].choices[0]
    assert c.array == "content" and c.noncanonical and c.verdict == "reject"


def test_reported_length_above_the_limit():
    ids = TOK.canonical(TEXT.encode())
    client, auditor = client_for(FakeUpstream(lambda body: completion(TEXT, billed=5)))
    ask(client, max_tokens=5)
    assert "over_cap" in auditor.records[0].findings and len(ids) > 5
