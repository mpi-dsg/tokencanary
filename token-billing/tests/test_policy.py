import logging
import random

import httpx
import pytest

from tokencanary import Calibration, TiktokenTokenizer, load_config
from tokencanary.calibration import excess_risk
from tokencanary.config import best_match
from tokencanary.monitor import Monitor
from tokencanary.tokenizer import TokenizerRegistry

from helpers import MODEL, TEXT, TOK, FakeScorer, FakeUpstream, ask, client_for, completion, split_once


def test_best_match_prefers_exact_then_most_literal():
    table = {"gpt-4*": "a", "gpt-4o*": "b", "gpt-4o-mini": "c", "*": "d"}
    assert best_match("gpt-4o-mini", table) == "c"
    assert best_match("GPT-4o-2024", table) == "b"
    assert best_match("gpt-4-turbo", table) == "a"
    assert best_match("llama", table) == "d"
    assert best_match("llama", {"gpt*": 1}) is None


def test_registry_user_beats_bundled_and_guess():
    reg = TokenizerRegistry({"gpt-*": "tiktoken:cl100k_base"})
    assert reg.spec("gpt-4o-mini") == "tiktoken:cl100k_base"  # user entry wins despite being less specific
    assert TokenizerRegistry().spec("gpt-4o-mini") == "tiktoken:o200k_base"
    assert TokenizerRegistry().spec("Qwen/Qwen2.5-7B-Instruct") == "hf:Qwen/Qwen2.5-7B-Instruct"
    assert TokenizerRegistry(guess_hf=False).spec("Qwen/Qwen2.5-7B-Instruct") is None
    assert TokenizerRegistry().spec("accounts/fireworks/models/glm-5p2") is None


def test_load_config(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("""
alpha = 0.02
[checks]
count_mismatch = false
[tokenizers]
"my-model" = "tiktoken:o200k_base"
[models."meta-llama/*"]
tolerance = 5
[models."meta-llama/llama-3.1-8b*"]
tolerance = 8
""")
    cfg = load_config(str(path))
    assert cfg.alpha == 0.02 and cfg.count_mismatch is False and cfg.tokenizers == {"my-model": "tiktoken:o200k_base"}
    assert cfg.for_model("meta-llama/Llama-3.1-8B-Instruct").tolerance == 8
    assert cfg.for_model("meta-llama/Llama-3.3-70B").tolerance == 5
    assert cfg.for_model("gpt-4o").tolerance == 3.0
    path.write_text('[models."x"]\nfoo = 1\n')
    with pytest.raises(ValueError):
        load_config(str(path))


def test_excess_risk():
    assert abs(excess_risk(100, 0.01, 0.03) - 0.97**100) < 1e-12  # k = 1: all 100 honest scores must beat the rate
    assert excess_risk(300, 0.01, 0.03) < 0.01
    assert excess_risk(50, 0.01, 0.03) == 0.0  # k = 0: the test never rejects


def _run(monitor, rate, n, rng, p0=0.03, delta=1e-6, extra=5):
    for i in range(n):
        rejected = rng.random() < rate
        st = monitor.update("h|m", p0, delta, 0.001, rejected, extra if rejected else 0, 100)
        if st["newly_flagged"]:
            return i + 1
    return None


def test_monitor_honest_never_flags_and_padding_flags_fast():
    rng = random.Random(0)
    assert all(_run(Monitor(), 0.01, 20_000, rng) is None for _ in range(20))
    assert _run(Monitor(), 1.0, 100, rng) <= 8
    assert _run(Monitor(), 0.15, 5_000, rng) is not None


def test_monitor_materiality_and_persistence(tmp_path):
    rng = random.Random(1)
    assert _run(Monitor(), 1.0, 200, rng, extra=0) is None  # rejections without extra tokens cost nothing
    path = str(tmp_path / "state.json")
    m = Monitor(path)
    m.update("h|m", 0.03, 1e-6, 0.001, True, 5, 100)
    assert Monitor(path).state["h|m"]["rejected"] == 1
    assert Monitor(path).update("h|m", 0.05, 1e-6, 0.001, False, 0, 100)["tested"] == 1  # new p0 restarts


def test_language_header_is_stripped_and_keys_calibration():
    ids = TOK.canonical(TEXT.encode())
    up = FakeUpstream(lambda body: completion(TEXT, ids=split_once(ids, 3)))
    cal = Calibration()
    for s in [0.0] * 19 + [-10.0]:
        cal.add(f"{MODEL}|de", s)
    client, auditor = client_for(up, scorers={MODEL: FakeScorer(2.0)}, calibration=cal, alpha=0.05)
    seen = []
    auditor_transport = client._client._transport
    inner = auditor_transport.inner
    auditor_transport.inner = httpx.MockTransport(lambda r: (seen.append(dict(r.headers)), inner.handle_request(r))[1])
    ask(client, extra_headers={"x-tokencanary-language": "de"})
    assert all("x-tokencanary-language" not in h for h in seen)
    c = auditor.records[0].choices[0]
    assert c.language == "de" and c.tau == -10.0 and c.verdict == "pass"


class DuplicateTokenizer(TiktokenTokenizer):
    """o200k with a fake second id for every single-byte token, as byte-fallback vocabularies have."""

    def ids_for_bytes(self, b):
        ids = super().ids_for_bytes(b)
        return [10**6 + ids[0], *ids] if len(b) == 1 and ids else ids


class PreferRealIds(FakeScorer):
    def logprob(self, request, ids):
        return super().logprob(request, ids) - 50 * sum(t >= 10**6 for t in ids)


def test_duplicate_ids_resolve_in_providers_favor():
    tok = DuplicateTokenizer("o200k_base")
    ids = TOK.canonical(TEXT.encode())
    scorer = PreferRealIds(2.0)
    cal = Calibration()
    for s in [0.0] * 19 + [-2.0]:
        cal.add(f"{MODEL}|latin", s)
    up = FakeUpstream(lambda body: completion(TEXT))
    client, auditor = client_for(up, scorers={MODEL: scorer}, calibration=cal, alpha=0.05, tokenizers={MODEL: tok})
    ask(client)
    assert auditor.records[0].choices[0].verdict == "pass" and scorer.calls == 0  # canonical is one reading

    up.respond = lambda body: completion(TEXT, ids=split_once(ids, 1))
    ask(client)
    c = auditor.records[1].choices[0]
    assert c.llr == -2.0 and c.verdict == "pass"  # real ids chosen: no -50 penalties


def test_token_outside_vocabulary_is_unverifiable(caplog):
    bad = completion(TEXT)
    entries = bad["choices"][0]["logprobs"]["content"]
    merged = bytes(entries[0]["bytes"] + entries[1]["bytes"] + entries[2]["bytes"])
    bad["choices"][0]["logprobs"]["content"] = [{"token": merged.decode(), "bytes": list(merged)}] + entries[3:]
    bad["usage"]["completion_tokens"] -= 2
    client, auditor = client_for(FakeUpstream(lambda body: bad), scorers={MODEL: FakeScorer()})
    with caplog.at_level(logging.WARNING, "tokencanary"):
        ask(client)
    c = auditor.records[0].choices[0]
    assert c.verdict == "unverifiable" and "not in the" in c.reason
    assert "tokenizer entry" in caplog.text


def test_provider_alert_end_to_end():
    ids = TOK.canonical(TEXT.encode())
    cal = Calibration()
    for s in [0.0] * 97 + [-2.0, -2.0, -2.0]:
        cal.add(f"{MODEL}|latin", s)
    alerts = []
    up = FakeUpstream(lambda body: completion(TEXT, ids=split_once(ids, 3)))
    client, auditor = client_for(up, scorers={MODEL: FakeScorer(2.0)}, calibration=cal, on_alert=lambda kind, d: alerts.append(kind))
    for _ in range(10):
        ask(client)
    assert alerts == ["provider"]  # one alert once evidence crosses 1/delta, none per response
    assert "FLAGGED" in auditor.report()
