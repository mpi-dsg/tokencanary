import math

import pytest

from tokencanary.calibration import Calibration, binom_cdf, binom_sf, conformal_threshold, script_bucket
from tokencanary.tokenizer import split_utf8

from helpers import TOK


def test_split_utf8():
    data = "héllo".encode() + b"\xff\xfe" + "世".encode() + b"\xe4\xb8"
    parts = split_utf8(data)
    assert b"".join(p for _, p in parts) == data
    assert parts == [(True, "héllo".encode()), (False, b"\xff\xfe"), (True, "世".encode()), (False, b"\xe4\xb8")]


@pytest.mark.parametrize("text", ["Damascus is old.", "日本語のテキスト 🚀", "", "<|endoftext|> literal"])
def test_canonical_roundtrip(text):
    ids = TOK.canonical(text.encode())
    assert b"".join(TOK.token_bytes(t) for t in ids) == text.encode()


def test_canonical_invalid_bytes():
    data = b"ok \xe4\xb8"
    ids = TOK.canonical(data)
    assert b"".join(TOK.token_bytes(t) for t in ids) == data


def test_conformal_threshold():
    assert conformal_threshold([0.0] * 50, 0.01) == -math.inf  # k = 0
    scores = [0.0] * 97 + [-5.0, -3.0, -1.0]  # n = 100, k = 1
    assert conformal_threshold(scores, 0.01) == -5.0
    assert conformal_threshold(scores, 0.03) == -1.0  # k = floor(3.03) = 3 -> third smallest


def test_script_bucket():
    assert script_bucket("Hello world") == "latin"
    assert script_bucket("مرحبا بالعالم") == "arabic"
    assert script_bucket("नमस्ते दुनिया") == "devanagari"
    assert script_bucket("你好，世界") == "cjk"


def test_binom_cdf():
    assert binom_cdf(-1, 10, 0.01) == 0.0
    assert binom_cdf(10, 10, 0.3) == 1.0
    assert abs(binom_cdf(0, 10, 0.5) - 0.5**10) < 1e-12
    assert binom_sf(0, 10, 0.3) == 1.0
    assert abs(binom_sf(100, 100, 0.05) - 0.05**100) < 1e-140  # no cancellation in tiny tails


def test_calibration_roundtrip(tmp_path):
    path = str(tmp_path / "cal.json")
    cal = Calibration(path)
    for s in [0.0] * 18 + [-4.0, -2.0]:
        cal.add("m|latin", s)
    cal.save()
    again = Calibration(path)
    assert again.threshold("m|latin", 0.05) == -4.0
    assert again.threshold("m|latin", 0.01) == -math.inf
    assert again.threshold("m|arabic", 0.05) is None
