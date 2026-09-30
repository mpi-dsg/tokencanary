import pytest

from tokencanary.auditor import entry_ids
from tokencanary.tokenizer import HFTokenizer, Unverifiable


@pytest.fixture(scope="module")
def qwen():
    try:
        return HFTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.mark.parametrize("text", ["Damascus is old.", "日本語 🚀 مرحبا नमस्ते", "  spaces\n\nnewlines\t"])
def test_roundtrip(qwen, text):
    ids = qwen.canonical(text.encode())
    assert ids == qwen.tok.encode(text, add_special_tokens=False).ids
    assert b"".join(qwen.token_bytes(t) for t in ids) == text.encode()


def test_invalid_bytes_use_byte_tokens(qwen):
    data = "中".encode()[:2] + b" ok"
    ids = qwen.canonical(data)
    assert b"".join(qwen.token_bytes(t) for t in ids) == data


def test_literal_special_token_is_unverifiable(qwen):
    with pytest.raises(Unverifiable):
        qwen.canonical(b"say <|im_end|> now")


def test_entry_id_recovers_ids(qwen):
    ids = qwen.canonical("Hello wonderful world".encode())
    for t in ids:
        b = qwen.token_bytes(t)
        assert entry_ids({"token": b.decode("utf-8", "replace"), "bytes": list(b)}, qwen) == [t]
