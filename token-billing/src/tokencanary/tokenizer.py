from __future__ import annotations

import functools
import json
import logging
import re
import tomllib
from importlib import resources

from .config import best_match

log = logging.getLogger("tokencanary")


class Unverifiable(Exception):
    pass


def _char_len(data: bytes, i: int) -> int:
    lead = data[i]
    if lead < 0x80:
        return 1
    k = 2 if 0xC2 <= lead <= 0xDF else 3 if 0xE0 <= lead <= 0xEF else 4 if 0xF0 <= lead <= 0xF4 else 0
    if not k or i + k > len(data):
        return 0
    try:
        data[i : i + k].decode("utf-8")
    except UnicodeDecodeError:
        return 0
    return k


def split_utf8(data: bytes) -> list[tuple[bool, bytes]]:
    """Split into maximal valid UTF-8 runs and the invalid spans between them."""
    out, i, start = [], 0, 0
    while i < len(data):
        if k := _char_len(data, i):
            i += k
            continue
        if start < i:
            out.append((True, data[start:i]))
        j = i
        while j < len(data) and not _char_len(data, j):
            j += 1
        out.append((False, data[i:j]))
        i = start = j
    if start < len(data):
        out.append((True, data[start:]))
    return out


class Tokenizer:
    name = "?"

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def byte_token(self, byte: int) -> int:
        raise NotImplementedError

    def token_bytes(self, tid: int) -> bytes:
        raise NotImplementedError

    def vocab(self):
        """Yield (id, bytes) for every non-special token."""
        raise NotImplementedError

    def special_id(self, text: str) -> int | None:
        return None

    def canonical(self, data: bytes) -> list[int]:
        """Encode valid UTF-8 runs normally and invalid spans as byte tokens."""
        ids = []
        for valid, chunk in split_utf8(data):
            ids += self.encode(chunk.decode()) if valid else [self.byte_token(b) for b in chunk]
        if b"".join(map(self.token_bytes, ids)) != data:
            raise Unverifiable(f"{self.name}: canonical encoding does not round-trip")
        return ids

    @functools.cached_property
    def _by_bytes(self) -> dict[bytes, list[int]]:
        index: dict[bytes, list[int]] = {}
        for tid, b in self.vocab():
            index.setdefault(b, []).append(tid)
        return index

    def ids_for_bytes(self, b: bytes) -> list[int]:
        return self._by_bytes.get(b, [])


class TiktokenTokenizer(Tokenizer):
    def __init__(self, encoding: str):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"
        self._special = dict(self.enc._special_tokens)

    def encode(self, text):
        return self.enc.encode(text, disallowed_special=())

    def byte_token(self, byte):
        return self.enc.encode_single_token(bytes([byte]))

    def token_bytes(self, tid):
        return self.enc.decode_single_token_bytes(tid)

    def vocab(self):
        special = set(self._special.values())
        for tid in range(self.enc.n_vocab):
            if tid not in special:
                try:
                    yield tid, self.enc.decode_single_token_bytes(tid)
                except KeyError:
                    pass

    def special_id(self, text):
        return self._special.get(text)


@functools.cache
def _bytes_to_unicode() -> dict[int, str]:
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTE_PIECE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class HFTokenizer(Tokenizer):
    """Hugging Face `tokenizers`: byte-level BPE or SentencePiece-style byte fallback."""

    def __init__(self, tok, name: str = "hf"):
        self.tok = tok
        self.name = name
        spec = json.loads(tok.to_str())
        self.byte_level = '"ByteLevel"' in json.dumps([spec.get("pre_tokenizer"), spec.get("decoder")])
        self._u2b = {c: b for b, c in _bytes_to_unicode().items()}
        added = tok.get_added_tokens_decoder()
        self._added = {tid: t.content for tid, t in added.items()}
        self._special = {t.content: tid for tid, t in added.items() if t.special}

    @classmethod
    def from_pretrained(cls, repo: str) -> HFTokenizer:
        from tokenizers import Tokenizer as T

        return cls(T.from_pretrained(repo), name=f"hf:{repo}")

    @classmethod
    def from_file(cls, path: str) -> HFTokenizer:
        from tokenizers import Tokenizer as T

        return cls(T.from_file(path), name=f"file:{path}")

    def encode(self, text):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if set(ids) & set(self._special.values()):
            raise Unverifiable("response contains literal special-token text")
        return ids

    def byte_token(self, byte):
        piece = _bytes_to_unicode()[byte] if self.byte_level else f"<0x{byte:02X}>"
        if (tid := self.tok.token_to_id(piece)) is None:
            raise Unverifiable(f"{self.name}: no byte token for 0x{byte:02X}")
        return tid

    def token_bytes(self, tid):
        if tid in self._added:
            return self._added[tid].encode()
        piece = self.tok.id_to_token(tid)
        if piece is None:
            raise Unverifiable(f"{self.name}: unknown token id {tid}")
        if self.byte_level:
            try:
                return bytes(self._u2b[c] for c in piece)
            except KeyError:
                return piece.encode()
        if m := _BYTE_PIECE.match(piece):
            return bytes([int(m.group(1), 16)])
        return piece.replace("▁", " ").encode()

    def vocab(self):
        special = set(self._special.values())
        for tid in self.tok.get_vocab(with_added_tokens=True).values():
            if tid not in special:
                try:
                    yield tid, self.token_bytes(tid)
                except Unverifiable:
                    pass

    def special_id(self, text):
        return self._special.get(text)


def load_tokenizer(spec: str | Tokenizer) -> Tokenizer:
    """`tiktoken:<encoding>`, `hf:<repo>`, `file:<tokenizer.json>`, or a Tokenizer."""
    if isinstance(spec, Tokenizer):
        return spec
    kind, _, arg = spec.partition(":")
    if kind == "tiktoken":
        return TiktokenTokenizer(arg)
    if kind == "hf":
        return HFTokenizer.from_pretrained(arg)
    if kind == "file":
        return HFTokenizer.from_file(arg)
    raise ValueError(f"unknown tokenizer spec {spec!r}")


@functools.cache
def bundled_registry() -> dict[str, str]:
    return tomllib.loads(resources.files(__package__).joinpath("registry.toml").read_text())["tokenizers"]


class TokenizerRegistry:
    """Model name -> tokenizer: user patterns, then the bundled registry, then `hf:<model>` for `org/model` names."""

    def __init__(self, user: dict | None = None, guess_hf: bool = True):
        self.user = dict(user or {})
        self.guess_hf = guess_hf
        self._cache: dict[str, Tokenizer | None] = {}

    def spec(self, model: str) -> str | Tokenizer | None:
        spec = best_match(model, self.user) or best_match(model, bundled_registry())
        if spec is None and self.guess_hf and "/" in model and not model.startswith("accounts/"):
            spec = f"hf:{model}"
        return spec

    def get(self, model: str) -> Tokenizer | None:
        if model not in self._cache:
            tok = None
            if (spec := self.spec(model)) is not None:
                try:
                    tok = load_tokenizer(spec)
                except Exception as exc:
                    log.warning("tokencanary: cannot load tokenizer %s for %r: %s", spec, model, exc)
            else:
                log.warning("tokencanary: no tokenizer for %r; add it under [tokenizers] in the config", model)
            self._cache[model] = tok
        return self._cache[model]
