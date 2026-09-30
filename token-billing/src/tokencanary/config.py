from __future__ import annotations

import dataclasses
import fnmatch
import os
import tomllib

HOME = os.path.join(os.path.expanduser("~"), ".tokencanary")
POLICY_KEYS = {"alpha", "tolerance", "confidence", "min_overcharge", "over_cap", "count_mismatch"}


@dataclasses.dataclass(frozen=True)
class Config:
    alpha: float = 0.01  # per-response false-alarm level of the likelihood test
    tolerance: float = 3.0  # provider alert needs rejection rate > tolerance * alpha
    confidence: float = 1e-6  # endpoint alert when evidence reaches 1 / confidence
    min_overcharge: float = 0.001  # provider alert needs worst-case overcharge above this share
    over_cap: bool = True
    count_mismatch: bool = True
    count_rules: bool = True  # count-only rules for responses without log-probs (needs calibration gaps)
    batch_size: int = 100
    batch_alpha: float = 0.05
    batch_margin: float = 0.0005  # share of canonical tokens subtracted per response in the batch rule
    guess_hf: bool = True  # treat unknown `org/model` names as Hugging Face repos
    tokenizers: dict = dataclasses.field(default_factory=dict)  # pattern -> tokenizer spec
    scorers: dict = dataclasses.field(default_factory=dict)  # pattern -> EchoScorer arguments
    models: dict = dataclasses.field(default_factory=dict)  # pattern -> policy overrides
    calibration: str | None = None
    log: str | None = os.path.join(HOME, "audit.jsonl")
    state: str | None = os.path.join(HOME, "state.json")

    def __post_init__(self):
        for key in ("calibration", "log", "state"):
            if value := getattr(self, key):
                object.__setattr__(self, key, os.path.expanduser(value))

    def for_model(self, model: str) -> Config:
        """This config with the most specific `[models."pattern"]` override applied."""
        override = best_match(model, self.models)
        return dataclasses.replace(self, **override) if override else self

    def replace(self, **changes) -> Config:
        return dataclasses.replace(self, **changes)


def best_match(name: str, table: dict):
    """Value of the most specific matching pattern: exact beats glob, then most literal characters."""
    name = name.lower()
    best, best_len = None, -1
    for pattern, value in table.items():
        p = pattern.lower()
        if p == name:
            return value
        literal = len(p) - p.count("*") - p.count("?")
        if literal > best_len and fnmatch.fnmatchcase(name, p):
            best, best_len = value, literal
    return best


def load_config(path: str | None = None) -> Config:
    """Explicit path, else $TOKENCANARY_CONFIG (empty disables), else ~/.tokencanary/config.toml if present."""
    if path is None:
        path = os.environ.get("TOKENCANARY_CONFIG", os.path.join(HOME, "config.toml"))
        if not path or not os.path.exists(path):
            return Config()
    with open(os.path.expanduser(path), "rb") as f:
        data = tomllib.load(f)
    data.update(data.pop("checks", {}))
    for pattern, override in data.get("models", {}).items():
        if unknown := set(override) - POLICY_KEYS:
            raise ValueError(f"[models.{pattern!r}]: unknown keys {sorted(unknown)}")
    try:
        return Config(**data)
    except TypeError as exc:
        raise ValueError(f"{path}: {exc}") from None
