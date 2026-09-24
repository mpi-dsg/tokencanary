import pytest


@pytest.fixture(autouse=True)
def no_user_config(monkeypatch):
    monkeypatch.setenv("TOKENCANARY_CONFIG", "")
