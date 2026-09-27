"""Suite-wide: no test can ever reach a billable data API.

The cost guard (confirm=True, the price ceiling) is the first line of defence.
This is the second, and it does not depend on the guard being correct. On
2026-09-27 the guard was deliberately broken to prove the tests catch it, and a
test that had no network stub of its own fell straight through to Databento
and fetched real, billed MES data. A broken guard must only ever produce a
failing test.

So for every test: the API key is removed from the environment, and the
Databento client factory is replaced with one that raises. Tests that need a
client install their own fake (monkeypatch.setattr on `_client`), which
overrides this one for that test only.
"""
from __future__ import annotations

import pytest


class NetworkDisabledInTests(RuntimeError):
    """A test tried to build a real Databento client."""


@pytest.fixture(autouse=True)
def _no_billable_network(monkeypatch):
    import data.sources.databento_client as dc

    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(dc, "load_dotenv", lambda *a, **k: False)

    def refuse(*_a, **_k):
        raise NetworkDisabledInTests(
            "a test tried to reach Databento; stub _client in the test instead")
    monkeypatch.setattr(dc, "_client", refuse)
    yield
