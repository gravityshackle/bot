"""The billable-fetch guard must never be bypassable by accident.

A cost estimate is not consent. This suite runs entirely offline: it points the
cache at an empty temp dir and disables estimation, so the guard is exercised
without any network call or spend.
"""
from __future__ import annotations

import copy

import pytest
import yaml

from data.sources.databento_client import (
    ConfirmationRequired,
    fetch_ohlcv,
    resolve_window,
)


def base_cfg(tmp_path):
    with open("config/data.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    cfg["cache"]["dir"] = str(tmp_path / "cache")   # guaranteed cache miss
    cfg["cost_control"]["estimate_before_fetch"] = False   # no network
    return cfg


def symbol_cfg():
    with open("config/symbols/MES.yaml") as fh:
        return yaml.safe_load(fh)


def test_uncached_fetch_is_refused_without_confirmation(tmp_path):
    with pytest.raises(ConfirmationRequired, match="would BILL"):
        fetch_ohlcv(base_cfg(tmp_path), symbol_cfg())


def test_refusal_names_the_symbol_and_window(tmp_path):
    cfg = base_cfg(tmp_path)
    start, end = resolve_window(cfg)
    with pytest.raises(ConfirmationRequired) as exc:
        fetch_ohlcv(cfg, symbol_cfg())
    msg = str(exc.value)
    assert "MES" in msg
    assert f"{start:%Y-%m-%d}" in msg and f"{end:%Y-%m-%d}" in msg


def test_dry_run_never_trips_the_guard(tmp_path):
    """Pricing a request must stay free and unblocked."""
    r = fetch_ohlcv(base_cfg(tmp_path), symbol_cfg(), dry_run=True)
    assert r.bars.empty and not r.from_cache


def test_guard_is_what_blocks_not_something_else(tmp_path, monkeypatch):
    """With the guard off, execution must reach the API call -- and no further.

    The network is stubbed out with a sentinel so this proves the guard was the
    blocker WITHOUT issuing a billable request. An earlier version of this test
    let the real call through and actually spent money, which is exactly the
    behaviour this whole module exists to prevent.
    """
    import data.sources.databento_client as dc

    class Reached(RuntimeError):
        pass

    def no_network(*_a, **_kw):
        raise Reached("reached the API layer")

    monkeypatch.setattr(dc, "_client", no_network)

    cfg = base_cfg(tmp_path)
    cfg["cost_control"]["require_explicit_confirmation"] = False
    with pytest.raises(Reached):
        fetch_ohlcv(cfg, symbol_cfg())

    # and with the guard on, it never gets that far
    cfg["cost_control"]["require_explicit_confirmation"] = True
    with pytest.raises(ConfirmationRequired):
        fetch_ohlcv(cfg, symbol_cfg())


def test_confirm_true_still_does_not_bypass_the_cost_ceiling(tmp_path, monkeypatch):
    """Approval covers the request, not an unbounded bill."""
    import data.sources.databento_client as dc
    from data.sources.databento_client import CostLimitExceeded

    monkeypatch.setattr(dc, "estimate_cost", lambda *a, **k: 999.0)
    cfg = base_cfg(tmp_path)
    cfg["cost_control"]["estimate_before_fetch"] = True
    with pytest.raises(CostLimitExceeded, match="exceeds ceiling"):
        fetch_ohlcv(cfg, symbol_cfg(), confirm=True)


def test_cached_window_is_free_and_needs_no_confirmation():
    """The pinned Phase 1 window is on disk; reading it must not be gated."""
    with open("config/data.yaml") as fh:
        cfg = yaml.safe_load(fh)
    r = fetch_ohlcv(cfg, symbol_cfg())
    assert r.from_cache
    assert r.cost_usd == 0.0
    assert not r.bars.empty
