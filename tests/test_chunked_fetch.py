"""Chunked historical fetch: calendar-year chunks, each cached as it lands.

The first 5-year pull held a whole instrument in one request and in memory.
A dropped stream lost the instrument, and the machine's memory reaper stopped
the job mid-fetch, so billed transfer could be lost. These tests pin the
redesign: chunks split at 1 January, each saved atomically the moment it
arrives, resume skips saved chunks, one retry on a dropped stream, and the
cost guard applied to the total still to be fetched.

Entirely offline: a fake Databento client records every request and can drop
the stream on demand. No network call, no spend.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml
from databento.common.error import BentoError

import data.sources.databento_client as dc
from data.sources.databento_client import (
    ConfirmationRequired,
    CostLimitExceeded,
    fetch_ohlcv,
    year_chunks,
)
from data.sources.base import SchemaError

UTC = "UTC"
T = pd.Timestamp


def cfg_for(tmp_path, months=60, end="2026-09-12"):
    with open("config/data.yaml", encoding="utf-8") as fh:
        cfg = copy.deepcopy(yaml.safe_load(fh))
    cfg["cache"]["dir"] = str(tmp_path / "cache")
    cfg["window"] = {"months": months, "end": end}
    cfg["cost_control"]["estimate_before_fetch"] = True
    cfg["cost_control"]["require_explicit_confirmation"] = True
    return cfg


def mes():
    with open("config/symbols/MES.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class FakeClient:
    """Stands in for databento.Historical. `drop` maps (schema, chunk start
    date) -> how many times that request drops its stream before succeeding."""

    def __init__(self, drop=None, expirations=None):
        self.calls: list[tuple[str, str]] = []
        self.drop = dict(drop or {})
        self.expirations = expirations or {}
        self.timeseries = self

    def get_range(self, *, dataset, symbols, stype_in, schema, start, end):
        key = (schema, str(T(start).date()))
        self.calls.append(key)
        if self.drop.get(key, 0) > 0:
            self.drop[key] -= 1
            raise BentoError("Error streaming response: Response ended prematurely")
        s, e = T(start), T(end)
        raw = f"MESH{(s.year + 1) % 10}"            # a contract listed that year
        if schema == "definition":
            exp = self.expirations.get(raw, T(f"{s.year + 1}-03-19", tz=UTC))
            return _Store(pd.DataFrame({"raw_symbol": [raw], "expiration": [exp]}))
        ts = [s + pd.Timedelta("1h"), e - pd.Timedelta("1h")]
        df = pd.DataFrame({"symbol": raw, "open": 100.0, "high": 100.5,
                           "low": 99.75, "close": 100.25, "volume": 10},
                          index=pd.DatetimeIndex(ts, name="ts_event"))
        return _Store(df)


class _Store:
    def __init__(self, df):
        self.df = df

    def to_df(self):
        return self.df


@pytest.fixture
def fake(monkeypatch):
    holder = {"client": FakeClient(), "priced": []}
    monkeypatch.setattr(dc, "_client", lambda cfg: holder["client"])

    def price(cfg, scfg, start, end):
        holder["priced"].append(str(T(start).date()))
        return holder.get("per_chunk", 5.0)
    monkeypatch.setattr(dc, "estimate_cost", price)
    return holder


FIVE_YEARS = [("2021-09-12", "2022-01-01"), ("2022-01-01", "2023-01-01"),
              ("2023-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"),
              ("2025-01-01", "2026-01-01"), ("2026-01-01", "2026-09-12")]


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def test_the_window_splits_at_each_first_of_january():
    got = year_chunks(T("2021-09-12", tz=UTC), T("2026-09-12", tz=UTC))
    assert [(str(a.date()), str(b.date())) for a, b in got] == FIVE_YEARS


def test_a_window_inside_one_year_is_one_chunk_under_the_old_cache_name(tmp_path):
    """Keeps the Phase 1-3 caches (2026-06-12..2026-09-12) readable as-is."""
    got = year_chunks(T("2026-06-12", tz=UTC), T("2026-09-12", tz=UTC))
    assert got == [(T("2026-06-12", tz=UTC), T("2026-09-12", tz=UTC))]
    path = dc._cache_path(cfg_for(tmp_path, 3), "MES", *got[0], "ohlcv1m")
    assert path.name == "ohlcv1m_20260612_20260912.parquet"


# --------------------------------------------------------------------------
# the cost guard, per chunk
# --------------------------------------------------------------------------

def test_an_uncached_window_still_needs_confirmation(tmp_path, fake):
    fake["per_chunk"] = 4.0                           # $24 total, under the ceiling
    with pytest.raises(ConfirmationRequired, match="would BILL about \\$24.00") as exc:
        fetch_ohlcv(cfg_for(tmp_path), mes())
    assert "2021-09-12" in str(exc.value) and "2026-09-12" in str(exc.value)
    assert fake["client"].calls == []                 # nothing was requested


def test_the_ceiling_applies_to_the_total_still_to_fetch(tmp_path, fake):
    """Six chunks at $5 is $30, over the $25 ceiling, even though each chunk
    alone is under it. Approval covers the request, not an unbounded bill."""
    with pytest.raises(CostLimitExceeded, match="exceeds ceiling"):
        fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    assert fake["client"].calls == []


def test_dry_run_prices_only_the_chunks_not_on_disk(tmp_path, fake):
    fake["per_chunk"] = 1.0
    fetch_ohlcv(cfg_for(tmp_path, months=12, end="2026-09-12"), mes(), confirm=True)
    fake["priced"].clear()
    # the 12-month window's second chunk (2026-01-01..2026-09-12) is exactly the
    # 5-year window's last chunk, so it is already saved and is not priced
    r = fetch_ohlcv(cfg_for(tmp_path), mes(), dry_run=True)
    assert fake["priced"] == ["2021-09-12", "2022-01-01", "2023-01-01",
                              "2024-01-01", "2025-01-01"]
    assert r.cost_usd == pytest.approx(5.0)
    assert r.bars.empty and not r.from_cache


# --------------------------------------------------------------------------
# saved as it lands, resume, retry
# --------------------------------------------------------------------------

def test_each_chunk_is_saved_as_it_arrives_and_a_rerun_resumes(tmp_path, fake):
    fake["per_chunk"] = 1.0
    fake["client"] = FakeClient(drop={("ohlcv-1m", "2023-01-01"): 2})
    with pytest.raises(BentoError):
        fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    cache = tmp_path / "cache" / "MES"
    saved = sorted(p.name for p in cache.glob("ohlcv1m_*.parquet"))
    assert saved == ["ohlcv1m_20210912_20220101.parquet",
                     "ohlcv1m_20220101_20230101.parquet"]
    assert not list(cache.glob("*.tmp")), "no half-written file left behind"

    fake["client"] = FakeClient()
    fake["priced"].clear()
    r = fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    assert fake["priced"] == ["2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"]
    assert ("ohlcv-1m", "2021-09-12") not in fake["client"].calls
    assert r.cost_usd == pytest.approx(4.0) and not r.from_cache


def test_a_dropped_stream_is_retried_once(tmp_path, fake):
    fake["per_chunk"] = 1.0
    fake["client"] = FakeClient(drop={("ohlcv-1m", "2021-09-12"): 1})
    fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    calls = fake["client"].calls
    assert calls.count(("ohlcv-1m", "2021-09-12")) == 2
    assert (tmp_path / "cache" / "MES" / "ohlcv1m_20210912_20220101.parquet").exists()


def test_a_definitions_stream_drop_is_retried_too(tmp_path, fake):
    """The first 5-year attempt failed exactly here."""
    fake["per_chunk"] = 1.0
    fake["client"] = FakeClient(drop={("definition", "2021-09-12"): 1})
    r = fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    assert len(r.metas) == 6


def test_a_fully_cached_window_is_free_and_needs_no_confirmation(tmp_path, fake):
    fake["per_chunk"] = 1.0
    fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    fake["client"] = FakeClient(); fake["priced"].clear()
    r = fetch_ohlcv(cfg_for(tmp_path), mes())          # no confirm
    assert r.from_cache and r.cost_usd == 0.0
    assert fake["client"].calls == [] and fake["priced"] == []


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def test_assembled_bars_span_every_chunk_in_order(tmp_path, fake):
    fake["per_chunk"] = 1.0
    r = fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    assert len(r.bars) == 12                            # 2 per chunk
    key = ["raw_symbol", "ts"]
    assert r.bars[key].equals(r.bars.sort_values(key)[key].reset_index(drop=True))
    assert not r.bars.duplicated(key).any()
    assert set(r.metas) == {f"MESH{y % 10}" for y in range(2022, 2028)}


def test_contract_years_come_from_expiration_across_the_window(tmp_path, fake):
    """2021's chunk lists MESH2 (2022) and 2026's lists MESH7 (2027)."""
    fake["per_chunk"] = 1.0
    r = fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)
    assert r.metas["MESH2"].year == 2022 and r.metas["MESH7"].year == 2027


def test_conflicting_expirations_for_one_contract_raise(tmp_path, fake, monkeypatch):
    fake["per_chunk"] = 1.0
    real = FakeClient.get_range
    seen = {"n": 0}

    def flaky(self, **kw):
        if kw["schema"] == "definition":
            seen["n"] += 1
            exp = T("2023-03-17", tz=UTC) if seen["n"] == 1 else T("2023-03-24", tz=UTC)
            return _Store(pd.DataFrame({"raw_symbol": ["MESH3"], "expiration": [exp]}))
        return real(self, **kw)
    monkeypatch.setattr(FakeClient, "get_range", flaky)
    with pytest.raises(SchemaError, match="MESH3"):
        fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True)


def test_assemble_false_caches_without_holding_the_window(tmp_path, fake):
    """What the pull uses: download and save chunk by chunk, return no bars."""
    fake["per_chunk"] = 1.0
    r = fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True, assemble=False)
    assert r.bars.empty and r.cost_usd == pytest.approx(6.0)
    assert len(list((tmp_path / "cache" / "MES").glob("ohlcv1m_*.parquet"))) == 6
