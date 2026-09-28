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


# --------------------------------------------------------------------------
# conflicting definitions across chunks (found on real MBT and MCL data)
# --------------------------------------------------------------------------

class ScriptedClient:
    """Two chunks (2025-09-12..2026-01-01, 2026-01-01..2026-09-12) that each
    define MESH6, with per-chunk expirations; MESH6's bars end at `last_bar`."""

    def __init__(self, exp_old, exp_new, last_bar, bar_symbol="MESH6"):
        self.exps = {"2025-09-12": exp_old, "2026-01-01": exp_new}
        self.last_bar, self.bar_symbol = last_bar, bar_symbol
        self.timeseries = self

    def get_range(self, *, dataset, symbols, stype_in, schema, start, end):
        s, e = T(start), T(end)
        if schema == "definition":
            return _Store(pd.DataFrame({"raw_symbol": ["MESH6"],
                                        "expiration": [self.exps[str(s.date())]]}))
        ts = [s + pd.Timedelta("1h")]
        if s <= self.last_bar < e:
            ts.append(self.last_bar)
        return _Store(pd.DataFrame({"symbol": self.bar_symbol, "open": 100.0,
                                    "high": 100.5, "low": 99.75, "close": 100.25,
                                    "volume": 10},
                                   index=pd.DatetimeIndex(ts, name="ts_event")))


def _two_chunks(tmp_path, fake, exp_old, exp_new, last_bar, **kw):
    fake["per_chunk"] = 1.0
    fake["client"] = ScriptedClient(T(exp_old, tz=UTC), T(exp_new, tz=UTC),
                                    T(last_bar, tz=UTC), **kw)
    return fetch_ohlcv(cfg_for(tmp_path, months=12), mes(), confirm=True)


def test_the_most_recent_definition_wins_when_the_bars_agree(tmp_path, fake):
    """MBT's real case: 2022's definitions dated MBTH3 a week early, 2023's
    corrected it, and the contract traded until one minute before the later
    date. The newer definition wins, and the resolution is reported."""
    r = _two_chunks(tmp_path, fake, "2026-03-13 15:00", "2026-03-20 15:00",
                    "2026-03-20 14:59")
    assert r.metas["MESH6"].expiration == T("2026-03-20 15:00", tz=UTC)
    assert any("MESH6" in x for x in r.resolutions)


def test_a_correction_to_an_earlier_date_wins_too(tmp_path, fake):
    """MCL's real case: MCLN2's expiry was brought forward by a holiday, so the
    newer definition is the EARLIER date. Newest wins either way."""
    r = _two_chunks(tmp_path, fake, "2026-03-23 18:30", "2026-03-20 18:30",
                    "2026-03-20 18:24")
    assert r.metas["MESH6"].expiration == T("2026-03-20 18:30", tz=UTC)


def test_trading_after_the_chosen_expiration_raises(tmp_path, fake):
    with pytest.raises(SchemaError, match="MESH6.*after"):
        _two_chunks(tmp_path, fake, "2026-03-27 15:00", "2026-03-20 15:00",
                    "2026-03-21 10:00")


def test_bars_stopping_well_before_the_chosen_expiration_raise(tmp_path, fake):
    """'Trading right up to it' is part of the corroboration: a last bar ten
    days early does not confirm the chosen date."""
    with pytest.raises(SchemaError, match="MESH6.*before"):
        _two_chunks(tmp_path, fake, "2026-03-13 15:00", "2026-03-20 15:00",
                    "2026-03-10 15:00")


def test_a_conflict_with_no_bars_to_judge_by_raises(tmp_path, fake):
    with pytest.raises(SchemaError, match="MESH6.*no bars"):
        _two_chunks(tmp_path, fake, "2026-03-13 15:00", "2026-03-20 15:00",
                    "2026-03-20 14:59", bar_symbol="MESM6")


def test_assemble_false_caches_without_holding_the_window(tmp_path, fake):
    """What the pull uses: download and save chunk by chunk, return no bars."""
    fake["per_chunk"] = 1.0
    r = fetch_ohlcv(cfg_for(tmp_path), mes(), confirm=True, assemble=False)
    assert r.bars.empty and r.cost_usd == pytest.approx(6.0)
    assert len(list((tmp_path / "cache" / "MES").glob("ohlcv1m_*.parquet"))) == 6
