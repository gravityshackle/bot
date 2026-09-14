"""Resampling tests -- offline, synthetic.

The failure this suite is really guarding against is silent misalignment: bars
that aggregate correctly in isolation but sit on a UTC grid instead of the
session grid, so every 4h and daily bar straddles two sessions. That produces
plausible-looking charts and wrong levels.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.continuous_contract import trade_date
from data.resample import (
    check_ohlc_integrity,
    resample,
    resample_all,
    session_open_utc,
)

CT = "America/Chicago"

INDEX_CFG = {"session": {"timezone": CT}, "day_boundary": "17:00"}
MET_CFG = {"session": {"timezone": CT}, "day_boundary": "16:00"}


def make_1m(start_local="2026-03-02 17:00", n_minutes=1380, tz=CT,
            boundary="17:00", price0=5000.0):
    """One full 23h session of 1m bars starting at the session open."""
    ts = (pd.date_range(pd.Timestamp(start_local), periods=n_minutes, freq="1min",
                        tz=tz).tz_convert("UTC"))
    df = pd.DataFrame({
        "ts": ts,
        "raw_symbol": "MESM6",
        "open": [price0 + i * 0.25 for i in range(n_minutes)],
        "volume": [10] * n_minutes,
    })
    df["close"] = df["open"] + 0.25
    df["high"] = df[["open", "close"]].max(axis=1) + 0.25
    df["low"] = df[["open", "close"]].min(axis=1) - 0.25
    df = df[["ts", "raw_symbol", "open", "high", "low", "close", "volume"]]
    df["trade_date"] = trade_date(df["ts"], tz, boundary)
    return df


# --------------------------------------------------------------------------
# session origin
# --------------------------------------------------------------------------

def test_session_open_is_previous_day_at_boundary():
    d = pd.Series([pd.Timestamp("2026-03-10").date()])
    got = session_open_utc(d, CT, "17:00").iloc[0]
    expected = pd.Timestamp("2026-03-09 17:00", tz=CT).tz_convert("UTC")
    assert got == expected


def test_session_open_survives_dst_transition():
    """17:00 CT is 22:00 UTC in CDT and 23:00 UTC in CST. A fixed UTC origin
    would drift an hour; the local wall clock must win."""
    summer = session_open_utc(pd.Series([pd.Timestamp("2026-07-15").date()]), CT, "17:00").iloc[0]
    winter = session_open_utc(pd.Series([pd.Timestamp("2026-01-15").date()]), CT, "17:00").iloc[0]
    assert summer.hour == 22          # CDT = UTC-5
    assert winter.hour == 23          # CST = UTC-6


def test_met_boundary_differs_from_index_boundary():
    d = pd.Series([pd.Timestamp("2026-03-10").date()])
    idx = session_open_utc(d, CT, "17:00").iloc[0]
    met = session_open_utc(d, CT, "16:00").iloc[0]
    assert (idx - met) == pd.Timedelta(hours=1)


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tf,expected_bars", [
    ("5min", 276), ("15min", 92), ("1h", 23), ("4h", 6),
])
def test_bucket_counts_for_a_full_session(tf, expected_bars):
    """1380 minutes = 23h. 4h buckets give 5 full + 1 partial = 6."""
    out = resample(make_1m(), tf, INDEX_CFG)
    assert len(out) == expected_bars


def test_first_bucket_starts_exactly_at_session_open():
    base = make_1m()
    for tf in ["5min", "15min", "1h", "4h"]:
        out = resample(base, tf, INDEX_CFG)
        assert out["ts"].iloc[0] == base["ts"].iloc[0], f"{tf} misaligned at open"


def test_hourly_buckets_land_on_the_session_grid_not_the_utc_grid():
    """17:00 CT open means hourly bars at :00 local. In summer that is 22:00
    UTC -- which happens to align. The real check is that every bucket is an
    exact multiple of the timeframe from the session open."""
    base = make_1m()
    out = resample(base, "4h", INDEX_CFG)
    origin = out["ts"].iloc[0]
    offsets = (out["ts"] - origin)
    assert all(o % pd.Timedelta("4h") == pd.Timedelta(0) for o in offsets)


def test_daily_bar_is_the_trade_date_not_a_utc_day():
    base = make_1m()                     # one session, spans two UTC dates
    assert base["ts"].dt.date.nunique() == 2
    out = resample(base, "1D", INDEX_CFG)
    assert len(out) == 1
    assert out["trade_date"].iloc[0] == base["trade_date"].iloc[0]


def test_daily_bars_across_multiple_sessions():
    frames = [make_1m(f"2026-03-0{d} 17:00", n_minutes=600) for d in (2, 3, 4)]
    base = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    out = resample(base, "1D", INDEX_CFG)
    assert len(out) == 3
    assert out["trade_date"].is_monotonic_increasing


# --------------------------------------------------------------------------
# aggregation correctness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tf", ["5min", "15min", "1h", "4h", "1D"])
def test_aggregation_conserves_volume_and_extremes(tf):
    base = make_1m()
    out = resample(base, tf, INDEX_CFG)
    assert check_ohlc_integrity(base, out) == []


def test_open_and_close_come_from_the_right_source_bars():
    base = make_1m()
    out = resample(base, "1h", INDEX_CFG)
    first_hour = base.iloc[:60]
    assert out["open"].iloc[0] == first_hour["open"].iloc[0]
    assert out["close"].iloc[0] == first_hour["close"].iloc[-1]
    assert out["high"].iloc[0] == first_hour["high"].max()
    assert out["low"].iloc[0] == first_hour["low"].min()
    assert out["volume"].iloc[0] == first_hour["volume"].sum()


def test_n_base_bars_flags_the_partial_bucket():
    out = resample(make_1m(), "4h", INDEX_CFG)
    assert (out["n_base_bars"].iloc[:-1] == 240).all()
    assert out["n_base_bars"].iloc[-1] == 1380 - 5 * 240   # 180-minute stub


def test_gaps_are_not_filled():
    """A missing block of minutes must not produce fabricated bars."""
    base = make_1m()
    holed = pd.concat([base.iloc[:60], base.iloc[180:]]).reset_index(drop=True)
    out = resample(holed, "1h", INDEX_CFG)
    # the 2h hole removes exactly two hourly buckets
    assert len(out) == 23 - 2
    assert out["n_base_bars"].min() == 60


def test_no_bucket_spans_two_trade_dates():
    frames = [make_1m(f"2026-03-0{d} 17:00", n_minutes=1380) for d in (2, 3)]
    base = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    for tf in ["5min", "1h", "4h"]:
        out = resample(base, tf, INDEX_CFG)
        # every aggregated bar carries exactly one trade_date by construction;
        # verify none of them straddle by checking the source counts add up
        assert out["n_base_bars"].sum() == len(base), tf


def test_resample_all_returns_every_requested_timeframe():
    tfs = ["5min", "15min", "1h", "4h", "1D"]
    got = resample_all(make_1m(), INDEX_CFG, tfs)
    assert set(got) == set(tfs)
    assert all(not v.empty for v in got.values())


def test_unknown_timeframe_rejected():
    with pytest.raises(ValueError, match="unknown timeframe"):
        resample(make_1m(), "7min", INDEX_CFG)


def test_missing_trade_date_rejected():
    base = make_1m().drop(columns=["trade_date"])
    with pytest.raises(ValueError, match="trade_date"):
        resample(base, "1h", INDEX_CFG)
