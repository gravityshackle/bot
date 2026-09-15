"""S2 prior day/week, S3 gaps, S5 consolidation.

The boundary this suite exists for is session scope. MES/MNQ/MYM take their
daily range from RTH bars only; MCL/MGC/SIL/MET take it from the whole session.
A mistake there does not raise -- it silently draws prior-day levels from the
wrong bars, and every trigger built on those levels inherits it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from data.continuous_contract import trade_date
from features import levels
from features.schema import load_params

CT = "America/Chicago"


def load_sym(name):
    with open(f"config/symbols/{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def session_bars(days=5, start="2026-03-02", boundary="17:00",
                 rth_price=100.0, eth_price=200.0):
    """Bars across full 23h sessions, with RTH and overnight at clearly
    different price levels so scope errors are unmissable."""
    rows = []
    for d in range(days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        # session runs 17:00 previous day -> 16:00 this day, hourly
        open_ts = (day - pd.Timedelta(days=1)).replace(hour=17)
        for h in range(23):
            ts = open_ts + pd.Timedelta(hours=h)
            # On-the-hour bars inside MES RTH (08:30-15:15 CT) are 09:00
            # through 15:00 inclusive. 08:00 is before the open and 16:00 is
            # after the close -- getting these edges wrong is exactly the
            # scope bug this fixture is meant to expose, so match them exactly.
            is_rth = (9 <= ts.hour <= 15) and ts.date() == day.date()
            base = rth_price + d if is_rth else eth_price + d
            rows.append({"ts": ts, "raw_symbol": "X",
                         "open": base, "high": base + 1.0,
                         "low": base - 1.0, "close": base + 0.5,
                         "volume": 100})
    df = pd.DataFrame(rows)
    # 2026-03-08 02:00 CT does not exist (spring forward); a 12-day fixture
    # spans it.
    df["ts"] = (df["ts"].dt.tz_localize(CT, nonexistent="shift_forward",
                                        ambiguous=True)
                .dt.tz_convert("UTC"))
    df["trade_date"] = trade_date(df["ts"], CT, boundary)
    return df.sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------
# scope: the RTH / continuous split
# --------------------------------------------------------------------------

def test_index_micros_scope_to_rth_only():
    df = session_bars()
    mask = levels.scope_mask(df, load_sym("MES"))
    assert mask.any() and not mask.all()
    # every in-scope bar must fall inside 08:30-15:15 CT
    local = df.loc[mask, "ts"].dt.tz_convert(CT)
    mins = local.dt.hour * 60 + local.dt.minute
    assert mins.between(8 * 60 + 30, 15 * 60 + 15 - 1).all()


@pytest.mark.parametrize("sym", ["MCL", "MGC", "SIL", "MET"])
def test_continuous_symbols_scope_to_everything(sym):
    df = session_bars()
    assert levels.scope_mask(df, load_sym(sym)).all()


def test_prior_day_levels_differ_between_the_two_scopes():
    """The same bars must give different prior-day levels for MES vs MCL --
    this is the whole point of the scope split."""
    df = session_bars(rth_price=100.0, eth_price=200.0)
    mes = levels.attach_prior_levels(df, load_sym("MES"))
    mcl = levels.attach_prior_levels(df, load_sym("MCL"))
    day = sorted(df["trade_date"].unique())[2]
    mes_high = mes.loc[mes["trade_date"] == day, levels.PRIOR_DAY_HIGH].iloc[0]
    mcl_high = mcl.loc[mcl["trade_date"] == day, levels.PRIOR_DAY_HIGH].iloc[0]
    assert mes_high < 150, "MES prior-day high must come from RTH bars (~100)"
    assert mcl_high > 150, "MCL prior-day high must include overnight (~200)"


# --------------------------------------------------------------------------
# S2 correctness and causality
# --------------------------------------------------------------------------

def test_prior_day_high_equals_the_previous_days_scoped_high():
    df = session_bars(days=4)
    scfg = load_sym("MCL")                      # full-session scope, simplest
    out = levels.attach_prior_levels(df, scfg)
    dates = sorted(df["trade_date"].unique())
    for prev, cur in zip(dates, dates[1:]):
        expected_hi = df.loc[df["trade_date"] == prev, "high"].max()
        expected_lo = df.loc[df["trade_date"] == prev, "low"].min()
        got = out[out["trade_date"] == cur].iloc[0]
        assert got[levels.PRIOR_DAY_HIGH] == pytest.approx(expected_hi)
        assert got[levels.PRIOR_DAY_LOW] == pytest.approx(expected_lo)


def test_first_day_has_no_prior_levels():
    out = levels.attach_prior_levels(session_bars(), load_sym("MCL"))
    first = sorted(out["trade_date"].unique())[0]
    row = out[out["trade_date"] == first].iloc[0]
    assert pd.isna(row[levels.PRIOR_DAY_HIGH])
    assert pd.isna(row[levels.PRIOR_DAY_LOW])


def test_prior_day_level_never_uses_the_current_day():
    """A bar must not see its own session's range."""
    df = session_bars(days=4)
    out = levels.attach_prior_levels(df, load_sym("MCL"))
    for d in sorted(df["trade_date"].unique())[1:]:
        today_hi = df.loc[df["trade_date"] == d, "high"].max()
        got = out.loc[out["trade_date"] == d, levels.PRIOR_DAY_HIGH].iloc[0]
        assert got != pytest.approx(today_hi)


def test_prior_day_level_is_constant_within_a_session():
    out = levels.attach_prior_levels(session_bars(days=4), load_sym("MCL"))
    for _, g in out.groupby("trade_date"):
        assert g[levels.PRIOR_DAY_HIGH].nunique(dropna=False) == 1


def test_weekly_levels_roll_at_week_start():
    df = session_bars(days=12, start="2026-03-02")   # spans two ISO weeks
    out = levels.attach_prior_levels(df, load_sym("MCL"))
    wk = levels.week_key(out["trade_date"])
    assert wk.nunique() >= 2
    first_week = sorted(wk.unique())[0]
    # week 1 has no prior week
    assert out.loc[wk == first_week, levels.PRIOR_WEEK_HIGH].isna().all()
    # week 2 sees exactly week 1's extremes
    second = sorted(wk.unique())[1]
    expected = df.loc[wk == first_week, "high"].max()
    assert out.loc[wk == second, levels.PRIOR_WEEK_HIGH].dropna().iloc[0] \
        == pytest.approx(expected)


def test_weekly_level_is_constant_within_a_week():
    df = session_bars(days=12)
    out = levels.attach_prior_levels(df, load_sym("MCL"))
    wk = levels.week_key(out["trade_date"])
    for _, idx in out.groupby(wk).groups.items():
        assert out.loc[idx, levels.PRIOR_WEEK_HIGH].nunique(dropna=False) == 1


# --------------------------------------------------------------------------
# S3 gaps
# --------------------------------------------------------------------------

def gap_frame(opens_closes, boundary="17:00"):
    """One bar per session, so open/close per day are fully controlled."""
    rows = []
    for d, (o, c) in enumerate(opens_closes):
        day = pd.Timestamp("2026-03-02") + pd.Timedelta(days=d)
        for h, px in ((9, o), (14, c)):
            ts = day.replace(hour=h)
            rows.append({"ts": ts, "raw_symbol": "X", "open": px,
                         "high": max(o, c) + 1, "low": min(o, c) - 1,
                         "close": px, "volume": 10})
    df = pd.DataFrame(rows)
    df["ts"] = df["ts"].dt.tz_localize(CT).dt.tz_convert("UTC")
    df["trade_date"] = trade_date(df["ts"], CT, boundary)
    return df.sort_values("ts").reset_index(drop=True)


def _atr_by_date(df, value):
    dates = sorted(df["trade_date"].unique())
    return pd.Series([value] * len(dates),
                     index=pd.Index(dates, name="trade_date"))


def test_gap_detected_only_above_threshold():
    df = gap_frame([(100, 100), (110, 110), (111, 111)])
    p = load_params("MCL")
    # ATR 10 -> threshold 1.5; the 10-pt gap qualifies, the 1-pt one does not
    g = levels.find_gaps(df, load_sym("MCL"), _atr_by_date(df, 10.0), p)
    assert len(g) == 1
    assert g.iloc[0]["direction"] == "up"
    assert g.iloc[0]["gap"] == pytest.approx(10.0)


def test_gap_zone_spans_prior_close_to_open():
    df = gap_frame([(100, 100), (110, 110)])
    g = levels.find_gaps(df, load_sym("MCL"), _atr_by_date(df, 10.0),
                         load_params("MCL")).iloc[0]
    assert g["zone_low"] == pytest.approx(100.0)
    assert g["zone_high"] == pytest.approx(110.0)


def test_up_gap_fills_only_when_price_reaches_the_far_edge():
    # day 3 dips to 105 (inside the zone, not a fill); day 4 reaches 100
    df = gap_frame([(100, 100), (110, 110), (106, 106), (99, 99)])
    g = levels.find_gaps(df, load_sym("MCL"), _atr_by_date(df, 10.0),
                         load_params("MCL")).iloc[0]
    dates = sorted(df["trade_date"].unique())
    assert g["filled_date"] == dates[3]


def test_unfilled_gap_has_no_fill_date():
    df = gap_frame([(100, 100), (110, 110), (112, 112), (113, 113)])
    g = levels.find_gaps(df, load_sym("MCL"), _atr_by_date(df, 10.0),
                         load_params("MCL")).iloc[0]
    assert pd.isna(g["filled_date"])


def test_gap_threshold_uses_the_prior_days_atr_not_todays():
    df = gap_frame([(100, 100), (110, 110)])
    dates = sorted(df["trade_date"].unique())
    # tiny ATR yesterday, huge ATR today: the gap must still be detected,
    # because only yesterday's ATR may inform today's threshold
    atr = pd.Series([1.0, 1000.0], index=pd.Index(dates, name="trade_date"))
    g = levels.find_gaps(df, load_sym("MCL"), atr, load_params("MCL"))
    assert len(g) == 1


def test_gaps_require_trade_date_indexed_atr():
    df = gap_frame([(100, 100), (110, 110)])
    bad = pd.Series([10.0, 10.0])          # positional index, not trade_date
    with pytest.raises(ValueError, match="indexed by trade_date"):
        levels.find_gaps(df, load_sym("MCL"), bad, load_params("MCL"))


# --------------------------------------------------------------------------
# S5 consolidation
# --------------------------------------------------------------------------

def test_narrow_span_flags_a_range():
    n = 60
    df = pd.DataFrame({
        "high": [100.5] * n, "low": [99.5] * n,
        "open": [100.0] * n, "close": [100.0] * n,
    })
    atr = pd.Series([5.0] * n)            # span 1.0 << 1.5 * 5.0
    atr_mean = pd.Series([5.0] * n)
    out = levels.consolidation(df, atr, atr_mean, load_params("MES"))
    assert out[levels.IN_RANGE].iloc[20:].all()


def test_wide_span_and_normal_vol_is_not_a_range():
    n = 60
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 3, n))
    df = pd.DataFrame({"high": close + 3, "low": close - 3,
                       "open": close, "close": close})
    atr = pd.Series([1.0] * n)            # span >> 1.5 * ATR, ATR == mean
    atr_mean = pd.Series([1.0] * n)
    out = levels.consolidation(df, atr, atr_mean, load_params("MES"))
    assert not out[levels.IN_RANGE].iloc[20:].all()


def test_volatility_compression_alone_flags_a_range():
    n = 60
    df = pd.DataFrame({"high": np.arange(n) + 10.0, "low": np.arange(n) * 1.0,
                       "open": np.arange(n) * 1.0, "close": np.arange(n) * 1.0})
    atr = pd.Series([1.0] * n)            # 1.0 < 0.7 * 10.0 -> compressed
    atr_mean = pd.Series([10.0] * n)
    out = levels.consolidation(df, atr, atr_mean, load_params("MES"))
    assert out[levels.IN_RANGE].all()


def test_range_boundaries_only_defined_inside_the_regime():
    n = 60
    df = pd.DataFrame({"high": [100.5] * n, "low": [99.5] * n,
                       "open": [100.0] * n, "close": [100.0] * n})
    atr = pd.Series([0.01] * n)           # span 1.0 > 1.5*0.01, no compression
    atr_mean = pd.Series([0.01] * n)
    out = levels.consolidation(df, atr, atr_mean, load_params("MES"))
    assert out.loc[~out[levels.IN_RANGE], levels.RANGE_HIGH].isna().all()
