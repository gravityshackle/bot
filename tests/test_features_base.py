"""ATR, CLV and volume-expansion tests.

These are the highest-leverage unit tests in the repo: ATR sets the breakout
buffer, test-zone tolerance, rejection body filter, gap threshold, three-tail
tolerance, stop distance and -- through the stop -- position size. A quiet 20%
error here is a 20% sizing error on every trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from features import confirmation, risk_state
from features.schema import (
    ATR,
    CLV,
    SESSION,
    VOLUME_EXPANDED,
    VOLUME_RATIO,
    UnresolvedParameter,
    load_params,
)

CT = "America/Chicago"


def bars(n=120, start="2026-03-02 08:30", freq="5min", vol=1000):
    ts = pd.date_range(pd.Timestamp(start), periods=n, freq=freq,
                       tz=CT).tz_convert("UTC")
    base = 5000 + np.arange(n) * 0.25
    return pd.DataFrame({
        "ts": ts, "raw_symbol": "MESM6",
        "open": base, "high": base + 1.0, "low": base - 1.0, "close": base + 0.5,
        "volume": [vol] * n,
    })


def load_sym(name):
    with open(f"config/symbols/{name}.yaml") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# S13 CLV
# --------------------------------------------------------------------------

@pytest.mark.parametrize("o,h,lo,c,expected", [
    (10, 12, 8, 12, 1.0),      # closed at the high
    (10, 12, 8, 8, -1.0),      # closed at the low
    (10, 12, 8, 10, 0.0),      # closed mid-range
    (10, 12, 8, 11, 0.5),
    (10, 10, 10, 10, 0.0),     # zero-range bar must not divide by zero
])
def test_clv_values(o, h, lo, c, expected):
    df = pd.DataFrame({"open": [o], "high": [h], "low": [lo], "close": [c]})
    assert confirmation.clv(df).iloc[0] == pytest.approx(expected)


def test_clv_is_bounded():
    rng = np.random.default_rng(0)
    low = rng.uniform(90, 100, 500)
    high = low + rng.uniform(0.01, 5, 500)
    close = rng.uniform(low, high)
    open_ = rng.uniform(low, high)
    v = confirmation.clv(pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}))
    assert v.between(-1, 1).all()


# --------------------------------------------------------------------------
# S15 ATR
# --------------------------------------------------------------------------

def test_true_range_uses_prior_close():
    df = pd.DataFrame({"open": [10, 10], "high": [11, 12],
                       "low": [9, 11.5], "close": [10.5, 11.8]})
    tr = risk_state.true_range(df)
    assert tr.iloc[0] == pytest.approx(2.0)          # first bar falls back to H-L
    # second bar gaps up: H-prevC = 1.5 beats H-L = 0.5
    assert tr.iloc[1] == pytest.approx(1.5)


def test_atr_is_wilder_not_simple_mean():
    """Wilder's RMA and a simple rolling mean are different indicators, and
    every ATR-derived threshold inherits whichever one is used."""
    df = bars(60)
    df.loc[30, "high"] = df.loc[30, "high"] + 50      # one volatility shock
    tr = risk_state.true_range(df)
    wilder = risk_state.wilder_rma(tr, 14)
    simple = tr.rolling(14).mean()
    # 15 bars after the shock the simple mean has dropped it entirely;
    # Wilder still carries a decayed share of it
    assert wilder.iloc[46] != pytest.approx(simple.iloc[46])
    assert wilder.iloc[46] > simple.iloc[46]


def test_atr_seed_is_the_simple_mean_of_the_first_period():
    tr = risk_state.true_range(bars(40))
    a = risk_state.wilder_rma(tr, 14)
    assert np.isnan(a.iloc[12])
    assert a.iloc[13] == pytest.approx(tr.iloc[:14].mean())


def test_atr_is_causal():
    """Truncating future bars must not change any earlier ATR value."""
    df = bars(80)
    full = risk_state.atr(df, 14)
    part = risk_state.atr(df.iloc[:50].copy(), 14)
    pd.testing.assert_series_equal(full.iloc[:50], part, check_names=False)


def test_volatility_regime_baseline_excludes_current_bar():
    a = risk_state.atr(bars(120), 14)
    regime, baseline = risk_state.volatility_regime(a, 50, 1.3, 0.7)
    i = 100
    assert baseline.iloc[i] == pytest.approx(a.iloc[i - 50:i].mean())
    assert set(regime.unique()) <= {"high", "normal", "low", "unknown"}


def test_regime_classification_thresholds():
    a = pd.Series([10.0] * 60)
    a.iloc[55] = 14.0        # 1.4x -> high
    a.iloc[56] = 6.0         # 0.6x -> low
    regime, _ = risk_state.volatility_regime(a, 50, 1.3, 0.7)
    assert regime.iloc[55] == "high"
    assert regime.iloc[56] == "low"
    assert regime.iloc[54] == "normal"


def test_reward_risk_and_zero_risk_guard():
    assert risk_state.reward_risk(100, 95, 115, True) == pytest.approx(3.0)
    assert risk_state.reward_risk(100, 105, 85, False) == pytest.approx(3.0)
    assert np.isnan(risk_state.reward_risk(100, 100, 120, True))


# --------------------------------------------------------------------------
# S12 volume expansion
# --------------------------------------------------------------------------

def test_volume_baseline_excludes_current_bar():
    df = bars(40, vol=100)
    df.loc[39, "volume"] = 10_000                  # a huge spike at the end
    out = confirmation.apply(df, load_params("MES"), load_sym("MES"))
    # the spike must not inflate the baseline it is judged against
    assert out["volume_baseline"].iloc[39] == pytest.approx(100.0)
    assert out[VOLUME_RATIO].iloc[39] == pytest.approx(100.0)
    assert bool(out[VOLUME_EXPANDED].iloc[39])


def test_flat_volume_is_never_expansion():
    out = confirmation.apply(bars(60, vol=500), load_params("MES"), load_sym("MES"))
    assert not out[VOLUME_EXPANDED].iloc[1:].any()


def test_session_matched_baseline_separates_rth_from_eth():
    """An RTH bar must not be measured against overnight volume."""
    df = bars(288, start="2026-03-02 00:00")       # 24h of 5m bars
    sess = confirmation.classify_session(df, load_sym("MES"))
    df["volume"] = [2000 if s == "rth" else 100 for s in sess]
    out = confirmation.apply(df, load_params("MES"), load_sym("MES"))
    rth = out[out[SESSION] == "rth"].iloc[5:]
    # a normal RTH bar sits near 1.0 against an RTH baseline, not 20x
    assert rth[VOLUME_RATIO].median() == pytest.approx(1.0, abs=0.05)
    assert not rth[VOLUME_EXPANDED].any()


def test_session_labels_match_the_configured_rth_window():
    df = bars(288, start="2026-03-02 00:00")
    sess = confirmation.classify_session(df, load_sym("MES"))
    local = df["ts"].dt.tz_convert(CT)
    mins = local.dt.hour * 60 + local.dt.minute
    expected_rth = (mins >= 8 * 60 + 30) & (mins < 15 * 60 + 15)
    assert ((sess == "rth") == expected_rth).all()


def test_continuous_symbols_use_hour_of_day_baseline():
    for sym in ["MCL", "MGC", "SIL", "MET"]:
        scfg = load_sym(sym)
        assert scfg["volume_baseline"] == "hour_of_day"
        out = confirmation.apply(bars(300, start="2026-03-02 00:00"),
                                 load_params(sym), scfg)
        assert (out[SESSION] == "continuous").all()


def test_index_symbols_keep_session_matched():
    for sym in ["MES", "MNQ", "MYM"]:
        assert load_sym(sym)["volume_baseline"] == "session_matched"


# --------------------------------------------------------------------------
# params plumbing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "swings.major_depth_reference",
    "trend.neutral_policy",
    "engulfing.status",
    "channels.status",
])
def test_unresolved_parameters_raise_rather_than_defaulting(path):
    with pytest.raises(UnresolvedParameter, match="open_questions"):
        load_params("MES").get(path)


def test_resolved_parameters_load_normally():
    p = load_params("MES")
    assert p.get("atr.period") == 14
    assert p.get("targets.min_reward_risk") == 2.0
    assert p.values["_symbol"]["symbol"] == "MES"


def test_apply_pipeline_runs_on_every_symbol():
    for sym in ["MES", "MNQ", "MYM", "MCL", "MGC", "SIL", "MET"]:
        p, scfg = load_params(sym), load_sym(sym)
        out = risk_state.apply(bars(120), p)
        out = confirmation.apply(out, p, scfg)
        assert out[ATR].notna().sum() > 0
        assert out[CLV].between(-1, 1).all()
