"""S1 swing detection tests.

The two properties worth real scrutiny are repainting and lookahead. A pivot
that is treated as known before it has survived its N bars makes a backtest
use information that did not exist yet, and every level, target and trailing
stop derived from it inherits the error silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import risk_state, structure
from features.schema import load_params

CT = "America/Chicago"


def frame(highs, lows=None, start="2026-03-02 09:00", freq="5min"):
    n = len(highs)
    lows = lows if lows is not None else [h - 2.0 for h in highs]
    ts = pd.date_range(pd.Timestamp(start), periods=n, freq=freq,
                       tz=CT).tz_convert("UTC")
    return pd.DataFrame({
        "ts": ts, "raw_symbol": "MESM6",
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": list(map(float, highs)), "low": list(map(float, lows)),
        "close": [(h + l) / 2 for h, l in zip(highs, lows)],
        "volume": [100] * n,
    })


# --------------------------------------------------------------------------
# fractal geometry
# --------------------------------------------------------------------------

def test_finds_a_single_obvious_swing_high():
    df = frame([10, 11, 15, 11, 10])
    p = structure.find_pivots(df, n=2)
    highs = p[p["kind"] == "high"]
    assert len(highs) == 1
    assert highs.iloc[0]["idx"] == 2
    assert highs.iloc[0]["price"] == 15.0


def test_finds_a_single_obvious_swing_low():
    df = frame([20, 19, 18, 19, 20], lows=[15, 14, 10, 14, 15])
    lows = structure.find_pivots(df, n=2)
    lows = lows[lows["kind"] == "low"]
    assert len(lows) == 1
    assert lows.iloc[0]["idx"] == 2
    assert lows.iloc[0]["price"] == 10.0


def test_strict_inequality_means_a_flat_plateau_is_not_a_pivot():
    """Spec S1 uses > not >=, so equal highs have no single turning point."""
    df = frame([10, 11, 15, 15, 11, 10])
    highs = structure.find_pivots(df, n=2)
    assert highs[highs["kind"] == "high"].empty


def test_n_controls_sensitivity():
    # a small bump that survives 1 bar either side but not 3
    highs = [10, 11, 12, 11, 10, 9, 8, 7, 6]
    assert len(structure.find_pivots(frame(highs), n=1).query("kind=='high'")) >= 1
    assert structure.find_pivots(frame(highs), n=3).query("kind=='high'").empty


def test_edges_cannot_produce_pivots():
    """The first and last N bars lack a full window on one side."""
    df = frame([20, 10, 10, 10, 20])
    p = structure.find_pivots(df, n=2)
    assert (p["idx"] >= 2).all() and (p["idx"] <= len(df) - 3).all()


def test_too_short_a_frame_returns_empty():
    assert structure.find_pivots(frame([1, 2, 3]), n=2).empty


def test_invalid_n_rejected():
    with pytest.raises(ValueError, match="pivot N"):
        structure.find_pivots(frame([1, 2, 3, 4, 5]), n=0)


# --------------------------------------------------------------------------
# no repainting
# --------------------------------------------------------------------------

def test_confirmation_lags_the_pivot_by_exactly_n_bars():
    df = frame([10, 11, 15, 11, 10, 9, 8])
    p = structure.find_pivots(df, n=2)
    row = p[p["kind"] == "high"].iloc[0]
    assert row["confirmed_idx"] == row["idx"] + 2
    assert row["confirmed_ts"] == df["ts"].iloc[row["idx"] + 2]


def test_pivot_is_invisible_before_confirmation():
    df = frame([10, 11, 15, 11, 10, 9, 8])
    p = structure.find_pivots(df, n=2)
    pivot_bar = int(p[p["kind"] == "high"].iloc[0]["idx"])
    # at the pivot bar itself, and the bar after, nothing is known yet
    assert structure.last_confirmed_swings(p, pivot_bar).empty
    assert structure.last_confirmed_swings(p, pivot_bar + 1).empty
    assert not structure.last_confirmed_swings(p, pivot_bar + 2).empty


def test_detection_is_causal_truncating_the_future_changes_nothing():
    """Pivots found on a prefix must match those found on the full series."""
    rng = np.random.default_rng(7)
    highs = 100 + np.cumsum(rng.normal(0, 1, 300))
    df = frame(highs)
    full = structure.find_pivots(df, n=2)
    cut = 200
    part = structure.find_pivots(df.iloc[:cut].copy(), n=2)
    # every pivot confirmed by bar `cut - 1` in the full series must appear
    # identically in the truncated one
    a = full[full["confirmed_idx"] < cut - 2][["idx", "kind", "price"]]
    b = part[part["confirmed_idx"] < cut - 2][["idx", "kind", "price"]]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_pivots_are_ordered_by_when_they_became_known():
    rng = np.random.default_rng(3)
    df = frame(100 + np.cumsum(rng.normal(0, 1, 200)))
    p = structure.find_pivots(df, n=3)
    assert p["confirmed_idx"].is_monotonic_increasing


# --------------------------------------------------------------------------
# major / minor -- backward-looking depth
# --------------------------------------------------------------------------

def _classified(pivots, atr_value, multiple=1.0):
    atr = pd.Series([atr_value] * len(pivots))
    return structure.classify_swings(pivots, atr, multiple)


def test_depth_measures_from_the_prior_opposite_swing():
    # low at 10, then high at 30 -> depth of the high is 20, not anything
    # involving a later swing
    highs = [12, 13, 14, 13, 12, 13, 30, 13, 12, 11, 10]
    lows = [11, 11, 11, 11, 10, 11, 20, 11, 11, 11, 9]
    p = structure.find_pivots(frame(highs, lows), n=2)
    c = _classified(p, atr_value=5.0)
    high_rows = c[c["kind"] == "high"].dropna(subset=["depth"])
    assert not high_rows.empty
    r = high_rows.iloc[0]
    prior_low = c[(c["kind"] == "low") & (c["idx"] == r["prior_opposite_idx"])].iloc[0]
    assert r["depth"] == pytest.approx(r["price"] - prior_low["price"])


def test_majority_threshold_is_atr_scaled():
    highs = [12, 13, 14, 13, 12, 13, 30, 13, 12, 11, 10]
    lows = [11, 11, 11, 11, 10, 11, 20, 11, 11, 11, 9]
    p = structure.find_pivots(frame(highs, lows), n=2)
    r = _classified(p, atr_value=5.0).dropna(subset=["depth"]).iloc[0]
    depth = abs(r["depth"])
    # major when ATR is small relative to depth, minor when ATR is large
    small = _classified(p, atr_value=depth / 2).dropna(subset=["depth"]).iloc[0]
    large = _classified(p, atr_value=depth * 2).dropna(subset=["depth"]).iloc[0]
    assert bool(small["is_major"]) is True
    assert bool(large["is_major"]) is False


def test_first_pivot_has_no_prior_opposite_and_is_unclassified():
    df = frame([10, 11, 15, 11, 10, 9, 8, 9, 10])
    c = _classified(structure.find_pivots(df, n=2), atr_value=1.0)
    assert pd.isna(c.iloc[0]["is_major"])
    assert c.iloc[0]["prior_opposite_idx"] == -1


def test_unknown_atr_leaves_majority_unclassified_not_false():
    df = frame([12, 13, 14, 13, 12, 13, 30, 13, 12, 11, 10])
    p = structure.find_pivots(df, n=2)
    c = structure.classify_swings(p, pd.Series([np.nan] * len(p)), 1.0)
    assert c["is_major"].isna().all()


def test_major_only_filter_excludes_unclassified():
    df = frame([10, 11, 15, 11, 10, 9, 8, 9, 10])
    c = _classified(structure.find_pivots(df, n=2), atr_value=1.0)
    out = structure.last_confirmed_swings(c, as_of_idx=10**6, major_only=True)
    assert not out["is_major"].isna().any()


# --------------------------------------------------------------------------
# HTF ATR alignment
# --------------------------------------------------------------------------

def test_htf_atr_alignment_is_causal():
    """An LTF bar may only see HTF bars that have already CLOSED."""
    ltf = frame(100 + np.arange(240) * 0.1, freq="5min")
    htf = frame(100 + np.arange(20) * 1.0, freq="1h")
    htf_atr = pd.Series(np.arange(20, dtype="float64"))   # value == bar index

    aligned = structure.htf_atr_at(ltf, htf, htf_atr, "1h")

    # the first LTF bar sits inside HTF bar 0, which has not closed -> no value
    assert pd.isna(aligned.iloc[0])
    # an LTF bar just after HTF bar 0 closes sees value 0, never 1
    first_known = aligned.dropna()
    assert first_known.iloc[0] == 0.0
    # and the alignment never runs ahead of the clock
    assert aligned.dropna().is_monotonic_increasing


def test_swings_requires_an_htf_atr_series():
    df = frame(100 + np.arange(60) * 0.1)
    with pytest.raises(ValueError, match="ATR\\(14, HTF\\)"):
        structure.swings(df, load_params("MES"))


# --------------------------------------------------------------------------
# end to end on synthetic but realistic input
# --------------------------------------------------------------------------

def test_full_pipeline_produces_both_kinds_and_some_majors():
    rng = np.random.default_rng(11)
    n = 1200
    close = 5000 + np.cumsum(rng.normal(0, 2.0, n))
    ltf = frame(close + 1.0, close - 1.0, freq="5min")
    htf = ltf.iloc[::12].reset_index(drop=True)          # 1h from 5m
    htf_atr = risk_state.atr(htf, 14)

    p = load_params("MES")
    sw = structure.swings(ltf, p, htf=htf, htf_atr=htf_atr)
    assert set(sw["kind"]) == {"high", "low"}
    assert sw["is_major"].notna().any()
    assert (sw["confirmed_idx"] > sw["idx"]).all()
