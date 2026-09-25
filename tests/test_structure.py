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


@pytest.mark.parametrize("ltf_unit,htf_unit", [("us", "ns"), ("ns", "us")])
def test_alignment_works_across_timestamp_resolutions(ltf_unit, htf_unit):
    """Regression, found on real data. The intraday resample path yields
    microsecond timestamps while the daily frame keeps the base bars'
    nanoseconds, and merge_asof refuses mismatched keys. So aligning ANY daily
    value (S18's daily exhaustion, for one) onto the entry frame raised."""
    ltf = frame(100 + np.arange(240) * 0.1, freq="5min")
    htf = frame(100 + np.arange(20) * 1.0, freq="1h")
    ltf["ts"] = ltf["ts"].astype(f"datetime64[{ltf_unit}, UTC]")
    htf["ts"] = htf["ts"].astype(f"datetime64[{htf_unit}, UTC]")
    vals = pd.Series(np.arange(20, dtype="float64"))
    aligned = structure.align_htf(ltf, htf, vals, "1h")
    same = structure.align_htf(ltf.assign(ts=ltf["ts"].astype("datetime64[ns, UTC]")),
                               htf.assign(ts=htf["ts"].astype("datetime64[ns, UTC]")),
                               vals, "1h")
    assert aligned.equals(same)
    assert pd.isna(aligned.iloc[0]) and aligned.dropna().iloc[0] == 0.0


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


# --------------------------------------------------------------------------
# liveness: a major swing dies on the first close beyond it BY THE S4 BUFFER
# --------------------------------------------------------------------------

def _swings(prices_kinds_idx, major=True):
    """(price, kind, idx) rows as confirmed swings, confirmed at idx+2."""
    return pd.DataFrame([{
        "idx": i, "ts": pd.NaT, "price": pr, "kind": k,
        "confirmed_idx": i + 2, "confirmed_ts": pd.NaT, "depth": 5.0,
        "is_major": major, "prior_opposite_idx": -1}
        for pr, k, i in prices_kinds_idx])


def _closes(values):
    return pd.Series(values, dtype="float64")


def _buf(n, value=0.5):
    return pd.Series([value] * n, dtype="float64")


def _deaths(swings, closes, buf=0.5, major=True):
    c = _closes(closes)
    b = buf if isinstance(buf, pd.Series) else _buf(len(c), buf)
    return structure.mark_swing_deaths(_swings(swings, major), c,
                                       major_buffer=b,
                                       minor_buffer=_buf(len(c), 0.0))


def test_a_swing_high_dies_on_the_first_close_beyond_the_buffer():
    piv = _deaths([(15.0, "high", 2)], [10, 11, 14, 12, 13, 15.6, 14, 13])
    assert piv["dead_idx"].iloc[0] == 5


def test_a_close_inside_the_buffer_does_not_kill_a_swing():
    """Regression, measured on real data. A close beyond the level but inside
    S4's buffer is not a breakout, yet under a bare-close rule it killed the
    swing. The real breakout that followed was then of a dead level, and gate
    2 failed it. That was 19% of swing deaths and 14% of S8/S9 on swing levels."""
    piv = _deaths([(15.0, "high", 2)], [10, 11, 14, 12, 15.3, 14, 15.6, 13])
    assert piv["dead_idx"].iloc[0] == 6        # not bar 4's 15.3


def test_closing_exactly_at_the_buffer_edge_does_not_kill_it():
    """Same strict inequality as S4: a breakout needs close > level + buffer."""
    piv = _deaths([(15.0, "high", 2)], [10, 11, 14, 12, 15.5, 13])
    assert pd.isna(piv["dead_idx"].iloc[0])


def test_the_buffer_is_read_on_the_closing_bar():
    """The buffer is ATR-scaled, so it moves bar to bar."""
    buf = pd.Series([0.5, 0.5, 0.5, 0.5, 1.0, 0.2], dtype="float64")
    piv = _deaths([(15.0, "high", 2)], [10, 11, 14, 12, 15.6, 15.3], buf)
    assert piv["dead_idx"].iloc[0] == 5        # 15.6 < 16.0 at bar 4; 15.3 > 15.2


def test_a_wick_through_does_not_kill_a_swing():
    """Same close-not-wick distinction as S4: only closes are read at all."""
    df = frame([10, 11, 15, 12, 17, 12, 11])        # bar 4's HIGH clears 15.5
    df["close"] = [9, 10, 14, 11, 14.5, 11, 10]      # but it closes below
    piv = structure.mark_swing_deaths(_swings([(15.0, "high", 2)]),
                                      df["close"], major_buffer=_buf(len(df)),
                                      minor_buffer=_buf(len(df), 0.0))
    assert pd.isna(piv["dead_idx"].iloc[0])


def test_a_swing_low_dies_on_the_first_close_beyond_the_buffer():
    piv = _deaths([(10.0, "low", 2)], [12, 11, 10.5, 11, 9.7, 9.4, 11])
    assert piv["dead_idx"].iloc[0] == 5        # 9.7 is inside the buffer


def test_live_swings_are_confirmed_and_not_yet_dead():
    """Live from confirmation until the bar that closes beyond, and dead ON
    that bar: it has closed, so the break is known at that bar's decision."""
    piv = _deaths([(15.0, "high", 2)], [10, 11, 14, 12, 13, 15.6, 14])
    live = lambda i: len(structure.live_major_swings(piv, i))
    assert [live(i) for i in range(7)] == [0, 0, 0, 0, 1, 0, 0]


def test_liveness_ignores_closes_before_the_pivot():
    """A close above 15 BEFORE the swing formed says nothing about it."""
    piv = _deaths([(15.0, "high", 3)], [16, 11, 12, 14, 12, 13, 14])
    assert pd.isna(piv["dead_idx"].iloc[0])


def test_a_minor_swing_dies_on_a_bare_close_inside_the_buffer():
    """Spec S1: minor swings die on a bare close, because the only trigger
    defined against them (S11 momentum) is a bare close. The same closes
    leave a MAJOR swing alive until the buffer clears."""
    closes = [10, 11, 14, 12, 15.3, 14, 15.6, 13]
    assert _deaths([(15.0, "high", 2)], closes, major=False)["dead_idx"].iloc[0] == 4
    assert _deaths([(15.0, "high", 2)], closes, major=True)["dead_idx"].iloc[0] == 6


def test_an_unclassified_swing_gets_no_death():
    """is_major NA is neither major nor minor, so it is a level under neither
    rule and no threshold applies. It is excluded downstream either way."""
    piv = _deaths([(15.0, "high", 2)], [10, 11, 14, 12, 16, 17], major=pd.NA)
    assert pd.isna(piv["dead_idx"].iloc[0])


def test_mark_swing_deaths_requires_both_buffers_by_name():
    """Both thresholds are required, so neither can silently default."""
    with pytest.raises(TypeError):
        structure.mark_swing_deaths(_swings([(15.0, "high", 2)]),
                                    _closes([1, 2, 3]), _buf(3))


def test_mark_swing_deaths_requires_the_buffers_to_match_the_closes():
    for major, minor in ((_buf(2), _buf(3)), (_buf(3), _buf(2))):
        with pytest.raises(ValueError, match="buffer"):
            structure.mark_swing_deaths(_swings([(15.0, "high", 2)]),
                                        _closes([1, 2, 3]),
                                        major_buffer=major, minor_buffer=minor)


def test_live_major_swings_refuses_unmarked_pivots():
    """Without deaths marked, 'live' would silently mean 'every swing ever'
    -- the ~470-levels-per-bar failure this rule exists to prevent."""
    with pytest.raises(ValueError, match="mark_swing_deaths"):
        structure.live_major_swings(_swings([(15.0, "high", 2)]), 10)


def test_mark_swing_deaths_on_no_swings():
    empty = _swings([]).reindex(columns=structure.PIVOT_COLUMNS
                                + ["depth", "is_major", "prior_opposite_idx"])
    out = structure.mark_swing_deaths(empty, _closes([1, 2, 3]),
                                      major_buffer=_buf(3),
                                      minor_buffer=_buf(3, 0.0))
    assert "dead_idx" in out.columns and out.empty
