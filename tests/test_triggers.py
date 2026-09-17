"""Trigger tests: S4, S7, S8, S9, S10, S11, S14, S19, S20.

Hand-built candles throughout, so each assertion pins one geometric rule rather
than whatever a random walk happened to produce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import confirmation, triggers
from features.schema import load_params

CT = "America/Chicago"
P = load_params("MES")          # tick 0.25


def mk(rows, start="2026-03-02 09:00", freq="5min"):
    """rows: list of (open, high, low, close) or (o,h,l,c,volume)."""
    n = len(rows)
    ts = pd.date_range(pd.Timestamp(start), periods=n, freq=freq,
                       tz=CT).tz_convert("UTC")
    df = pd.DataFrame({
        "ts": ts, "raw_symbol": "MESM6",
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [r[4] if len(r) > 4 else 100 for r in rows],
    })
    return df


def feats_of(df):
    return confirmation.candle_anatomy(df).assign(
        **{"clv": confirmation.clv(df)})


def flat_atr(df, value=4.0):
    return pd.Series([value] * len(df), index=df.index)


# --------------------------------------------------------------------------
# S4 breakout: close beyond vs wick through
# --------------------------------------------------------------------------

def test_close_beyond_buffer_is_a_breakout():
    # level 100, ATR 4 -> buffer = max(0.5, 0.4) = 0.5
    df = mk([(99, 101.0, 98, 100.8)])
    b = triggers.breakouts(df, 100.0, flat_atr(df), P)
    assert bool(b["up_close"].iloc[0])
    assert not bool(b["wick_up_only"].iloc[0])


def test_close_inside_the_buffer_is_not_a_breakout():
    df = mk([(99, 101.0, 98, 100.3)])       # above level but inside buffer
    b = triggers.breakouts(df, 100.0, flat_atr(df), P)
    assert not bool(b["up_close"].iloc[0])


def test_wick_through_without_a_close_is_explicitly_not_a_breakout():
    """S4 is emphatic about this, and S8 depends on the distinction."""
    df = mk([(99, 103.0, 98, 99.5)])
    b = triggers.breakouts(df, 100.0, flat_atr(df), P)
    assert not bool(b["up_close"].iloc[0])
    assert bool(b["wick_up_only"].iloc[0])


def test_buffer_floors_at_two_ticks_when_atr_is_tiny():
    df = mk([(99, 101, 98, 100.3)])
    b = triggers.breakouts(df, 100.0, pd.Series([0.0]), P)
    assert b["buffer"].iloc[0] == pytest.approx(2 * 0.25)


def test_breakdown_is_symmetric():
    df = mk([(101, 102, 98, 99.2)])
    b = triggers.breakouts(df, 100.0, flat_atr(df), P)
    assert bool(b["down_close"].iloc[0])


# --------------------------------------------------------------------------
# S6 test zone
# --------------------------------------------------------------------------

def test_test_zone_catches_near_misses_and_straddles():
    df = mk([(100, 100.4, 99.6, 100.0),    # straddles the level
             (103, 103.5, 102.9, 103.2),   # far away
             (100.5, 100.7, 100.45, 100.6)])  # within 0.6 tolerance
    z = triggers.in_test_zone(df, 100.0, flat_atr(df), P)   # tol = 0.6
    assert bool(z.iloc[0]) and not bool(z.iloc[1]) and bool(z.iloc[2])


# --------------------------------------------------------------------------
# S7 rejection
# --------------------------------------------------------------------------

def test_bullish_rejection_requires_all_three_conditions():
    # body 1.0, lower wick 3.0, CLV high, ATR 4 -> min body 1.0
    df = mk([(100.0, 101.2, 97.0, 101.0)])
    r = triggers.rejection(df, feats_of(df), flat_atr(df), P)
    assert r.iloc[0] == triggers.LONG


def test_doji_is_rejected_by_the_body_filter():
    """Tiny body with a huge wick still fails S7's min body size."""
    df = mk([(100.0, 100.05, 97.0, 100.02)])
    r = triggers.rejection(df, feats_of(df), flat_atr(df), P)
    assert r.iloc[0] is None


def test_short_wick_fails_the_ratio():
    df = mk([(100.0, 103.0, 99.5, 102.9)])   # big body, small lower wick
    r = triggers.rejection(df, feats_of(df), flat_atr(df), P)
    assert r.iloc[0] is None


def test_bearish_rejection_mirrors():
    df = mk([(101.0, 105.0, 100.8, 101.0 - 1.0)])
    df.loc[0, "close"] = 100.9
    df.loc[0, "open"] = 102.0
    r = triggers.rejection(df, feats_of(df), flat_atr(df), P)
    assert r.iloc[0] == triggers.SHORT


# --------------------------------------------------------------------------
# S20 engulfing
# --------------------------------------------------------------------------

def test_bullish_engulfing_body_to_body():
    df = mk([(102.0, 102.5, 100.8, 101.0),      # red body 102 -> 101
             (100.9, 103.5, 100.5, 103.0)])     # green body 100.9 -> 103
    e = triggers.engulfing(df, feats_of(df), flat_atr(df), P)
    assert e.iloc[1] == triggers.LONG


def test_engulfing_ignores_wicks_not_bodies():
    """Prior bar has a huge wick the current bar does not cover; body-to-body
    still qualifies, because S20 is explicitly body-to-body."""
    df = mk([(102.0, 120.0, 80.0, 101.0),
             (100.9, 103.5, 100.5, 103.0)])
    assert triggers.engulfing(df, feats_of(df), flat_atr(df), P).iloc[1] == triggers.LONG


def test_marginal_engulf_fails_the_strength_filter():
    # prior body 1.0, current body 1.05 -> under the 1.3x requirement
    df = mk([(102.0, 102.5, 100.8, 101.0),
             (100.95, 102.4, 100.9, 102.0)])
    assert triggers.engulfing(df, feats_of(df), flat_atr(df), P).iloc[1] is None


def test_engulfing_of_a_doji_fails_the_absolute_floor():
    """The relative multiplier alone has no floor.

    Regression: an ordinary bar trivially clears 1.3x a one-tick body, so any
    bar following a doji scored as an engulfing. On real data that degenerate
    case, not genuine conviction, was the majority of all firings. ATR is 4
    here, so the prior body must reach 0.10 x 4 = 0.4.
    """
    doji = (102.0, 102.6, 101.9, 101.75)        # body 0.25, under the floor
    df = mk([doji, (101.7, 104.0, 101.6, 103.5)])
    assert triggers.engulfing(df, feats_of(df), flat_atr(df), P).iloc[1] is None

    # identical geometry, but a prior body that clears the floor: fires
    real = (102.0, 102.6, 101.3, 101.4)         # body 0.6 >= 0.4
    df2 = mk([real, (101.3, 104.0, 101.2, 103.5)])
    assert triggers.engulfing(df2, feats_of(df2), flat_atr(df2),
                              P).iloc[1] == triggers.LONG


def test_engulfing_needs_both_strength_conditions_not_either():
    """A big prior body does not excuse a weak multiple, and vice versa."""
    # prior body 2.0 clears the floor, current body 2.1 misses 1.3x
    df = mk([(102.0, 102.5, 99.8, 100.0), (99.9, 102.3, 99.7, 102.0)])
    assert triggers.engulfing(df, feats_of(df), flat_atr(df), P).iloc[1] is None


def test_same_colour_bars_never_engulf():
    df = mk([(100.0, 101.0, 99.9, 100.8),
             (99.5, 103.0, 99.4, 102.5)])      # both green
    assert triggers.engulfing(df, feats_of(df), flat_atr(df), P).iloc[1] is None


# --------------------------------------------------------------------------
# S19 three tail
# --------------------------------------------------------------------------

def _tail_bar(price_extreme, side="lower", base=100.0):
    if side == "lower":
        return (base, base + 0.2, price_extreme, base + 0.1)
    return (base, price_extreme, base - 0.2, base - 0.1)


def test_three_aligned_lower_tails_fire_long():
    rows = [(100, 100.3, 99.8, 100.1)] * 2
    rows += [_tail_bar(97.00), (100, 100.3, 99.9, 100.1),
             _tail_bar(97.05), _tail_bar(96.98)]
    df = mk(rows)
    out = triggers.three_tail(df, feats_of(df), flat_atr(df), P)   # tol 0.4
    assert not out.empty
    assert out.iloc[-1]["direction"] == triggers.LONG
    assert out.iloc[-1]["meta"]["count"] >= 3


def test_unaligned_tails_do_not_cluster():
    """Three wicks at unrelated prices are not a level."""
    rows = [(100, 100.3, 99.8, 100.1)]
    rows += [_tail_bar(97.0), _tail_bar(94.0), _tail_bar(91.0)]
    df = mk(rows)
    out = triggers.three_tail(df, feats_of(df), flat_atr(df), P)
    assert out.empty


def test_two_tails_are_not_enough_by_default():
    rows = [(100, 100.3, 99.8, 100.1)]
    rows += [_tail_bar(97.0), (100, 100.3, 99.9, 100.1), _tail_bar(97.02)]
    df = mk(rows)
    assert triggers.three_tail(df, feats_of(df), flat_atr(df), P).empty


def test_upper_tails_fire_short():
    # needs at least three_tail.lookback_bars (6) rows or no window exists
    rows = [(100, 100.3, 99.8, 100.1)] * 3
    rows += [_tail_bar(103.0, "upper"), _tail_bar(103.05, "upper"),
             _tail_bar(102.97, "upper")]
    df = mk(rows)
    out = triggers.three_tail(df, feats_of(df), flat_atr(df), P)
    assert not out.empty and out.iloc[-1]["direction"] == triggers.SHORT


def test_large_body_disqualifies_a_tail_bar():
    """A trend candle with a small wick is not a rejection wick."""
    rows = [(100, 106.0, 97.0, 105.5)] * 4      # body 5.5 > 0.5 * ATR(4)
    df = mk(rows)
    t = triggers.tail_bars(df, feats_of(df), flat_atr(df), P)
    assert not t["lower"].any() and not t["upper"].any()


# --------------------------------------------------------------------------
# S8 failed breakout / S10 range reclaim
# --------------------------------------------------------------------------

def test_failed_breakout_above_resistance_is_a_short():
    df = mk([(99, 99.5, 98.5, 99.0),
             (100, 101.5, 99.8, 101.0),     # closes above 100 + buffer
             (101, 101.2, 99.0, 99.4)])     # closes back below
    out = triggers.failed_breakouts(df, 100.0, flat_atr(df), P)
    assert len(out) == 1
    assert out.iloc[0]["direction"] == triggers.SHORT
    assert out.iloc[0]["meta"]["bars_to_fail"] == 1


def test_failed_breakdown_is_a_long():
    df = mk([(101, 101.5, 100.5, 101.0),
             (100, 100.2, 98.5, 99.0),
             (99, 101.0, 98.9, 100.9)])
    out = triggers.failed_breakouts(df, 100.0, flat_atr(df), P)
    assert out.iloc[0]["direction"] == triggers.LONG


def test_return_outside_the_window_is_not_a_failed_breakout():
    df = mk([(99, 99.4, 98.5, 99.0)]                # below the level first
            + [(100, 101.5, 99.8, 101.0)]           # the breakout transition
            + [(101, 101.5, 100.8, 101.2)] * 5      # holds above for 5 bars
            + [(101, 101.2, 99.0, 99.0)])           # returns too late (K=3)
    out = triggers.failed_breakouts(df, 100.0, flat_atr(df), P)
    assert out.empty


def test_range_reclaim_shares_the_failed_breakout_math():
    df = mk([(99, 99.5, 98.5, 99.0),
             (100, 101.5, 99.8, 101.0),
             (101, 101.2, 99.0, 99.4)])
    fb = triggers.failed_breakouts(df, 100.0, flat_atr(df), P)
    rr = triggers.range_reclaims(df, 100.0, flat_atr(df), P, side="high")
    assert len(rr) == len(fb) == 1
    assert rr.iloc[0]["kind"] == "range_reclaim"
    assert rr.iloc[0]["direction"] == fb.iloc[0]["direction"]


def test_range_reclaim_ignores_the_wrong_side_of_a_boundary():
    """Closing BELOW the range high and back above is not a reclaim of it.

    Regression: range_reclaims() inherited failed_breakouts()'s two-sided math,
    so ordinary trade inside the range scored as a reclaim of its own upper
    boundary. On real data that roughly doubled the count.
    """
    # dips below the 100 edge, then closes back above: inside-range noise
    df = mk([(101, 101.5, 100.8, 101.0),
             (100.5, 100.8, 98.5, 99.0),          # closes below the edge
             (99.2, 101.4, 99.0, 101.0)])         # and back above
    assert triggers.range_reclaims(df, 100.0, flat_atr(df), P, side="high").empty
    # the same bars ARE a reclaim of a range LOW at 100
    assert len(triggers.range_reclaims(df, 100.0, flat_atr(df), P,
                                       side="low")) == 1
    # and S8, on a discrete S/R level, still accepts both sides
    assert len(triggers.failed_breakouts(df, 100.0, flat_atr(df), P)) == 1


def test_range_reclaim_requires_an_explicit_side():
    df = mk([(99, 99.5, 98.5, 99.0)])
    with pytest.raises(ValueError):
        triggers.range_reclaims(df, 100.0, flat_atr(df), P, side="upper")


# --------------------------------------------------------------------------
# S9 breakout / retest
# --------------------------------------------------------------------------

def test_breakout_retest_fires_on_a_rejection_at_the_level():
    df = mk([(99, 99.5, 98.5, 99.0),
             (100, 102.0, 99.8, 101.5),           # breakout above 100
             (101.5, 101.6, 100.8, 101.0),        # drifts back
             (100.5, 102.0, 97.5, 101.8)])        # retest w/ bullish rejection
    rej = triggers.rejection(df, feats_of(df), flat_atr(df), P)
    assert rej.iloc[3] == triggers.LONG, "fixture must produce the rejection"
    out = triggers.breakout_retests(df, 100.0, flat_atr(df), P, rej)
    assert len(out) == 1
    assert out.iloc[0]["direction"] == triggers.LONG
    assert out.iloc[0]["meta"]["bars_to_retest"] == 2


def test_retest_invalidated_by_a_close_back_through_the_level():
    """That sequence is an S8 failed breakout, not a retest -- they must not
    both fire on it."""
    df = mk([(99, 99.5, 98.5, 99.0),
             (100, 102.0, 99.8, 101.5),
             (101, 101.2, 98.0, 98.5),            # closes back below
             (98.5, 99.0, 95.0, 98.9)])
    rej = triggers.rejection(df, feats_of(df), flat_atr(df), P)
    assert triggers.breakout_retests(df, 100.0, flat_atr(df), P, rej).empty


def test_retest_without_a_rejection_candle_does_not_fire():
    df = mk([(99, 99.5, 98.5, 99.0),
             (100, 102.0, 99.8, 101.5),
             (101.5, 101.6, 100.1, 100.4)])       # touches zone, no rejection
    rej = triggers.rejection(df, feats_of(df), flat_atr(df), P)
    assert triggers.breakout_retests(df, 100.0, flat_atr(df), P, rej).empty


# --------------------------------------------------------------------------
# S14 trend filter
# --------------------------------------------------------------------------

def test_trend_bias_needs_both_price_and_slope():
    up = mk([(i, i + 1, i - 1, i + 0.5) for i in np.arange(100, 160, 1.0)])
    bias = triggers.trend_bias(up, P)
    assert bias.iloc[-1] == "bullish"

    flat = mk([(100, 101, 99, 100)] * 60)
    assert triggers.trend_bias(flat, P).iloc[-1] == "neutral"

    down = mk([(i, i + 1, i - 1, i - 0.5) for i in np.arange(160, 100, -1.0)])
    assert triggers.trend_bias(down, P).iloc[-1] == "bearish"


def test_price_above_a_falling_ema_is_not_bullish():
    """S14 needs BOTH conditions. A spike above the EMA during a downtrend
    satisfies close > EMA but not the slope, so it must stay neutral."""
    rows = [(i, i + 1, i - 1, i - 0.5) for i in np.arange(160, 100, -1.0)]
    rows.append((104, 130.0, 103.0, 129.0))     # one spike far above the EMA
    bias = triggers.trend_bias(mk(rows), P)
    assert bias.iloc[-1] == "neutral", "falling EMA must block a bullish read"


# --------------------------------------------------------------------------
# S11 momentum continuation
# --------------------------------------------------------------------------

BELOW = (100.0, 100.4, 99.6, 100.0)      # closes under a 100.5 minor level
CROSS = (100.0, 103.2, 99.9, 103.0)      # crosses it, body ratio ~0.91


def _mom(rows, bias_val="bullish", vol_ok=True, level=100.5):
    df = mk(rows)
    n = len(rows)
    return triggers.momentum_continuation(
        df, feats_of(df), level, P,
        pd.Series([bias_val] * n, index=df.index),
        pd.Series([vol_ok] * n, index=df.index))


def test_momentum_requires_trend_body_and_volume():
    out = _mom([BELOW, CROSS])
    assert len(out) == 1 and out.iloc[0]["direction"] == triggers.LONG
    assert out.iloc[0]["idx"] == 1, "fires on the bar that crosses"


@pytest.mark.parametrize("bias_val,vol_ok,rows", [
    ("neutral", True, [BELOW, CROSS]),                        # trend fails
    ("bullish", False, [BELOW, CROSS]),                       # volume fails
    ("bullish", True, [BELOW, (100.0, 103.2, 99.9, 100.6)]),  # body ratio fails
])
def test_momentum_blocked_when_any_condition_fails(bias_val, vol_ok, rows):
    assert _mom(rows, bias_val, vol_ok).empty


def test_momentum_fires_once_per_crossing_not_every_bar_beyond():
    """S11 is an event, not a state.

    Regression: written as a bare `close > minor_level` predicate this re-fired
    on every later strong-bodied, volume-expanded bar while price simply stayed
    beyond the level -- 2,000-4,300 firings per instrument on real data against
    150-460 for the other selective triggers.
    """
    out = _mom([BELOW, CROSS, CROSS, CROSS])
    assert len(out) == 1 and out.iloc[0]["idx"] == 1


def test_momentum_can_fire_again_after_price_returns():
    """Re-crossing is a new event; only staying beyond is not."""
    out = _mom([BELOW, CROSS, BELOW, CROSS])
    assert list(out["idx"]) == [1, 3]


def test_momentum_first_bar_is_never_an_event():
    """No prior bar means no transition to observe -- same rule as breakouts()."""
    assert _mom([CROSS]).empty


# --------------------------------------------------------------------------
# combined level-free frame
# --------------------------------------------------------------------------

def test_candle_triggers_returns_all_three_pattern_columns():
    df = mk([(100.0, 101.2, 97.0, 101.0)] * 8)
    out = triggers.candle_triggers(df, feats_of(df), flat_atr(df), P)
    assert list(out.columns) == ["rejection", "engulfing", "three_tail"]
    assert len(out) == len(df)
