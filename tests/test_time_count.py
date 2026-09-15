"""S18 time count exhaustion.

The flat-bar rule is the part most easily got wrong: an unchanged close neither
resets nor extends the count. Up, up, flat, up is three, not one and not four.
"""
from __future__ import annotations

import pandas as pd
import pytest

from features import time_count as tc
from features.schema import load_params

P = load_params("MES")          # exhaustion_threshold = 6


def counts(closes):
    return tc.time_count(pd.Series(closes, dtype="float64"))


# --------------------------------------------------------------------------
# direction
# --------------------------------------------------------------------------

def test_bar_direction_classifies_against_the_prior_close():
    d = tc.bar_direction(pd.Series([10.0, 11.0, 11.0, 10.0]))
    assert list(d) == [tc.FLAT, tc.UP, tc.FLAT, tc.DOWN]


def test_first_bar_has_no_direction():
    assert tc.bar_direction(pd.Series([10.0])).iloc[0] == tc.FLAT


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------

def test_consecutive_up_bars_accumulate():
    out = counts([1.0, 2, 3, 4, 5])
    assert list(out["time_count"]) == [0, 1, 2, 3, 4]
    assert out["count_direction"].iloc[-1] == tc.UP


def test_a_reversal_resets_to_one():
    out = counts([1.0, 2, 3, 2])
    assert list(out["time_count"]) == [0, 1, 2, 1]
    assert out["count_direction"].iloc[-1] == tc.DOWN


def test_flat_bar_holds_the_count_neither_resetting_nor_extending():
    """The S18 rule that is easiest to implement wrong."""
    out = counts([1.0, 2, 3, 3, 4])
    #             -  up up flat up
    assert list(out["time_count"]) == [0, 1, 2, 2, 3]
    assert out["count_direction"].iloc[-1] == tc.UP


def test_several_flat_bars_in_a_row_still_hold():
    out = counts([1.0, 2, 3, 3, 3, 3, 4])
    assert list(out["time_count"]) == [0, 1, 2, 2, 2, 2, 3]


def test_flat_bar_does_not_bridge_a_reversal():
    out = counts([1.0, 2, 3, 3, 2])
    assert list(out["time_count"]) == [0, 1, 2, 2, 1]
    assert out["count_direction"].iloc[-1] == tc.DOWN


# --------------------------------------------------------------------------
# exhaustion flag
# --------------------------------------------------------------------------

def test_exhaustion_fires_at_the_threshold():
    out = tc.exhaustion(pd.Series([float(i) for i in range(10)]), P)
    first = out.index[out["exhausted"]][0]
    assert out.loc[first, "time_count"] == 6         # threshold
    assert not out["exhausted"].iloc[:first].any()


def test_exhaustion_direction_is_reported():
    out = tc.exhaustion(pd.Series([float(-i) for i in range(10)]), P)
    assert out["exhausted"].iloc[-1]
    assert out["count_direction"].iloc[-1] == tc.DOWN


def test_flat_series_never_exhausts():
    out = tc.exhaustion(pd.Series([5.0] * 20), P)
    assert not out["exhausted"].any()


# --------------------------------------------------------------------------
# per timeframe, never blended
# --------------------------------------------------------------------------

def test_timeframes_are_computed_independently():
    frames = {
        "1D": pd.DataFrame({"close": [float(i) for i in range(10)]}),
        "5min": pd.DataFrame({"close": [1.0, 2, 1, 2, 1, 2, 1, 2, 1, 2]}),
    }
    out = tc.exhaustion_by_timeframe(frames, P)
    assert set(out) == {"1D", "5min"}
    assert out["1D"]["exhausted"].any()
    assert not out["5min"]["exhausted"].any()     # oscillating, never a run


# --------------------------------------------------------------------------
# S18 is a filter, not a trigger
# --------------------------------------------------------------------------

def test_continuation_into_exhaustion_is_reduced():
    assert tc.confidence_adjustment(True, tc.UP, "momentum", "long") == "reduce"
    assert tc.confidence_adjustment(True, tc.DOWN, "momentum", "short") == "reduce"


def test_continuation_against_exhaustion_is_unaffected():
    assert tc.confidence_adjustment(True, tc.UP, "momentum", "short") == "none"


@pytest.mark.parametrize("kind", ["rejection", "three_tail", "failed_breakout",
                                  "range_reclaim", "engulfing"])
def test_reversal_fading_an_exhausted_move_is_increased(kind):
    assert tc.confidence_adjustment(True, tc.UP, kind, "short") == "increase"
    assert tc.confidence_adjustment(True, tc.DOWN, kind, "long") == "increase"


def test_reversal_in_the_exhausted_direction_gets_no_bonus():
    assert tc.confidence_adjustment(True, tc.UP, "rejection", "long") == "none"


def test_no_adjustment_when_not_exhausted():
    for kind in ("momentum", "rejection", "three_tail"):
        assert tc.confidence_adjustment(False, tc.UP, kind, "long") == "none"
        assert tc.confidence_adjustment(True, tc.FLAT, kind, "long") == "none"
