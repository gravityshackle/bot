"""Stage 2 confidence scoring.

Each test here is written to FAIL against the naive reading of the spec, not
just to pass against the implementation. The naive readings are what the spec
text says when taken literally: `abs(CLV)`, reversal context with no
with-trend case, exhaustion read at the decision bar, confluence that ignores
the traded level's own type, magnitude read off one bar. Every one of those
produced wrong scores on real data.

Fixtures reuse the gate tests' geometry: ATR 4 (test-zone tolerance 0.6),
prior-day high/low at 110/90, flat bars closing at 100.2.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from features import levels
from features.schema import CLV, VOL_REGIME, VOLUME_RATIO
from signal_engine import gates, scoring
from signal_engine.gates import Candidate
from tests.test_engine import frame as anatomy_frame
from tests.test_gates import FLAT, LOW, N, REJ, cand, ctx, low_rows, pivots, rows_with

W = scoring.load_scoring()


def frame(rows, vol_ratio=1.5, regime="normal"):
    df = anatomy_frame(rows)
    df[VOLUME_RATIO] = vol_ratio
    df[VOL_REGIME] = regime
    return df


def no_exhaustion(entry):
    flat = pd.DataFrame({"exhausted": False, "count_direction": 0},
                        index=entry.index)
    return {"5min": flat}


def sctx(c, exhaustion=None):
    return scoring.ScoreContext.build(
        c, exhaustion=exhaustion if exhaustion is not None
        else no_exhaustion(c.entry))


def score(candidate, c, exhaustion=None):
    rep = gates.evaluate(candidate, c)
    return scoring.score(rep, sctx(c, exhaustion))


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_weights_sum_to_one_and_match_the_spec():
    w = W.get("weights")
    assert sum(w.values()) == pytest.approx(1.0)
    assert w == {"trigger_quality": 0.25, "confirmation_strength": 0.20,
                 "level_confluence": 0.20, "directional_context": 0.15,
                 "reward_risk_quality": 0.15, "volatility_fit": 0.05}


def test_the_total_is_100_times_the_weighted_sum_of_the_logged_parts():
    """The log is the record: the final score must be reproducible from it."""
    e = frame(rows_with({20: REJ}))
    s = score(cand(), ctx(e, bias="bullish"))
    w = W.get("weights")
    expect = 100 * sum(w[k] * getattr(s, k) for k in w)
    assert s.score == pytest.approx(expect)


# --------------------------------------------------------------------------
# 1. trigger quality
# --------------------------------------------------------------------------

def test_rejection_magnitude_is_its_wick_ratio_over_twice_the_threshold():
    # REJ: body 0.4, lower wick 0.8 -> ratio 2.0; threshold 2.0 -> 2/(2*2) = 0.5
    s = score(cand(), ctx(frame(rows_with({20: REJ}))))
    assert s.tq_magnitude == pytest.approx(0.5)
    assert s.trigger_quality == pytest.approx(0.70 * (0.85 + 0.15 * 0.5))


def test_three_tail_magnitude_averages_capped_bars_so_one_doji_cannot_saturate():
    """Naive: read the completing bar (a zero-body doji here, ratio infinite,
    so 1.0), or average uncapped ratios (infinite). Spec reading (#4): cap
    each tail bar, then average."""
    doji = (100.0, 100.4, 97.0, 100.0)          # zero body, wick 3.0 -> 1.0
    tail = (100.0, 100.4, 97.0, 99.0)           # body 1.0, wick 2.0 -> 0.5
    tt = frame(rows_with({7: tail, 8: tail, 9: doji})[:20])
    c = ctx(frame([FLAT] * N), tt=tt)
    three = Candidate("three_tail", "long", "three_tail", 9, 20, (7, 8, 9), 97.0)
    s = score(three, c)
    assert s.tq_magnitude == pytest.approx((0.5 + 0.5 + 1.0) / 3)


def test_breakout_retest_uses_its_retest_bars_rejection_magnitude():
    retest = cand("breakout_retest", level=90.0, bars_=(18, 19, 20))
    s = score(retest, ctx(frame(rows_with({20: REJ}))))
    assert s.tq_magnitude == pytest.approx(0.5)
    assert s.tq_magnitude_basis.startswith("retest bar rejection")


@pytest.mark.parametrize("kind", ["failed_breakout", "range_reclaim"])
def test_undefined_magnitudes_take_the_documented_midpoint(kind):
    c = cand(kind, "short", level=110.0, bars_=(19, 20))
    s = score(c, ctx(frame([FLAT] * N)))
    assert s.tq_magnitude == 0.5 and "no magnitude defined" in s.tq_magnitude_basis


def test_momentum_magnitude_is_body_ratio_over_the_configured_full_ratio():
    strong = (100.2, 101.5, 100.1, 101.4)       # body 1.2 / range 1.4
    s = score(cand("momentum", level=100.4), ctx(frame(rows_with({20: strong}))))
    assert s.tq_magnitude == pytest.approx(min((1.2 / 1.4) / 0.8, 1.0))


def test_engulfing_magnitude_is_body_over_prior_body_over_twice_the_multiplier():
    rows = rows_with({19: (100.0, 100.2, 98.0, 98.5),     # prior body 1.5
                      20: (98.4, 101.0, 98.3, 100.8)})    # body 2.4 -> 1.6x
    e = frame(rows)
    e[levels.PRIOR_DAY_LOW] = 98.0                        # a level to be at
    s = score(cand("engulfing", bars_=(19, 20)), ctx(e))
    assert s.tq_magnitude == pytest.approx((2.4 / 1.5) / (2 * 1.3))


def test_a_weak_three_tail_still_outranks_a_perfect_momentum_on_quality():
    """Spec: magnitude is a tiebreaker within a type, never a way for a
    weak-type trigger to beat a strong-type one."""
    assert 1.00 * 0.85 > 0.60 * 1.00
    assert W.get("base_scores.three_tail") * W.get("magnitude.floor_multiplier") \
        > W.get("base_scores.momentum")


# --------------------------------------------------------------------------
# 2. confirmation strength
# --------------------------------------------------------------------------

def test_clv_against_the_trade_earns_nothing():
    """Naive `abs(CLV)` credits a long whose trigger bar closed at its LOW as
    fully as one that closed at its high. Spec reading (#5): signed, floored."""
    bearish_close = (100.8, 101.0, 99.0, 99.2)     # CLV -0.8
    e = frame(rows_with({20: bearish_close}))
    s = score(cand("momentum", "long", level=100.4), ctx(e))
    assert e[CLV].iloc[20] == pytest.approx(-0.8)
    assert s.cs_clv == 0.0


def test_clv_with_the_trade_is_credited_by_its_size():
    e = frame(rows_with({20: REJ}))                # CLV of REJ is +0.714...
    s = score(cand(), ctx(e))
    assert s.cs_clv == pytest.approx(float(e[CLV].iloc[20]))


def test_volume_score_is_ratio_over_twice_the_multiplier_capped():
    s = score(cand(), ctx(frame(rows_with({20: REJ}), vol_ratio=2.25)))
    assert s.cs_volume == pytest.approx(2.25 / 3.0)
    s = score(cand(), ctx(frame(rows_with({20: REJ}), vol_ratio=9.0)))
    assert s.cs_volume == 1.0


def test_an_unknown_volume_ratio_leaves_the_score_unknown_not_zero():
    e = frame(rows_with({20: REJ}))
    e.loc[20, VOLUME_RATIO] = math.nan
    s = score(cand(), ctx(e))
    assert math.isnan(s.confirmation_strength) and math.isnan(s.score)


# --------------------------------------------------------------------------
# 3. level confluence
# --------------------------------------------------------------------------

def test_a_lone_marked_level_counts_its_own_type():
    """Naive: count only OTHER levels near the traded one, so an isolated real
    level scores 0 -- the same as open space. Spec reading (#6): 1/3."""
    s = score(cand(), ctx(frame(rows_with({20: REJ}))))
    assert s.lc_types == "prior_day" and s.level_confluence == pytest.approx(1 / 3)


def test_confluence_counts_types_not_levels():
    """A prior-day high and a prior-day low stacked together are one type."""
    e = frame(rows_with({20: REJ}))
    e[levels.PRIOR_DAY_HIGH] = 90.3                # both inside the zone
    s = score(cand(), ctx(e))
    assert s.level_confluence == pytest.approx(1 / 3)


def test_confluence_is_capped_at_three_types():
    e = frame(rows_with({20: REJ}))
    e[levels.PRIOR_WEEK_LOW] = 90.1
    e[levels.RANGE_LOW] = 89.9
    c = ctx(e, piv=pivots((90.2, "low", 10, True)))
    s = score(cand(), c)
    assert len(s.lc_types.split(",")) == 4 and s.level_confluence == 1.0


def test_open_space_three_tail_has_zero_confluence():
    tt = frame(rows_with({7: (100, 100.4, 97.1, 100.2), 8: (100, 100.4, 96.9, 100.2),
                          9: (100, 100.4, 97.0, 100.2)})[:20])
    c = ctx(frame([FLAT] * N), tt=tt)
    three = Candidate("three_tail", "long", "three_tail", 9, 20, (7, 8, 9), 97.0)
    assert score(three, c).level_confluence == 0.0


# --------------------------------------------------------------------------
# 4. directional context
# --------------------------------------------------------------------------

def test_a_reversal_with_the_htf_trend_scores_full_context():
    """Naive: only 'fading', 'neutral' and 'exhausted' exist, so a long
    rejection in a bullish HTF falls to a default. Spec reading (#1): 1.0."""
    s = score(cand(), ctx(frame(rows_with({20: REJ})), bias="bullish"))
    assert (s.directional_context, s.dc_case) == (1.0, "with_trend")


@pytest.mark.parametrize("bias,value,case", [("neutral", 0.5, "neutral"),
                                             ("bearish", 0.2, "against_trend")])
def test_reversal_context_without_exhaustion(bias, value, case):
    s = score(cand(), ctx(frame(rows_with({20: REJ})), bias=bias))
    assert (s.directional_context, s.dc_case) == (value, case)


def test_exhaustion_is_read_on_the_bar_before_the_pattern():
    """A down run exhausted through bar 19; the long rejection at bar 20 is
    what breaks it. Naive (read at the decision bar): no longer exhausted, so
    it falls to 'against_trend' 0.2. Spec reading (#2): 1.0."""
    e = frame(rows_with({20: REJ}))
    ex = pd.DataFrame({"exhausted": False, "count_direction": 0}, index=e.index)
    ex.loc[19, ["exhausted", "count_direction"]] = [True, -1]
    ex.loc[20, ["exhausted", "count_direction"]] = [False, 1]
    s = score(cand(), ctx(e, bias="bearish"), exhaustion={"5min": ex})
    assert (s.directional_context, s.dc_case) == (1.0, "exhausted:5min")


def test_exhaustion_in_the_trade_direction_does_not_count_for_a_reversal():
    e = frame(rows_with({20: REJ}))
    ex = pd.DataFrame({"exhausted": False, "count_direction": 0}, index=e.index)
    ex.loc[19, ["exhausted", "count_direction"]] = [True, 1]      # UP run
    s = score(cand(), ctx(e, bias="bearish"), exhaustion={"5min": ex})
    assert s.dc_case == "against_trend"


def test_continuation_context_is_one_regardless_of_bias():
    """It only reaches scoring with the HTF aligned (gate 5)."""
    strong = (100.2, 101.5, 100.1, 101.4)
    s = score(cand("momentum", level=100.4), ctx(frame(rows_with({20: strong}))))
    assert (s.directional_context, s.dc_case) == (1.0, "continuation")


def test_unknown_htf_bias_leaves_a_reversal_unscored():
    s = score(cand(), ctx(frame(rows_with({20: REJ})), bias="unknown"))
    assert math.isnan(s.directional_context) and math.isnan(s.score)


# --------------------------------------------------------------------------
# 5. reward/risk quality, 6. volatility fit
# --------------------------------------------------------------------------

def test_a_2r_fallback_scores_zero_reward_risk_quality():
    s = score(cand(), ctx(frame(low_rows({20: REJ}))))
    assert s.reward_risk_quality == 0.0


def test_reward_risk_quality_is_linear_to_the_cap():
    e = frame(low_rows({20: REJ}))
    s = score(cand(), ctx(e, piv=pivots((97.0, "high", 10, True))))   # RR 3.0
    assert s.reward_risk_quality == pytest.approx(0.5)
    s = score(cand(), ctx(e, piv=pivots((101.0, "high", 10, True))))  # RR 5.0
    assert s.reward_risk_quality == 1.0


@pytest.mark.parametrize("kind,regime,want", [
    ("momentum", "low", 0.5), ("momentum", "high", 1.0),
    ("rejection", "high", 0.6), ("rejection", "low", 1.0),
    ("engulfing", "high", 0.6), ("failed_breakout", "high", 0.6),
])
def test_volatility_fit_by_trigger_group(kind, regime, want):
    c = cand(kind, level=90.0, bars_=(19, 20))          # at prior-day low
    s = score(c, ctx(frame(rows_with({20: REJ}), regime=regime)))
    assert s.volatility_fit == want


def test_unknown_regime_leaves_the_score_unknown():
    s = score(cand(), ctx(frame(rows_with({20: REJ}), regime="unknown")))
    assert math.isnan(s.volatility_fit) and math.isnan(s.score)


def test_only_candidates_are_scored():
    """Stage 2 never sees a setup that failed a Stage 1 gate."""
    e = frame([FLAT] * N)                           # open space: gate 2 fails
    c = ctx(e)
    rep = gates.evaluate(cand(), c)
    assert not rep.is_candidate
    with pytest.raises(ValueError, match="candidate"):
        scoring.score(rep, sctx(c))


# --------------------------------------------------------------------------
# three-tail confirmation is CLV only
# --------------------------------------------------------------------------

def _s19_ctx(vol_ratio=0.3):
    tt = frame(rows_with({7: (100, 100.4, 97.1, 100.2), 8: (100, 100.4, 96.9, 100.2),
                          9: (100, 100.4, 97.0, 100.2)})[:20], vol_ratio=vol_ratio)
    c = ctx(frame([FLAT] * N), tt=tt)
    return c, Candidate("three_tail", "long", "three_tail", 9, 20, (7, 8, 9), 97.0)


def test_three_tail_confirmation_is_its_clv_alone():
    """Its tail bars are quiet by construction (volume score averaged 0.22 on
    real data), the reason it is exempt from gate 3. Naive: average in volume
    anyway, which scored three-tail's confirmation ~10 points under others."""
    c, three = _s19_ctx(vol_ratio=0.3)
    s = score(three, c)
    assert s.confirmation_strength == pytest.approx(s.cs_clv)
    assert s.cs_basis == "clv only (three-tail)"


def test_three_tail_is_scored_without_a_volume_baseline():
    """Volume is not an input to it, so its absence is not 'unknown'."""
    c, three = _s19_ctx(vol_ratio=math.nan)
    s = score(three, c)
    assert not math.isnan(s.score)


def test_other_types_still_average_volume_and_clv():
    s = score(cand(), ctx(frame(rows_with({20: REJ}), vol_ratio=2.25)))
    assert s.confirmation_strength == pytest.approx((s.cs_volume + s.cs_clv) / 2)
    assert s.cs_basis == "average(volume, clv)"


# --------------------------------------------------------------------------
# momentum into exhaustion (S18)
# --------------------------------------------------------------------------

STRONG = (100.2, 101.5, 100.1, 101.4)


def _exh(e, at, direction):
    ex = pd.DataFrame({"exhausted": False, "count_direction": 0}, index=e.index)
    ex.loc[at, ["exhausted", "count_direction"]] = [True, direction]
    return {"1h": ex}


def test_momentum_into_exhaustion_scores_reduced_context():
    """S18: reduce confidence on momentum in the exhausted direction. Naive:
    continuation is always 1.0, so S18's clause applied nowhere."""
    e = frame(rows_with({20: STRONG}))
    s = score(cand("momentum", level=100.4), ctx(e), exhaustion=_exh(e, 19, 1))
    assert (s.directional_context, s.dc_case) == (0.6, "continuation_exhausted:1h")


def test_momentum_exhaustion_is_read_on_the_bar_before_the_pattern():
    """Same anchor as the reversal branch: exhausted through bar 19 only."""
    e = frame(rows_with({20: STRONG}))
    ex = _exh(e, 19, 1)
    ex["1h"].loc[20, ["exhausted", "count_direction"]] = [False, 0]
    s = score(cand("momentum", level=100.4), ctx(e), exhaustion=ex)
    assert s.directional_context == 0.6


def test_exhaustion_against_the_momentum_does_not_reduce_it():
    e = frame(rows_with({20: STRONG}))
    s = score(cand("momentum", level=100.4), ctx(e), exhaustion=_exh(e, 19, -1))
    assert (s.directional_context, s.dc_case) == (1.0, "continuation")


def test_only_momentum_is_reduced_by_exhaustion():
    """S18 names momentum; breakout/retest keeps full continuation context."""
    e = frame(rows_with({20: REJ}))
    retest = cand("breakout_retest", level=90.0, bars_=(18, 19, 20))
    s = score(retest, ctx(e), exhaustion=_exh(e, 17, 1))
    assert (s.directional_context, s.dc_case) == (1.0, "continuation")
