"""Signal Engine enumeration: which levels are scanned, over which bars.

Reuses the gate-test fixtures (ATR 4 everywhere, flat bars closing at 100.2,
prior-day levels far away at 90/110). The S4 breakout buffer is then
max(2 ticks, 0.1 x 4) = 0.5.

The load-bearing tests are the causal ones: a level is scanned only while it
is live, an event is kept only if the bar before its pattern lies inside that
live interval, and a minor swing yields momentum on its FIRST cross only.
"""
from __future__ import annotations

import pandas as pd
import pytest

from features import confirmation, levels
from features.schema import CLV
from signal_engine import engine
from tests.test_gates import FLAT, N, bars, ctx, params, pivots, rows_with

SWING = (100.4, "high", 10, True)          # pivot at 8, confirmed at 10


def frame(rows):
    """Gate-test bars plus the candle anatomy the detectors read."""
    df = bars(rows)
    df = pd.concat([df, confirmation.candle_anatomy(df)], axis=1)
    df[CLV] = confirmation.clv(df)
    return df


def kinds(cands, kind):
    return [c for c in cands if c.kind == kind]


# --------------------------------------------------------------------------
# live intervals
# --------------------------------------------------------------------------

def test_a_swing_is_live_from_confirmation_to_the_bar_before_its_death():
    rows = rows_with({15: (100.3, 101.4, 100.2, 101.2)})      # closes > 100.4
    lv = [x for x in engine.marked_level_intervals(ctx(frame(rows),
                                                       piv=pivots(SWING)))
          if "major swing high" in x.names]
    assert len(lv) == 1 and lv[0].intervals == ((10, 14),)


def test_an_unbroken_swing_is_live_to_the_end_of_the_data():
    lv = [x for x in engine.marked_level_intervals(ctx(frame([FLAT] * N),
                                                       piv=pivots(SWING)))
          if "major swing high" in x.names]
    assert lv[0].intervals == ((10, N - 1),)


def test_levels_on_the_same_tick_are_scanned_once_under_both_names():
    lv = engine.marked_level_intervals(
        ctx(frame([FLAT] * N), piv=pivots((110.0, "high", 10, True))))
    at_110 = [x for x in lv if x.price == 110.0]
    assert len(at_110) == 1
    assert set(at_110[0].names) == {"prior day high", "major swing high"}
    assert at_110[0].intervals == ((0, N - 1),)


def test_prior_day_levels_are_live_over_each_run_of_their_value():
    e = frame([FLAT] * N)
    e.loc[20:, levels.PRIOR_DAY_HIGH] = 111.0                 # next session
    lv = {x.price: x.intervals for x in engine.marked_level_intervals(ctx(e))}
    assert lv[110.0] == ((0, 19),) and lv[111.0] == ((20, N - 1),)


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------

BREAK = (100.3, 101.4, 100.2, 101.2)       # closes beyond 100.4 + 0.5
FAIL_BACK = (101.0, 101.1, 100.0, 100.1)   # and back below 100.4


def test_a_failed_breakout_of_a_live_swing_is_found_on_full_frame_positions():
    c = ctx(frame(rows_with({15: BREAK, 16: FAIL_BACK})), piv=pivots(SWING))
    fb = kinds(engine.level_dependent_candidates(c), "failed_breakout")
    assert len(fb) == 1
    assert (fb[0].idx, fb[0].direction, fb[0].pattern_bars) == (16, "short", (15, 16))
    assert "major swing high" in fb[0].level_name


def test_a_breakout_of_a_level_not_yet_confirmed_is_not_enumerated():
    """The swing confirms on the breakout bar itself, so the bar before the
    pattern did not have it as a marked level."""
    late = (100.4, "high", 15, True)
    c = ctx(frame(rows_with({15: BREAK, 16: FAIL_BACK})), piv=pivots(late))
    assert kinds(engine.level_dependent_candidates(c), "failed_breakout") == []


def test_a_sub_buffer_close_does_not_cost_the_real_breakout():
    """Regression, end to end. Bar 13 closes at 100.7: beyond the 100.4 swing
    but inside S4's 0.5 buffer, so it is not a breakout. Under a bare-close
    liveness rule it killed the swing, and the real breakout at bar 15 (and
    the failed breakout it produced) was then of a dead level."""
    rows = rows_with({13: (100.3, 100.8, 100.2, 100.7), 15: BREAK, 16: FAIL_BACK})
    c = ctx(frame(rows), piv=pivots(SWING))
    fb = kinds(engine.level_dependent_candidates(c), "failed_breakout")
    assert [f.idx for f in fb] == [16]
    log = engine.run(c)
    assert (log.loc[log["kind"] == "failed_breakout", "g2_level"] == "pass").all()


def test_a_pattern_starting_after_the_level_died_is_not_kept():
    """The scan window runs past the death bar so a pattern that began while
    the level was live can finish. A SECOND break-and-fail later in that same
    window starts after the swing died at bar 15, so it is not about a marked
    level and must not be enumerated."""
    rows = rows_with({15: BREAK, 16: FAIL_BACK, 20: BREAK, 21: FAIL_BACK})
    c = ctx(frame(rows), piv=pivots(SWING))
    fb = kinds(engine.level_dependent_candidates(c), "failed_breakout")
    assert [f.idx for f in fb] == [16]


def test_momentum_fires_on_the_first_cross_of_a_minor_swing_only():
    """The crossing close kills the minor swing, so a later re-cross of the
    same price is not a second momentum event -- the S11 'event, not state'
    rule, applied through liveness."""
    strong = (100.2, 101.5, 100.1, 101.4)      # body ratio ~0.86
    rows = rows_with({15: strong, 16: FLAT, 17: strong})
    c = ctx(frame(rows), piv=pivots((100.4, "high", 10, False)))
    mo = kinds(engine.level_dependent_candidates(c), "momentum")
    assert [m.idx for m in mo] == [15]


def test_a_sub_buffer_cross_of_a_minor_swing_is_its_only_momentum_event():
    """Regression. Bar 13 is a strong close at 100.7: through the 100.4 minor
    swing but inside S4's 0.5 buffer. When minor swings shared the major
    buffer, that cross did not kill the level, so after a dip the strong
    re-cross at bar 15 fired momentum a second time on the same level."""
    sub_buffer = (100.2, 100.8, 100.1, 100.7)      # body ratio ~0.71
    strong = (100.2, 101.5, 100.1, 101.4)
    rows = rows_with({13: sub_buffer, 14: FLAT, 15: strong})
    c = ctx(frame(rows), piv=pivots((100.4, "high", 10, False)))
    mo = kinds(engine.level_dependent_candidates(c), "momentum")
    assert [m.idx for m in mo] == [13]


def test_no_short_momentum_off_a_minor_swing_high():
    """Regression, end to end (4 of 3,249 momentum events on real data). Bar
    12 closes exactly at the 100.4 minor high; bar 13 closes strongly below
    it in a bearish trend. That is not a break of a swing high."""
    rows = rows_with({12: (100.0, 100.4, 99.9, 100.4),
                      13: (100.4, 100.45, 99.2, 99.3)})
    c = ctx(frame(rows), piv=pivots((100.4, "high", 10, False)), bias="bearish")
    assert kinds(engine.level_dependent_candidates(c), "momentum") == []


def test_minor_highs_and_lows_on_one_tick_stay_separate_levels():
    """Merging them would leave one level with no single side to match.
    Closes sit exactly at 100.4, which kills neither (both rules are strict)."""
    at = (100.0, 100.5, 99.9, 100.4)
    c = ctx(frame([at] * N), piv=pivots((100.4, "high", 10, False),
                                          (100.4, "low", 30, False)))
    sides = sorted(side for side, lv in engine.minor_level_intervals(c)
                   if lv.price == 100.4)
    assert sides == ["high", "low"]


def test_an_unclassified_swing_is_neither_major_nor_minor():
    strong = (100.2, 101.5, 100.1, 101.4)
    c = ctx(frame(rows_with({15: strong})), piv=pivots((100.4, "high", 10, pd.NA)))
    assert kinds(engine.level_dependent_candidates(c), "momentum") == []
    assert not [x for x in engine.marked_level_intervals(c) if "swing" in x.name]


def test_range_reclaim_is_scanned_against_the_edge_before_the_escape():
    """The rolling edge moves onto the escape bar's own high; the reclaim is
    of the edge that held until the bar before."""
    e = frame(rows_with({18: (100.3, 101.4, 100.2, 101.2),
                         19: (101.0, 101.1, 100.0, 100.2)}))
    e.loc[10:17, levels.RANGE_HIGH] = 100.5
    e.loc[18:, levels.RANGE_HIGH] = 101.4
    rr = kinds(engine.level_dependent_candidates(ctx(e)), "range_reclaim")
    assert len(rr) == 1
    assert (rr[0].idx, rr[0].direction, rr[0].level) == (19, "short", 100.5)


def test_breakout_mode_selects_s17_or_s9_never_both():
    pierce = (100.2, 101.0, 100.1, 100.3)      # high > 100.4, closes below
    confirm = (100.4, 101.8, 100.3, 101.5)     # closes > the pierce high
    rows = rows_with({15: pierce, 16: confirm})
    buf = engine.level_dependent_candidates(ctx(frame(rows), piv=pivots(SWING)))
    cs = engine.level_dependent_candidates(
        ctx(frame(rows), piv=pivots(SWING),
            p=params(breakout__mode="confirmation_signal")))
    assert kinds(buf, "confirmation_signal") == []
    s17 = kinds(cs, "confirmation_signal")
    assert len(s17) == 1 and s17[0].pattern_bars == (15, 16)
    assert kinds(cs, "breakout_retest") == []


def test_a_level_held_by_two_sources_yields_each_event_once():
    """A prior-day high on the swing's tick is merged before scanning."""
    e = frame(rows_with({15: BREAK, 16: FAIL_BACK}))
    e.loc[:, levels.PRIOR_DAY_HIGH] = 100.4        # same tick as the swing
    c = ctx(e, piv=pivots(SWING))
    assert len(kinds(engine.level_dependent_candidates(c), "failed_breakout")) == 1


# --------------------------------------------------------------------------
# the log
# --------------------------------------------------------------------------

def test_run_logs_every_trigger_with_every_gate():
    c = ctx(frame(rows_with({15: BREAK, 16: FAIL_BACK})), piv=pivots(SWING))
    log = engine.run(c)
    assert len(log) == len(engine.candidates(c)) >= 1
    for g, name in {1: "structure", 2: "level", 3: "confirmation",
                    4: "reward_risk", 5: "htf_alignment", 6: "risk_veto"}.items():
        assert f"g{g}_{name}" in log.columns and f"g{g}_detail" in log.columns
    assert (log["g6_risk_veto"] == "not_evaluated").all()
    assert not log["risk_evaluated"].any()


def test_a_failed_breakout_passes_gate_2_on_the_level_it_broke():
    """End to end: enumeration and gate 2's anchor agree on the level."""
    c = ctx(frame(rows_with({15: BREAK, 16: FAIL_BACK})), piv=pivots(SWING))
    log = engine.run(c)
    fb = log[log["kind"] == "failed_breakout"]
    assert (fb["g2_level"] == "pass").all()
