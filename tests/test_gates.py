"""Stage 1 hard gates.

Frames are hand-built and handed to a GateContext directly, so each test pins
one rule. Common geometry: ATR 4 on every bar, so the S6 test-zone tolerance
is max(2 ticks, 0.15 x 4) = 0.6 and the stop buffer is 0.20 x 4 = 0.8.

The load-bearing tests are the causality ones (an unconfirmed swing is
neither a level nor a target; a gap is not a level before it exists) and the
ones pinning the four outcomes: `unknown` and `not_evaluated` must never pass
as `pass`.
"""
from __future__ import annotations

import copy
import math

import pandas as pd
import pytest

from features import levels, structure
from features.schema import (
    ATR,
    VOLUME_BASELINE,
    VOLUME_EXPANDED,
    Params,
    load_params,
)
from signal_engine import candles, gates
from signal_engine.gates import (
    FAIL,
    NOT_EVALUATED,
    PASS,
    UNKNOWN,
    Candidate,
    GateContext,
    GateResult,
)
from signal_engine.timeframes import TimeframeSet

CT = "America/Chicago"
P = load_params("MES")                     # tick 0.25, breakout.mode buffer
FLAT = (100.0, 100.5, 99.5, 100.2)
N = 40
PIVOT_COLS = ["idx", "ts", "price", "kind", "confirmed_idx", "confirmed_ts",
              "depth", "is_major", "prior_opposite_idx"]


def params(**dotted) -> Params:
    values = copy.deepcopy(P.values)
    for k, v in dotted.items():
        section, key = k.split("__")
        values[section][key] = v
    return Params(values=values, symbol="MES")


def bars(rows, freq="5min", start="2026-03-02 09:00"):
    ts = pd.date_range(pd.Timestamp(start), periods=len(rows), freq=freq, tz=CT)
    return pd.DataFrame({
        "ts": ts.tz_convert("UTC"),
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": 100, "trade_date": pd.Timestamp("2026-03-02").date(),
        ATR: 4.0, VOLUME_BASELINE: 100.0, VOLUME_EXPANDED: True,
        levels.PRIOR_DAY_HIGH: 110.0, levels.PRIOR_DAY_LOW: 90.0,
        levels.PRIOR_WEEK_HIGH: math.nan, levels.PRIOR_WEEK_LOW: math.nan,
        levels.RANGE_HIGH: math.nan, levels.RANGE_LOW: math.nan,
    })


def pivots(*rows):
    """rows: (price, kind, confirmed_idx, is_major)."""
    if not rows:
        return pd.DataFrame(columns=PIVOT_COLS)
    return pd.DataFrame([{
        "idx": max(c - 2, 0), "ts": pd.NaT, "price": pr, "kind": k,
        "confirmed_idx": c, "confirmed_ts": pd.NaT, "depth": 5.0,
        "is_major": m, "prior_opposite_idx": -1} for pr, k, c, m in rows],
        columns=PIVOT_COLS)


NO_GAPS = pd.DataFrame(columns=["trade_date", "zone_low", "zone_high",
                                "filled_date", "active_from"])


def ctx(entry=None, *, tt=None, piv=None, gaps=None, bias="bullish",
        p=P, veto=None):
    entry = entry if entry is not None else bars([FLAT] * N)
    frames, roles = {"5min": entry}, {"entry": "5min", "three_tail": "5min"}
    if tt is not None:
        frames["10min"], roles["three_tail"] = tt, "10min"
    tfs = TimeframeSet(symbol="MES", params=p, symbol_cfg={}, roles=roles,
                       frames=frames)
    b = pd.Series([bias] * len(entry), index=entry.index, dtype="object")
    piv = structure.mark_swing_deaths(piv if piv is not None else pivots(),
                                      entry["close"])
    return GateContext(tfs=tfs, entry=entry, pivots=piv,
                       gaps=gaps if gaps is not None else NO_GAPS,
                       bias=b, veto=veto)


def rows_with(overrides: dict[int, tuple]):
    rows = [FLAT] * N
    for i, r in overrides.items():
        rows[i] = r
    return rows


def cand(kind="rejection", direction="long", idx=20, *, role="entry",
         decision_idx=None, bars_=None, level=math.nan):
    return Candidate(kind=kind, direction=direction, role=role, idx=idx,
                     decision_idx=idx if decision_idx is None else decision_idx,
                     pattern_bars=bars_ or (idx,), level=level)


# a long rejection bar touching prior-day low 90: low 89.8, close 91
REJ = (90.6, 91.2, 89.8, 91.0)
# base bars for the gate 4 tests: closing at 91, below every swing-high
# target used there, so those targets stay live (spec S1 liveness)
LOW = (90.8, 91.3, 90.5, 91.0)


def low_rows(overrides: dict[int, tuple]):
    rows = [LOW] * N
    for i, r in overrides.items():
        rows[i] = r
    return rows


# --------------------------------------------------------------------------
# gate 1: structure
# --------------------------------------------------------------------------

def test_two_sided_three_tail_is_no_trade_not_a_direction():
    """Spec S19: `both` is excluded explicitly. The other directional gates
    are not evaluated, since there is no direction to evaluate them in."""
    r = gates.evaluate(cand("three_tail", candles.BOTH), ctx())
    assert r.result(1).status == FAIL and "S19" in r.result(1).detail
    assert all(r.result(g).status == NOT_EVALUATED for g in range(2, 6))
    assert not r.is_candidate


def test_a_plain_breakout_is_not_a_trigger():
    """S4 feeds S8/S9; on its own it is not in gate 1's list."""
    r = gates.evaluate(cand("breakout"), ctx())
    assert r.result(1).status == FAIL and not r.is_candidate


def test_breakout_mode_decides_which_breakout_trigger_exists():
    """S17 is an alternative to S4, never both on one symbol."""
    s17 = cand("confirmation_signal", level=100.0, bars_=(18, 19, 20))
    assert gates.gate_structure(s17, ctx()).status == FAIL      # mode: buffer
    cs = ctx(p=params(breakout__mode="confirmation_signal"))
    assert gates.gate_structure(s17, cs).status == PASS
    retest = cand("breakout_retest", level=100.0, bars_=(18, 19, 20))
    assert gates.gate_structure(retest, cs).status == FAIL


# --------------------------------------------------------------------------
# gate 2: level
# --------------------------------------------------------------------------

def test_rejection_at_a_marked_level_passes():
    c = ctx(bars(rows_with({20: REJ})))
    r = gates.gate_level(cand(), c)
    assert r.status == PASS and "prior day low" in r.detail


def test_rejection_in_open_space_fails():
    assert gates.gate_level(cand(), ctx()).status == FAIL


def test_level_free_pattern_just_outside_the_zone_fails():
    near = (91.0, 91.4, 90.7, 91.2)        # low 0.7 above 90; tolerance 0.6
    assert gates.gate_level(cand(), ctx(bars(rows_with({20: near})))).status == FAIL
    inside = (91.0, 91.4, 90.5, 91.2)      # 0.5 above
    assert gates.gate_level(cand(), ctx(bars(rows_with({20: inside})))).status == PASS


def test_momentum_is_exempt_from_gate_2():
    assert gates.gate_level(cand("momentum", level=100.0), ctx()).status == PASS


def test_three_tail_is_exempt_only_while_config_says_so():
    tt = cand("three_tail", level=100.0, bars_=(18, 19, 20))
    assert gates.gate_level(tt, ctx()).status == PASS
    strict = ctx(p=params(three_tail__requires_nearby_level=True))
    assert gates.gate_level(tt, strict).status == FAIL


def test_level_defined_trigger_is_checked_on_its_own_level():
    on = cand("failed_breakout", "short", level=110.0, bars_=(19, 20))
    off = cand("failed_breakout", "short", level=105.0, bars_=(19, 20))
    assert gates.gate_level(on, ctx()).status == PASS
    assert gates.gate_level(off, ctx()).status == FAIL


def test_an_unconfirmed_major_swing_is_not_a_level_yet():
    """Pivots are invisible until confirmed_idx. Using one early is lookahead."""
    c_late = ctx(piv=pivots((100.0, "low", 25, True)))
    c_known = ctx(piv=pivots((100.0, "low", 15, True)))
    assert gates.gate_level(cand(), c_late).status == FAIL
    assert gates.gate_level(cand(), c_known).status == PASS


def test_a_broken_major_swing_is_not_a_level():
    """Spec S1: a swing high dies on the first close above it. The flat bars
    close at 100.2, so a high at 100.0 is dead and one at 100.4 is not --
    both sit inside the rejection bar's range."""
    dead = ctx(piv=pivots((100.0, "high", 15, True)))
    live = ctx(piv=pivots((100.4, "high", 15, True)))
    assert gates.gate_level(cand(), dead).status == FAIL
    assert gates.gate_level(cand(), live).status == PASS


def test_an_unclassified_swing_is_not_a_major_level():
    """is_major NA means unknown, not major."""
    c = ctx(piv=pivots((100.0, "low", 15, pd.NA)))
    assert gates.gate_level(cand(), c).status == FAIL


def test_a_gap_is_not_a_level_before_its_session_opens():
    """For the RTH instruments a trade date begins the evening before. The gap
    exists from the RTH open, not from the first bar carrying its date."""
    entry = bars([FLAT] * N)
    gap = lambda active: pd.DataFrame({
        "trade_date": [entry["trade_date"].iloc[0]], "zone_low": [100.0],
        "zone_high": [100.1], "filled_date": [pd.NaT], "active_from": [active]})
    later = entry["ts"].iloc[25]
    earlier = entry["ts"].iloc[5]
    assert gates.gate_level(cand(), ctx(entry, gaps=gap(later))).status == FAIL
    assert gates.gate_level(cand(), ctx(entry, gaps=gap(earlier))).status == PASS


# --------------------------------------------------------------------------
# gate 3: confirmation
# --------------------------------------------------------------------------

def test_rejection_needs_volume_expansion_despite_its_own_clv():
    """Scoring gate 3: CLV never substitutes for volume, on any trigger."""
    entry = bars(rows_with({20: REJ}))
    entry.loc[20, VOLUME_EXPANDED] = False
    assert gates.gate_confirmation(cand(), ctx(entry)).status == FAIL


def test_no_volume_baseline_is_unknown_not_a_failure_of_volume():
    entry = bars([FLAT] * N)
    entry.loc[20, VOLUME_BASELINE] = math.nan
    entry.loc[20, VOLUME_EXPANDED] = False     # what confirmation.apply writes
    assert gates.gate_confirmation(cand(), ctx(entry)).status == UNKNOWN


def test_three_tail_volume_is_read_on_its_own_10min_bar():
    entry = bars([FLAT] * N)
    tt = bars([FLAT] * 20, freq="10min")
    tt.loc[9, VOLUME_EXPANDED] = False           # the cluster's own bar
    c = cand("three_tail", role="three_tail", idx=9, decision_idx=20,
             bars_=(7, 8, 9), level=100.0)
    assert gates.gate_confirmation(c, ctx(entry, tt=tt)).status == FAIL
    tt.loc[9, VOLUME_EXPANDED] = True
    entry.loc[20, VOLUME_EXPANDED] = False       # entry bar is irrelevant
    assert gates.gate_confirmation(c, ctx(entry, tt=tt)).status == PASS


# --------------------------------------------------------------------------
# gate 4: reward / risk
# --------------------------------------------------------------------------

def test_stop_is_the_pattern_extreme_beyond_the_buffer():
    """Rejection: its own low 89.8, minus buffer 0.8 = stop 89.0."""
    plan = gates.plan_trade(cand(), ctx(bars(rows_with({20: REJ}))))
    assert plan.invalidation == 89.8 and plan.stop == pytest.approx(89.0)
    assert plan.entry == 91.0 and plan.risk == pytest.approx(2.0)


def test_engulfing_stop_spans_both_bars():
    rows = rows_with({19: (100.0, 100.2, 98.0, 98.5),
                      20: (98.4, 101.0, 98.3, 100.8)})
    c = cand("engulfing", bars_=(19, 20))
    assert gates.plan_trade(c, ctx(bars(rows))).invalidation == 98.0


def test_three_tail_stop_is_the_most_extreme_tail_tip():
    tt = bars(rows_with({7: (100, 100.4, 97.1, 100.2), 8: (100, 100.4, 96.9, 100.2),
                         9: (100, 100.4, 97.0, 100.2)})[:20], freq="10min")
    c = cand("three_tail", role="three_tail", idx=9, decision_idx=20,
             bars_=(7, 8, 9), level=97.0)
    plan = gates.plan_trade(c, ctx(tt=tt))
    assert plan.invalidation == 96.9 and plan.entry == 100.2


def test_no_major_level_falls_back_to_exactly_2r_and_passes():
    """Spec S15: never discard a setup because no major level has formed."""
    r, plan = gates.gate_reward_risk(cand(), ctx(bars(rows_with({20: REJ}))))
    assert plan.target_source == "2R_fallback"
    assert plan.rr == 2.0 and plan.target == pytest.approx(95.0)
    assert r.status == PASS


def test_a_farther_major_level_is_the_target_and_rr_runs_above_2():
    """2R is a floor, not a cap: this is what keeps reward_risk_quality alive."""
    c = ctx(bars(low_rows({20: REJ})), piv=pivots((98.0, "high", 10, True)))
    r, plan = gates.gate_reward_risk(cand(), c)
    assert plan.target_source == "major_level" and plan.target == 98.0
    assert plan.rr == pytest.approx(3.5) and r.status == PASS


def test_a_major_level_inside_2r_fails_the_gate():
    c = ctx(bars(low_rows({20: REJ})), piv=pivots((93.0, "high", 10, True)))
    r, plan = gates.gate_reward_risk(cand(), c)
    assert plan.rr == pytest.approx(1.0) and r.status == FAIL


def test_the_nearest_major_level_beyond_entry_is_used():
    c = ctx(bars(low_rows({20: REJ})),
            piv=pivots((99.0, "high", 10, True), (96.0, "high", 12, True),
                       (88.0, "low", 11, True)))           # below entry: ignored
    assert gates.plan_trade(cand(), c).target == 96.0


def test_an_unconfirmed_major_level_is_not_a_target():
    c = ctx(bars(low_rows({20: REJ})), piv=pivots((93.0, "high", 21, True)))
    assert gates.plan_trade(cand(), c).target_source == "2R_fallback"


def test_a_broken_major_swing_is_not_a_target():
    """The nearer high at 93 was closed above at bar 15, so the target is the
    next LIVE one at 98 -- not a level price has already traded through."""
    rows = low_rows({15: (92.8, 93.8, 92.6, 93.5), 20: REJ})
    c = ctx(bars(rows), piv=pivots((93.0, "high", 10, True),
                                    (98.0, "high", 11, True)))
    plan = gates.plan_trade(cand(), c)
    assert plan.target == 98.0 and plan.rr == pytest.approx(3.5)


def test_a_wick_through_does_not_kill_a_target():
    rows = low_rows({15: (92.8, 93.6, 92.6, 92.9), 20: REJ})   # high > 93
    c = ctx(bars(rows), piv=pivots((93.0, "high", 10, True)))
    assert gates.plan_trade(cand(), c).target == 93.0


def test_disagreement_is_flagged_but_never_moves_the_target():
    c = ctx(bars(low_rows({20: REJ})), piv=pivots((99.0, "high", 10, True)))
    r, plan = gates.gate_reward_risk(cand(), c)
    assert plan.target == 99.0 and plan.disagreement_flag      # 8 vs 4 = 2.0x
    assert r.status == PASS and "flag" in r.detail


def test_a_stop_that_is_not_beyond_entry_fails_rather_than_dividing():
    """Momentum enters at its minor level; a trigger bar entirely above that
    level puts the Part 0 stop above the entry. There is no risk to measure."""
    gapped = (103.0, 104.0, 102.5, 103.8)
    c = cand("momentum", level=100.0)
    r, plan = gates.gate_reward_risk(c, ctx(bars(rows_with({20: gapped}))))
    assert r.status == FAIL and plan.risk <= 0


def test_retest_entry_is_one_tick_better_than_the_level():
    long_ = cand("breakout_retest", "long", level=100.0, bars_=(18, 19, 20))
    short = cand("breakout_retest", "short", level=100.0, bars_=(18, 19, 20))
    assert gates.entry_price(long_, ctx()) == 99.75
    assert gates.entry_price(short, ctx()) == 100.25


def test_unseeded_atr_makes_rr_unknown():
    entry = bars(rows_with({20: REJ}))
    entry.loc[20, ATR] = math.nan
    r, plan = gates.gate_reward_risk(cand(), ctx(entry))
    assert r.status == UNKNOWN and plan is None


# --------------------------------------------------------------------------
# gate 5: HTF alignment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bias,want", [("bullish", PASS), ("neutral", FAIL),
                                       ("bearish", FAIL), ("unknown", UNKNOWN)])
def test_continuation_needs_aligned_htf(bias, want):
    c = cand("momentum", "long", level=100.0)
    assert gates.gate_htf(c, ctx(bias=bias)).status == want


def test_a_missing_htf_bias_is_unknown():
    """Before the first HTF bar closes, align() leaves NaN."""
    k = ctx()
    k.bias.iloc[20] = math.nan
    assert gates.gate_htf(cand("momentum", level=100.0), k).status == UNKNOWN


@pytest.mark.parametrize("bias", ["bullish", "neutral", "bearish", "unknown"])
def test_reversals_are_never_htf_gated(bias):
    for kind in sorted(gates.REVERSAL):
        assert gates.gate_htf(cand(kind, "short"), ctx(bias=bias)).status == PASS


def test_every_breakout_retest_is_continuation_type():
    """The reversal list is closed: a counter-trend retest is not a reversal."""
    c = cand("breakout_retest", "short", level=100.0, bars_=(18, 19, 20))
    assert gates.gate_htf(c, ctx(bias="bullish")).status == FAIL


# --------------------------------------------------------------------------
# gate 6 and the verdict
# --------------------------------------------------------------------------

def _clean():
    return ctx(bars(rows_with({20: REJ})), bias="bearish")


def test_without_a_risk_engine_gate_6_is_not_evaluated_and_the_setup_is_kept():
    r = gates.evaluate(cand(), _clean())
    assert r.result(6).status == NOT_EVALUATED
    assert r.is_candidate and not r.risk_evaluated


def test_a_veto_removes_the_candidate():
    k = _clean()
    k.veto = lambda c: GateResult(6, FAIL, "daily loss limit")
    assert not gates.evaluate(cand(), k).is_candidate


def test_a_veto_must_answer_for_gate_6():
    k = _clean()
    k.veto = lambda c: GateResult(3, PASS)
    with pytest.raises(ValueError, match="gate 3"):
        gates.evaluate(cand(), k)


def test_unknown_on_any_core_gate_is_not_a_candidate():
    entry = bars(rows_with({20: REJ}))
    entry.loc[20, VOLUME_BASELINE] = math.nan
    r = gates.evaluate(cand(), ctx(entry, bias="bearish"))
    assert r.result(3).status == UNKNOWN and not r.is_candidate


def test_every_gate_is_evaluated_even_after_a_failure():
    """The log shows everything wrong with a setup, not just the first thing."""
    entry = bars([FLAT] * N)                   # open space: gate 2 fails
    entry.loc[20, VOLUME_EXPANDED] = False     # and gate 3
    r = gates.evaluate(cand(), ctx(entry))
    assert [x.status for x in r.failed()] == [FAIL, FAIL]
    assert r.plan is not None


def test_neutral_policy_other_than_continuation_only_is_refused():
    p = params(trend__neutral_policy="gate_everything")
    tfs = TimeframeSet(symbol="MES", params=p, symbol_cfg={},
                       roles={"entry": "5min"}, frames={"5min": bars([FLAT])})
    with pytest.raises(ValueError, match="continuation_only"):
        GateContext.build(tfs, {})


# --------------------------------------------------------------------------
# candidate construction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind,meta,expected", [
    ("rejection", {}, (20,)),
    ("momentum", {}, (20,)),
    ("engulfing", {}, (19, 20)),
    ("three_tail", {"bars": [15, 17, 20]}, (15, 17, 20)),
    ("failed_breakout", {"breakout_idx": 18}, (18, 19, 20)),
    ("range_reclaim", {"breakout_idx": 19}, (19, 20)),
    ("breakout_retest", {"breakout_idx": 14}, tuple(range(14, 21))),
    ("confirmation_signal", {"pierce_idx": 17}, (17, 18, 19, 20)),
])
def test_part_0_pattern_bars_per_trigger(kind, meta, expected):
    assert gates.pattern_bars(kind, 20, meta) == expected


def test_a_higher_frame_trigger_needs_its_landing_bar():
    with pytest.raises(ValueError, match="align_events"):
        gates.from_event({"kind": "three_tail", "idx": 9, "direction": "long",
                          "meta": {"bars": [7, 8, 9]}}, role="three_tail")
