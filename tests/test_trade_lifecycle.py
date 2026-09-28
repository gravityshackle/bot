"""Phase 4.2 exit state machine: exit spec Part 3, corrected event-driven version.

IN_POSITION -> BREAKEVEN at 1R; BREAKEVEN -> TRAILING the moment a new
swing CONFIRMS, independent of the target price; TRAILING exits only on its
stop. Every test is a hand-built path whose outcome can be worked by hand.

Geometry (MES-like: tick 0.25, $5/point, 1-tick slippage, $0.75/side): a
long fills at 100.00 on minute 0, initial stop 99.00, Part 0 invalidation
99.25 (so the ATR buffer is 0.25), target 102.00, 1R = 101.00.

Timing: the decision bar is entry-frame bar 0 (opens T0 - 5min), so 5-minute
bar k covers minutes [5(k-1), 5k) after T0. A swing at bar p with
confirmed_idx c is KNOWN at the close of bar c, i.e. from minute 5c.

The load-bearing test is the lookahead one: a swing that has formed but not
yet confirmed must not stop the target from filling. The original Part 3
decided "trail or take the target" at target-touch, which needs exactly that
unconfirmed swing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from execution.simulated_execution import CostModel, ExecConfig
from execution.trade_lifecycle import ExitConfig, TradeSpec, run_trade
from signal_engine.gates import LONG, SHORT, TradePlan
from tests.test_simulated_execution import (
    COSTS_CFG,
    D0,
    EXEC_CFG,
    MIN,
    SYMBOL_CFG,
    T0,
    mbars,
)

EXITS_CFG = {"breakeven_at_r": 1.0, "order_changes_take_effect": "next_bar",
             "trail_swing_side": "stop_side", "trail_buffer": "initial"}
FIVE = pd.Timedelta(minutes=5)
N = 2                                               # pivot_n_ltf

LONG_SPEC = TradeSpec(direction=LONG, entry=100.00, stop=99.00, target=102.00,
                      invalidation=99.25)
SHORT_SPEC = TradeSpec(direction=SHORT, entry=100.00, stop=101.00, target=98.00,
                       invalidation=100.75)

FILL = (100.25, 100.50, 99.75, 100.25)             # fills the long at 100.00
MID = (100.75, 101.00, 100.75, 101.00)             # reaches 1R (101.00), nothing else
HOLD = (101.25, 101.50, 101.00, 101.25)            # above breakeven, below the target
THROUGH_TARGET = (101.75, 102.50, 101.75, 102.25)


def entry_frame(n=60) -> pd.DataFrame:
    return pd.DataFrame({"ts": [T0 - FIVE + k * FIVE for k in range(n)],
                         "trade_date": [D0] * n})


def pivots(*rows) -> pd.DataFrame:
    """(frame idx, price, kind) -> confirmed N bars later."""
    frame = entry_frame()
    out = [{"idx": p, "ts": frame["ts"][p], "price": price, "kind": kind,
            "confirmed_idx": p + N, "confirmed_ts": frame["ts"][p + N]}
           for p, price, kind in rows]
    return pd.DataFrame(out, columns=["idx", "ts", "price", "kind", "confirmed_idx",
                                      "confirmed_ts"])


def run(rows, piv=None, spec=LONG_SPEC):
    return run_trade(mbars(rows), entry_frame(), piv if piv is not None else pivots(),
                     spec, decision_idx=0, contracts=1,
                     cost=CostModel.from_config(COSTS_CFG, SYMBOL_CFG),
                     cfg=ExecConfig.from_config(EXEC_CFG),
                     xcfg=ExitConfig.from_config({"exits": EXITS_CFG}), timeframe="5min")


def states(res):
    return [t.to_state for t in res.transitions]


# ---------------------------------------------------------------------------
# IN_POSITION and BREAKEVEN
# ---------------------------------------------------------------------------

def test_stopped_out_before_1r():
    res = run([FILL, (99.50, 99.75, 99.00, 99.25)])
    assert res.path == "stopped_out" and res.exit.fill.fill_price == 98.75
    assert states(res) == []


def test_1r_moves_the_stop_to_breakeven_and_a_return_to_entry_scratches():
    res = run([FILL, MID, (100.50, 100.50, 100.00, 100.25)])
    assert states(res) == ["breakeven"]
    assert res.transitions[0].ts == T0 + 2 * MIN and res.transitions[0].stop == 100.00
    assert res.path == "scratch" and res.exit.fill.fill_price == 99.75


def test_the_1r_bar_keeps_the_old_stop_for_the_rest_of_that_bar():
    """The bar reached 101.00 AND 99.75. If the high came first, a breakeven
    stop would have been hit; if the low came first, it wasn't there yet.
    Unknowable, so the move takes effect from the next bar."""
    res = run([FILL, (100.50, 101.00, 99.75, 100.50), THROUGH_TARGET])
    assert res.path == "planned_exit", res


def test_the_fill_bar_cannot_trigger_breakeven():
    """The fill bar's 101.00 high may have come before the fill at 100.00."""
    res = run([(100.25, 101.00, 99.75, 100.50), (100.25, 100.50, 99.75, 100.00),
               THROUGH_TARGET])
    assert res.path == "planned_exit", res


def test_target_fills_in_breakeven_when_no_swing_has_confirmed():
    res = run([FILL, MID, HOLD, THROUGH_TARGET])
    assert states(res) == ["breakeven"]
    assert res.path == "planned_exit" and res.exit.fill.fill_price == 102.00
    assert res.round_trip["r_net"] == pytest.approx(1.70)


# ---------------------------------------------------------------------------
# BREAKEVEN -> TRAILING: event-driven, on confirmation, never at target-touch
# ---------------------------------------------------------------------------

def trailing_path(target_minute=None):
    """Fill, 1R on minute 1, hold until minute 19 (which dips to 100.50), then
    minute 20 trades through the target and minute 21 falls to 100.50."""
    rows = [FILL, MID] + [HOLD] * 18                # minutes 0..19
    rows[19] =(101.00, 101.25, 100.50, 101.00)
    if target_minute is not None:
        rows[target_minute] = THROUGH_TARGET
    return rows + [THROUGH_TARGET, (102.00, 102.00, 100.50, 100.75)]


SWING_LOW = pivots((2, 100.75, "low"))              # confirms at the close of bar 4: minute 20


def test_a_confirmed_swing_cancels_the_target_and_trails():
    res = run(trailing_path(), SWING_LOW)
    assert states(res) == ["breakeven", "trailing"]
    t = res.transitions[1]
    assert t.ts == T0 + 20 * MIN and t.stop == 100.50 and t.target is None
    assert res.path == "trailing"                   # minute 20's 102.50 fills nothing
    assert res.exit.fill.reference_price == 100.50 and res.exit.fill.fill_price == 100.25


def test_the_trail_starts_at_the_confirming_bar_close_not_its_open():
    """Minute 19 is inside bar 4, the bar that confirms the swing. It trades
    100.50, the trail's price, but the swing isn't known until bar 4 closes."""
    res = run(trailing_path(), SWING_LOW)
    assert res.exit.bar_pos == 21, res


def test_an_unconfirmed_swing_does_not_stop_the_target_filling():
    """The lookahead bug in the original Part 3. The swing at bar 2 has formed
    by minute 12, but confirms only at minute 20. The target trades through
    at minute 12, and the resting limit fills."""
    res = run(trailing_path(target_minute=12), SWING_LOW)
    assert res.path == "planned_exit" and res.exit.bar_pos == 12, res
    assert states(res) == ["breakeven"]


def test_a_swing_confirming_before_breakeven_does_not_trail():
    """Part 3 checks for new swings only once in BREAKEVEN. This one confirms
    at minute 15, while the trade is still IN_POSITION (1R comes at minute 16)."""
    rows = [FILL] + [(100.50, 100.75, 100.50, 100.75)] * 15 + [MID, HOLD, THROUGH_TARGET]
    res = run(rows, pivots((1, 100.50, "low")))
    assert res.path == "planned_exit" and states(res) == ["breakeven"]


def test_a_swing_that_would_not_tighten_the_stop_keeps_the_target():
    """Swing low 100.00 less the 0.25 buffer is 99.75, below the breakeven
    stop. Trailing to it would loosen nothing and cancel the planned exit."""
    rows = trailing_path(target_minute=None)
    res = run(rows, pivots((2, 100.00, "low")))
    assert states(res) == ["breakeven"] and res.path == "planned_exit"
    assert res.exit.bar_pos == 20


def test_a_swing_on_the_wrong_side_is_not_a_trail():
    """A long trails behind swing LOWS; a confirmed swing high moves nothing."""
    res = run(trailing_path(), pivots((2, 101.50, "high")))
    assert states(res) == ["breakeven"] and res.path == "planned_exit"


def test_the_trailing_stop_ratchets_and_never_loosens():
    rows = [FILL, MID] + [HOLD] * 38 + [(101.50, 101.50, 101.00, 101.25)]
    piv = pivots((2, 100.75, "low"),                # -> 100.50 at minute 20
                 (4, 100.50, "low"),                # would loosen to 100.25: ignored
                 (6, 101.25, "low"))                # -> 101.00 at minute 40
    res = run(rows, piv)
    assert [t.stop for t in res.transitions] == [100.00, 100.50, 101.00]
    assert res.path == "trailing" and res.exit.bar_pos == 40
    assert res.exit.fill.fill_price == 100.75


def test_a_short_trails_behind_swing_highs():
    fill = (99.75, 100.25, 99.50, 99.75)
    mid = (99.25, 99.25, 99.00, 99.00)              # 1R for the short
    hold = (98.75, 99.00, 98.50, 98.75)
    rows = [fill, mid] + [hold] * 18 + [(98.75, 99.50, 98.75, 99.25)]
    res = run(rows, pivots((2, 99.25, "high")), spec=SHORT_SPEC)
    assert states(res) == ["breakeven", "trailing"]
    assert res.transitions[1].stop == 99.50
    assert res.path == "trailing" and res.exit.fill.fill_price == 99.75


def test_trailing_uses_the_plans_own_buffer():
    """The trail's offset is the buffer fixed at signal time: plan stop minus
    the Part 0 invalidation for this trigger (0.50 here), not a generic one."""
    plan = TradePlan(entry=100.00, stop=99.00, target=102.00, risk=1.00, rr=2.0,
                     invalidation=99.50, target_source="major_level")
    spec = TradeSpec.from_plan(plan, LONG)
    assert spec.buffer == pytest.approx(0.50)
    res = run(trailing_path(), pivots((2, 101.00, "low")), spec=spec)
    assert res.transitions[1].stop == 100.50


# ---------------------------------------------------------------------------
# closing paths
# ---------------------------------------------------------------------------

def test_the_day_boundary_flattens_a_trailing_position():
    d1 = pd.Timestamp("2026-03-03").date()
    rows = trailing_path()[:21] + [(101.50, 101.75, 101.25, 101.50),
                                   (101.50, 101.75, 101.25, 101.50, d1)]
    res = run(rows, SWING_LOW)
    assert res.path == "day_boundary" and res.exit.bar_pos == 21
    assert res.exit.fill.fill_price == 101.25


def test_an_entry_that_never_fills_has_no_exit():
    res = run([(100.50, 100.75, 100.25, 100.50)] * 15)
    assert res.path == "not_filled" and res.entry.status == "expired"
    assert res.exit is None and res.round_trip is None


def test_data_ending_mid_trade_is_unresolved():
    res = run([FILL, MID, HOLD])
    assert res.path == "unresolved" and res.round_trip is None


def test_r_is_measured_against_the_initial_stop():
    res = run(trailing_path(), SWING_LOW)
    rt = res.round_trip
    assert rt["gross_usd"] == pytest.approx(1.25)   # 100.25 - 100.00, x $5
    assert rt["r_net"] == pytest.approx(-0.05)      # (1.25 - 1.50) / $5 risk


def test_unknown_exit_policies_are_refused():
    with pytest.raises(ValueError, match="order_changes_take_effect"):
        ExitConfig.from_config({"exits": {**EXITS_CFG, "order_changes_take_effect": "same_bar"}})


def test_real_execution_config_has_the_exit_rules():
    import yaml
    with open("config/execution.yaml", encoding="utf-8") as fh:
        x = ExitConfig.from_config(yaml.safe_load(fh))
    assert x.breakeven_at_r == 1.0
