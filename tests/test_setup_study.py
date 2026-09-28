"""Phase 4.3: position sizing (exit spec Part 1) and the setup-level study.

Sizing: risk = min(equity x risk_pct, max_risk); contracts = floor(risk /
(stop ticks x tick value)), capped; zero contracts DISCARDS the setup. The
stop distance is measured to the stop order that will actually rest (rounded
to the grid away from entry), not the raw plan price.

Study: one row per Stage 1 candidate, joining its realized outcome to the
log row. Each trade sees only its own trade date's bars; the tests pin that
the slicing changes nothing and that no bar after the trade can.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml

from backtest.setup_study import DayIndex, simulate_setup
from execution.simulated_execution import CostModel, ExecConfig
from execution.trade_lifecycle import ExitConfig, TradeSpec, run_trade
from risk_engine.sizing import SizingConfig, size_position
from signal_engine.gates import (
    FAIL,
    LONG,
    NOT_EVALUATED,
    PASS,
    Candidate,
    GateReport,
    GateResult,
    TradePlan,
)
from tests.test_simulated_execution import COSTS_CFG, EXEC_CFG, MIN, SYMBOL_CFG, T0, mbars
from tests.test_trade_lifecycle import EXITS_CFG, FILL, HOLD, MID, THROUGH_TARGET, entry_frame

RISK_CFG = {"account": {"starting_equity": 5000.0},
            "sizing": {"risk_pct": 0.01, "max_risk_per_trade": 50.0, "max_contracts": 1}}


def sizing(**over) -> SizingConfig:
    cfg = copy.deepcopy(RISK_CFG)
    cfg["sizing"].update(over)
    return SizingConfig.from_config(cfg)


COST = CostModel.from_config(COSTS_CFG, SYMBOL_CFG)          # tick 0.25, $1.25/tick
CFG = ExecConfig.from_config(EXEC_CFG)
XCFG = ExitConfig.from_config({"exits": EXITS_CFG})


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------

def test_risk_budget_is_the_smaller_of_the_percentage_and_the_cap():
    big = size_position(100.00, 99.00, LONG, equity=5000, cost=COST, cfg=CFG,
                        sizing=sizing(max_contracts=100))
    small = size_position(100.00, 99.00, LONG, equity=2000, cost=COST, cfg=CFG,
                          sizing=sizing(max_contracts=100))
    assert big.risk_budget_usd == 50.0 and small.risk_budget_usd == 20.0


def test_contracts_are_floored_never_rounded():
    """11 ticks x $1.25 = $13.75 a contract; $50 buys 3.64 -> 3, not 4."""
    z = size_position(100.00, 97.25, LONG, equity=5000, cost=COST, cfg=CFG,
                      sizing=sizing(max_contracts=100))
    assert z.stop_ticks == 11 and z.contracts == 3
    assert z.risk_usd == pytest.approx(3 * 13.75)


def test_contracts_are_capped():
    z = size_position(100.00, 99.00, LONG, equity=5000, cost=COST, cfg=CFG, sizing=sizing())
    assert z.contracts == 1 and z.status == "ok"


def test_a_setup_too_wide_for_the_budget_is_discarded_not_rounded_up():
    """41 ticks x $1.25 = $51.25 > $50."""
    z = size_position(100.00, 89.75, LONG, equity=5000, cost=COST, cfg=CFG, sizing=sizing())
    assert z.contracts == 0 and z.status == "discarded_zero_contracts"


def test_the_stop_distance_is_to_the_rounded_stop_order():
    """Plan stop 99.10 rests at 99.00 (away from entry): 4 ticks, $5 a contract,
    so $18 buys 3. The raw 3.6 ticks would say $4.50 and 4 contracts."""
    z = size_position(100.00, 99.10, LONG, equity=1800, cost=COST, cfg=CFG,
                      sizing=sizing(max_contracts=100))
    assert z.stop_ticks == 4 and z.contracts == 3


def test_a_short_sizes_from_its_stop_above():
    z = size_position(100.00, 101.00, "short", equity=5000, cost=COST, cfg=CFG,
                      sizing=sizing(max_contracts=100))
    assert z.stop_ticks == 4 and z.contracts == 10


def test_met_is_bound_by_the_contract_cap_not_the_budget():
    with open("config/symbols/MET.yaml", encoding="utf-8") as fh:
        met = yaml.safe_load(fh)
    with open("config/costs.yaml", encoding="utf-8") as fh:
        c = CostModel.from_config(yaml.safe_load(fh), met)
    z = size_position(2500.00, 2490.00, LONG, equity=5000, cost=c, cfg=CFG, sizing=sizing())
    assert z.stop_ticks == 20 and z.contracts == 1          # $1 a contract; cap binds


def test_real_risk_config_holds_the_spec_numbers():
    with open("config/risk.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    s = SizingConfig.from_config(raw)
    assert (s.risk_pct, s.max_risk_per_trade, s.max_contracts) == (0.01, 50.0, 1)
    assert raw["account"]["starting_equity"] == 5000.0


# ---------------------------------------------------------------------------
# one setup, end to end
# ---------------------------------------------------------------------------

PLAN = TradePlan(entry=100.00, stop=99.00, target=102.00, risk=1.00, rr=2.0,
                 invalidation=99.25, target_source="major_level")


def report(plan=PLAN, gate6=NOT_EVALUATED, gate3=PASS) -> GateReport:
    c = Candidate(kind="rejection", direction=LONG, role="entry", idx=0, decision_idx=0,
                  pattern_bars=(0,))
    results = tuple(GateResult(g, PASS) for g in (1, 2)) + (GateResult(3, gate3),) \
        + tuple(GateResult(g, PASS) for g in (4, 5)) + (GateResult(6, gate6),)
    return GateReport(c, results, plan)


def setup(rows, *, rep=None, sz=None, equity=5000.0, frame=None):
    bars = mbars(rows)
    return simulate_setup(rep or report(), bars=bars, days=DayIndex(bars),
                          entry_frame=frame if frame is not None else entry_frame(),
                          pivots=pd.DataFrame(columns=["idx", "ts", "price", "kind",
                                                       "confirmed_idx", "confirmed_ts"]),
                          cost=COST, cfg=CFG, xcfg=XCFG, sizing=sz or sizing(),
                          equity=equity, timeframe="5min")


def test_a_filled_setup_records_outcome_and_every_price():
    o = setup([FILL, MID, HOLD, THROUGH_TARGET])
    assert o["path"] == "planned_exit" and o["contracts"] == 1
    assert (o["entry_reference"], o["entry_order"], o["entry_fill"]) == (100.0, 100.0, 100.0)
    assert (o["exit_kind"], o["exit_fill"]) == ("target_limit", 102.0)
    assert o["entry_ts"] == T0 and o["exit_ts"] == T0 + 3 * MIN
    assert o["breakeven_ts"] == T0 + 2 * MIN and pd.isna(o["trailing_ts"])
    assert o["net_usd"] == pytest.approx(8.50) and o["r_net"] == pytest.approx(1.70)
    assert o["minutes_held"] == 3
    # two candidates can share bar, kind, level and direction and differ only
    # in the bars that formed them (and so their stop): the row must say which
    assert o["pattern_bars"] == (0,) and o["invalidation"] == 99.25


def test_contracts_scale_pnl_and_fees_but_not_r():
    o = setup([FILL, MID, HOLD, THROUGH_TARGET], sz=sizing(max_contracts=3))
    assert o["contracts"] == 3
    assert o["gross_usd"] == pytest.approx(30.0) and o["fees_usd"] == pytest.approx(4.50)
    assert o["r_net"] == pytest.approx(1.70)


def test_a_discarded_setup_keeps_its_row_but_is_never_simulated():
    o = setup([FILL, MID, HOLD, THROUGH_TARGET], equity=400.0)      # $4 budget < $5
    assert o["path"] == "discarded_size" and o["contracts"] == 0
    assert o["entry_status"] is None and pd.isna(o["r_net"])


def test_only_stage_1_candidates_are_simulated():
    with pytest.raises(ValueError, match="Stage 1"):
        setup([FILL], rep=report(gate3=FAIL))


def test_an_unfilled_setup_records_why():
    o = setup([(100.50, 100.75, 100.25, 100.50)] * 15)
    assert o["path"] == "not_filled" and o["entry_status"] == "expired"
    assert pd.isna(o["exit_ts"]) and pd.isna(o["r_net"])


def test_slicing_to_the_trade_date_matches_the_full_series():
    """Two trade dates of bars; the study hands the trade only its own day
    (plus the next day's first bar, to see the boundary). Same outcome as
    walking the whole series."""
    d1 = pd.Timestamp("2026-03-03").date()
    rows = [FILL, MID] + [HOLD] * 6 + [(101.25, 101.50, 101.00, 101.25, d1)] * 30
    o = setup(rows)
    bars = mbars(rows)
    full = run_trade(bars, entry_frame(), pd.DataFrame(columns=["idx", "ts", "price", "kind",
                                                              "confirmed_idx", "confirmed_ts"]),
                     TradeSpec.from_plan(PLAN, LONG), decision_idx=0, contracts=1, cost=COST,
                     cfg=CFG, xcfg=XCFG, timeframe="5min")
    assert o["path"] == full.path == "day_boundary"
    assert o["exit_ts"] == full.exit.fill.ts and o["exit_fill"] == full.exit.fill.fill_price


def test_bars_after_the_trade_cannot_change_it():
    base = [FILL, MID, HOLD, THROUGH_TARGET]
    a = setup(base + [HOLD] * 5)
    b = setup(base + [(90.00, 110.00, 90.00, 95.00)] * 5)
    keep = ["path", "exit_ts", "exit_fill", "net_usd", "breakeven_ts"]
    assert {k: a[k] for k in keep} == {k: b[k] for k in keep}


def test_day_index_finds_the_first_bar_of_the_next_trade_date():
    d1 = pd.Timestamp("2026-03-03").date()
    bars = mbars([FILL] * 4 + [(100.0, 100.0, 100.0, 100.0, d1)] * 3)
    di = DayIndex(bars)
    assert di.next_day_start(0) == 4 and di.next_day_start(3) == 4
    assert di.next_day_start(4) == 7                                # end of data
