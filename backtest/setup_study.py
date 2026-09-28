"""Phase 4.3: the setup-level study (docs/phase4_plan.md section 3.1).

Every Stage 1 candidate is simulated on its own through the fill model:
sized by exit spec Part 1 (risk_engine/sizing.py), then run from order
placement to CLOSED (execution/trade_lifecycle.py). There is no portfolio
and no daily limit: gate 6 stays `not_evaluated`, and every setup is sized
against the same starting equity. Each row is the engine's log row (every
gate, the plan, all six score components) joined to its realized outcome
and every fill's reference, order and fill price. That's the unit of record
for A1-A6.

A trade only ever needs its own trade date: entries are cancelled at the
day boundary, and positions are flattened there. So each setup is handed
that day's bars (plus the next day's first bar, to see the boundary),
found through `DayIndex` in O(log n) rather than masked over five years.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.resample import FREQ_ALIASES
from execution.simulated_execution import CostModel, ExecConfig
from execution.trade_lifecycle import (
    BREAKEVEN,
    TRAILING,
    ExitConfig,
    TradeSpec,
    run_trade,
)
from risk_engine.sizing import SizingConfig, size_position
from signal_engine.gates import GateReport

_NAN = math.nan
OUTCOME_COLUMNS = [
    "pattern_bars", "invalidation", "risk_budget_usd", "stop_ticks", "contracts", "stop_order", "target_order",
    "entry_status", "entry_ts", "entry_reference", "entry_order", "entry_fill",
    "path", "exit_kind", "exit_ts", "exit_reference", "exit_order", "exit_fill",
    "exit_detail", "breakeven_ts", "trailing_ts", "stop_moves",
    "gross_usd", "fees_usd", "net_usd", "risk_usd", "r_gross", "r_net", "minutes_held",
]


class DayIndex:
    """Positions where each trade date starts, for O(log n) day slicing."""

    def __init__(self, bars: pd.DataFrame):
        td = bars["trade_date"].to_numpy()
        self._starts = np.flatnonzero(td[1:] != td[:-1]) + 1
        self._n = len(bars)

    def next_day_start(self, pos: int) -> int:
        """First position after `pos` in a later trade date (len if none)."""
        k = np.searchsorted(self._starts, pos, side="right")
        return int(self._starts[k]) if k < len(self._starts) else self._n


def simulate_setup(report: GateReport, *, bars: pd.DataFrame, days: DayIndex,
                   entry_frame: pd.DataFrame, pivots: pd.DataFrame, cost: CostModel,
                   cfg: ExecConfig, xcfg: ExitConfig, sizing: SizingConfig,
                   equity: float, timeframe: str) -> dict:
    """The outcome columns for one Stage 1 candidate."""
    if not report.is_candidate:
        raise ValueError("only Stage 1 candidates are simulated; a failed gate "
                         "is never traded")
    c, plan = report.candidate, report.plan
    spec = TradeSpec.from_plan(plan, c.direction)
    out = dict.fromkeys(OUTCOME_COLUMNS, _NAN)
    out.update(entry_status=None, path=None, exit_kind=None, exit_detail=None,
               entry_ts=pd.NaT, exit_ts=pd.NaT, breakeven_ts=pd.NaT, trailing_ts=pd.NaT)

    z = size_position(spec.entry, spec.stop, c.direction, equity=equity, cost=cost,
                      cfg=cfg, sizing=sizing)
    out.update(pattern_bars=tuple(c.pattern_bars), invalidation=plan.invalidation,
               risk_budget_usd=z.risk_budget_usd, stop_ticks=z.stop_ticks,
               contracts=z.contracts,
               stop_order=cost.round_stop(spec.stop, c.direction, cfg),
               target_order=cost.round_target(spec.target, c.direction, cfg))
    if z.contracts == 0:
        out["path"] = "discarded_size"
        return out

    # this trade's day: from the decision bar's close to the next trade date's first bar
    live_from = entry_frame["ts"].iloc[c.decision_idx] + pd.Timedelta(FREQ_ALIASES[timeframe])
    lo = int(bars["ts"].searchsorted(live_from, side="left"))
    decision_day = entry_frame["trade_date"].iloc[c.decision_idx]
    if lo < len(bars) and bars["trade_date"].iloc[lo] == decision_day:
        hi = days.next_day_start(lo)
    else:
        hi = lo                                    # the decision was the day's last bar
    window = bars.iloc[lo:min(hi + 1, len(bars))].reset_index(drop=True)

    res = run_trade(window, entry_frame, pivots, spec, decision_idx=c.decision_idx,
                    contracts=z.contracts, cost=cost, cfg=cfg, xcfg=xcfg,
                    timeframe=timeframe)
    out.update(entry_status=res.entry.status, path=res.path)
    if res.entry.fill is not None:
        f = res.entry.fill
        out.update(entry_ts=f.ts, entry_reference=f.reference_price,
                   entry_order=f.order_price, entry_fill=f.fill_price)
    for t in res.transitions:
        if t.to_state == BREAKEVEN and pd.isna(out["breakeven_ts"]):
            out["breakeven_ts"] = t.ts
        if t.to_state == TRAILING and pd.isna(out["trailing_ts"]):
            out["trailing_ts"] = t.ts
    out["stop_moves"] = len(res.transitions)
    if res.exit is not None and res.exit.fill is not None:
        f = res.exit.fill
        out.update(exit_kind=f.kind, exit_ts=f.ts, exit_reference=f.reference_price,
                   exit_order=f.order_price, exit_fill=f.fill_price,
                   exit_detail=res.exit.detail,
                   minutes_held=(f.ts - res.entry.fill.ts) / pd.Timedelta(minutes=1))
    if res.round_trip is not None:
        out.update({k: res.round_trip[k] for k in
                    ("gross_usd", "fees_usd", "net_usd", "risk_usd", "r_gross", "r_net")})
    return out


@dataclass(frozen=True)
class StudyConfig:
    cost: CostModel
    cfg: ExecConfig
    xcfg: ExitConfig
    sizing: SizingConfig
    equity: float
    timeframe: str

    @classmethod
    def from_configs(cls, *, costs_cfg: dict, exec_cfg: dict, risk_cfg: dict,
                     symbol_cfg: dict, timeframe: str) -> "StudyConfig":
        return cls(CostModel.from_config(costs_cfg, symbol_cfg),
                   ExecConfig.from_config(exec_cfg), ExitConfig.from_config(exec_cfg),
                   SizingConfig.from_config(risk_cfg),
                   float(risk_cfg["account"]["starting_equity"]), timeframe)


def study(ctx, bars: pd.DataFrame, scfg: StudyConfig, sc=None) -> pd.DataFrame:
    """One row per Stage 1 candidate: the engine's log row plus its outcome.

    `ctx` is the GateContext built from the same 1-minute `bars`.
    """
    from signal_engine import engine, gates, scoring
    sc = sc or scoring.ScoreContext.build(ctx)
    days = DayIndex(bars)
    rows = []
    for c in engine.candidates(ctx):
        r = gates.evaluate(c, ctx)
        if not r.is_candidate:
            continue
        row = engine.report_row(r, ctx, sc)
        row.update(simulate_setup(r, bars=bars, days=days, entry_frame=ctx.entry,
                                  pivots=ctx.pivots, cost=scfg.cost, cfg=scfg.cfg,
                                  xcfg=scfg.xcfg, sizing=scfg.sizing,
                                  equity=scfg.equity, timeframe=scfg.timeframe))
        rows.append(row)
    return pd.DataFrame(rows)
