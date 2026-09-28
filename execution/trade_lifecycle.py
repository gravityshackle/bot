"""Phase 4.2 exit state machine: exit spec Part 3, corrected event-driven version.

    ENTRY_PENDING -> IN_POSITION -> BREAKEVEN -> TRAILING -> CLOSED

- IN_POSITION: the Part 0 stop and the target rest. When price reaches
  `breakeven_at_r` R in favour, the stop moves to the entry price.
- BREAKEVEN: the target still rests. The moment a new swing CONFIRMS on the
  stop's side and would tighten the stop, the target is cancelled and the
  stop trails behind it. Confirmation is the gates' own causal rule (a pivot
  is known at the close of bar `confirmed_idx`), so the decision never waits
  for, or looks at, the target price. The original Part 3 decided "trail or
  take the target" at target-touch, which needs a swing that hasn't
  confirmed yet: lookahead. If the target trades through first, the resting
  limit has simply filled.
- TRAILING: no target. Each later confirmed swing that tightens the stop
  moves it; it never loosens.

Every order change takes effect from the next 1-minute bar after the event
is knowable (config/execution.yaml `exits`). The 4.1 fill rules
(simulated_execution.py) decide each fill, and the day-boundary flatten
closes any state. The stop and its ATR buffer come from gate 4's TradePlan,
whose invalidation follows Part 0's per-trigger table (gates.pattern_bars);
nothing here recomputes them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.resample import FREQ_ALIASES
from execution.simulated_execution import (
    _GRID_EPS,
    CostModel,
    EntryResult,
    ExecConfig,
    ExitResult,
    _arrays,
    _sign,
    entry_window,
    round_trip,
    scan_exit,
    simulate_entry,
)
from signal_engine.gates import TradePlan

IN_POSITION, BREAKEVEN, TRAILING = "in_position", "breakeven", "trailing"

_EXIT_POLICIES = {
    "order_changes_take_effect": {"next_bar"},
    "trail_swing_side": {"stop_side"},
    "trail_buffer": {"initial"},
}
# how a closed trade got there (Part 3's CLOSED log)
_PATH_BY_STOP_STATE = {IN_POSITION: "stopped_out", BREAKEVEN: "scratch", TRAILING: "trailing"}


@dataclass(frozen=True)
class ExitConfig:
    breakeven_at_r: float

    @classmethod
    def from_config(cls, cfg: dict) -> "ExitConfig":
        x = cfg["exits"]
        for key, allowed in _EXIT_POLICIES.items():
            if x[key] not in allowed:
                raise ValueError(f"execution exits.{key}: {x[key]!r} is not implemented "
                                 f"(allowed: {sorted(allowed)})")
        r = float(x["breakeven_at_r"])
        if r <= 0:
            raise ValueError(f"execution exits.breakeven_at_r must be > 0, not {r}")
        return cls(breakeven_at_r=r)


@dataclass(frozen=True)
class TradeSpec:
    """The orders one setup trades, straight from its gate-4 plan."""
    direction: str
    entry: float
    stop: float                   # Part 0 invalidation -/+ the ATR buffer
    target: float
    invalidation: float           # Part 0 extreme, before the buffer

    @property
    def buffer(self) -> float:
        """The ATR buffer fixed at signal time, reused for every trailing stop."""
        return abs(self.stop - self.invalidation)

    @classmethod
    def from_plan(cls, plan: TradePlan, direction: str) -> "TradeSpec":
        if any(math.isnan(v) for v in (plan.entry, plan.stop, plan.target, plan.invalidation)):
            raise ValueError(f"trade plan is incomplete: {plan}")
        return cls(direction, plan.entry, plan.stop, plan.target, plan.invalidation)


@dataclass(frozen=True)
class Transition:
    ts: pd.Timestamp              # open of the 1-minute bar the new orders apply from
    from_state: str
    to_state: str
    stop: float
    target: float | None
    reason: str


@dataclass(frozen=True)
class TradeResult:
    # not_filled | stopped_out | scratch | planned_exit | trailing | day_boundary | unresolved
    path: str
    entry: EntryResult
    exit: ExitResult | None
    transitions: tuple[Transition, ...] = field(default_factory=tuple)
    round_trip: dict | None = None


def _ceil_ticks(x: float) -> int:
    return int(round(x)) if abs(x - round(x)) < _GRID_EPS else math.ceil(x)


def _trail_candidates(pivots: pd.DataFrame, entry_frame: pd.DataFrame, bars: pd.DataFrame,
                      *, side_kind: str, from_frame_idx: int, tf: pd.Timedelta
                      ) -> list[tuple[int, float, str]]:
    """(first 1m bar the swing is known at, swing price, label), in the order
    they become known. Only swings on the stop's side that formed at or after
    the entry's bar."""
    p = pivots[(pivots["kind"] == side_kind) & (pivots["idx"] >= from_frame_idx)]
    if p.empty:
        return []
    # known at the CLOSE of the confirming bar, never at the pivot's own bar
    known = entry_frame["ts"].iloc[p["confirmed_idx"].to_numpy(dtype=int)] + tf
    at = bars["ts"].searchsorted(known, side="left")
    rows = sorted(zip(at.tolist(), p["price"].tolist(), p["idx"].tolist()))
    return [(int(k), float(price), f"swing {side_kind} at entry bar {int(i)}")
            for k, price, i in rows]


def run_trade(bars: pd.DataFrame, entry_frame: pd.DataFrame, pivots: pd.DataFrame,
              spec: TradeSpec, *, decision_idx: int, contracts: int, cost: CostModel,
              cfg: ExecConfig, xcfg: ExitConfig, timeframe: str) -> TradeResult:
    """One setup from order placement to CLOSED, on 1-minute `bars`.

    `entry_frame` is the entry-timeframe frame the setup was decided on and
    `pivots` its S1 swings (structure.find_pivots, carrying confirmed_idx).
    """
    s = _sign(spec.direction)
    tf = pd.Timedelta(FREQ_ALIASES[timeframe])
    stop0 = cost.round_stop(spec.stop, spec.direction, cfg)
    target0 = cost.round_target(spec.target, spec.direction, cfg)
    live_from, live_until = entry_window(entry_frame, decision_idx, cfg, timeframe)
    entry = simulate_entry(bars, direction=spec.direction, limit=spec.entry, stop=stop0,
                           live_from=live_from, live_until=live_until,
                           trade_date=entry_frame["trade_date"].iloc[decision_idx],
                           contracts=contracts, cost=cost, cfg=cfg)
    if entry.status != "filled":
        return TradeResult("not_filled", entry, None)

    # A position never outlives its trade date: keep that day plus one bar,
    # so the scan can see the day end.
    td = bars["trade_date"].to_numpy()
    day = np.flatnonzero(td[entry.bar_pos:] != td[entry.bar_pos])
    end = entry.bar_pos + int(day[0]) + 1 if len(day) else len(bars)
    bars = bars.iloc[:end]

    ent_t = cost.to_ticks(entry.fill.order_price)
    stop_t = cost.to_ticks(stop0)
    tgt_t: int | None = cost.to_ticks(target0)
    # 1R in whole ticks, rounded so it is never reached early
    one_r_t = ent_t + s * _ceil_ticks(xcfg.breakeven_at_r * (ent_t - stop_t) * s)
    fill_frame_idx = int(entry_frame["ts"].searchsorted(entry.fill.ts, side="right")) - 1
    swings = _trail_candidates(pivots, entry_frame, bars,
                               side_kind="low" if s > 0 else "high",
                               from_frame_idx=fill_frame_idx, tf=tf)

    state, pos, fill_bar = IN_POSITION, entry.bar_pos, True
    si = 0                                   # next swing not yet looked at
    transitions: list[Transition] = []

    def scan(start, first_is_fill):
        return scan_exit(bars, start, direction=spec.direction, stop_ticks=stop_t,
                         target_ticks=tgt_t, contracts=contracts, cost=cost, cfg=cfg,
                         fill_bar=first_is_fill)

    def move(at, to_state, reason):
        transitions.append(Transition(bars["ts"].iloc[at], state, to_state,
                                      cost.price(stop_t),
                                      cost.price(tgt_t) if tgt_t is not None else None,
                                      reason))

    while True:
        ex = scan(pos, fill_bar)
        if state == IN_POSITION:
            # first bar reaching 1R, never the fill bar (its high may precede the fill)
            a = _arrays(bars.iloc[pos:], cost)
            reached = (a["high"] >= one_r_t) if s > 0 else (a["low"] <= one_r_t)
            if fill_bar:
                reached[0] = False
            j = pos + int(np.argmax(reached)) if reached.any() else None
            if ex.status != "unresolved" and (j is None or ex.bar_pos <= j):
                break                                  # exited under the original orders
            if j is None or j + 1 >= len(bars):
                break                                  # unresolved
            state_to, pos, fill_bar = BREAKEVEN, j + 1, False
            stop_t = ent_t
            move(pos, state_to, f"{xcfg.breakeven_at_r:g}R reached at {bars['ts'].iloc[j]}")
            state = state_to
            continue

        # BREAKEVEN / TRAILING: the next swing, known from bar `pos` onward, that
        # would tighten the stop. Each swing is looked at once: one known before
        # `pos` was never "new" here, and one that doesn't tighten the stop now
        # never will, since the stop only tightens.
        nxt = None
        while si < len(swings):
            k, price, label = swings[si]
            if k >= len(bars):
                break
            si += 1
            if k < pos:
                continue
            trail_t = cost.to_ticks(cost.round_stop(price - s * spec.buffer,
                                                    spec.direction, cfg))
            if (trail_t - stop_t) * s > 0:
                nxt = (k, trail_t, label)
                break
        if nxt is None or (ex.status != "unresolved" and ex.bar_pos < nxt[0]):
            break
        k, trail_t, label = nxt
        pos, stop_t, tgt_t = k, trail_t, None
        move(pos, TRAILING, f"{label} confirmed")
        state = TRAILING

    if ex.status == "unresolved":
        return TradeResult("unresolved", entry, ex, tuple(transitions))
    path = {"target": "planned_exit", "day_boundary": "day_boundary"}.get(
        ex.status) or _PATH_BY_STOP_STATE[state]
    rt = round_trip(entry.fill, ex.fill, direction=spec.direction, stop=stop0, cost=cost)
    return TradeResult(path, entry, ex, tuple(transitions), rt)
