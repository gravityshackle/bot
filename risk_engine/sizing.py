"""Position sizing: exit spec Part 1.

    risk_dollars = min(equity x risk_pct, max_risk_per_trade)
    contracts    = min(floor(risk_dollars / (stop_ticks x tick_value)), max_contracts)

Zero contracts discards the setup; it is never rounded up to 1. The stop
distance is measured to the stop ORDER that will rest (the plan stop rounded
to the grid away from entry, simulated_execution.CostModel.round_stop), so
the size matches the risk the order actually carries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from execution.simulated_execution import CostModel, ExecConfig


@dataclass(frozen=True)
class SizingConfig:
    risk_pct: float
    max_risk_per_trade: float
    max_contracts: int

    @classmethod
    def from_config(cls, risk_cfg: dict) -> "SizingConfig":
        s = risk_cfg["sizing"]
        out = cls(float(s["risk_pct"]), float(s["max_risk_per_trade"]),
                  int(s["max_contracts"]))
        if not (0 < out.risk_pct < 1) or out.max_risk_per_trade <= 0 or out.max_contracts < 1:
            raise ValueError(f"risk sizing config out of range: {out}")
        return out


@dataclass(frozen=True)
class Sizing:
    status: str                   # ok | discarded_zero_contracts
    contracts: int
    risk_budget_usd: float        # min(equity x pct, cap)
    stop_ticks: int               # entry to the rounded stop order
    risk_per_contract_usd: float
    risk_usd: float               # contracts x risk_per_contract_usd


def size_position(entry: float, stop: float, direction: str, *, equity: float,
                  cost: CostModel, cfg: ExecConfig, sizing: SizingConfig) -> Sizing:
    stop_order = cost.round_stop(stop, direction, cfg)
    ticks = abs(cost.to_ticks(entry) - cost.to_ticks(stop_order))
    if ticks == 0:
        raise ValueError(f"stop {stop} rounds onto the entry {entry}: no risk to size")
    budget = min(equity * sizing.risk_pct, sizing.max_risk_per_trade)
    per = ticks * cost.tick_size * cost.point_value           # = ticks x tick value
    n = min(math.floor(budget / per + 1e-9), sizing.max_contracts)
    return Sizing("ok" if n > 0 else "discarded_zero_contracts", n, budget, ticks, per,
                  n * per)
