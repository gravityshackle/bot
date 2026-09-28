"""Phase 4.1 fill simulator: exit spec Parts 2-3, docs/phase4_plan.md section 2.

Walks one setup's orders over the 1-minute bars inside the entry-timeframe
bars it was decided on. Every rule is in config/execution.yaml, and every
case OHLC can't settle (queue position, intrabar order) resolves to the
adverse outcome:

  - A limit fills only when a bar trades THROUGH it by `trade_through_ticks`,
    at the limit price (no gap price improvement). A touch is not a fill.
  - The entry is live from the decision bar's close for K entry-timeframe
    bars that exist, and is cancelled at the day boundary.
  - A stop-market triggers on a touch. A stop that was resting when a bar
    opened beyond it fills at that open. Market orders slip (costs.yaml).
  - A bar reaching both stop and target is a stop. The bar that fills the
    entry can stop out, but cannot reach the target.
  - An open position is flattened at market on the last bar of its trade
    date.
  - Data running out before an order resolves is `unresolved`, never
    `expired` or a flat exit.

All price comparisons are in integer ticks, so float noise such as
70.07 - 0.01 = 70.05999999999999 can't decide a fill. 4.1 takes fixed
stop and target prices; 4.2's breakeven and trailing stops re-run
`simulate_exit` over each stretch where the orders are constant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from data.resample import FREQ_ALIASES
from signal_engine.gates import LONG, SHORT

ENTRY_LIMIT = "entry_limit"
TARGET_LIMIT = "target_limit"
STOP_MARKET = "stop_market_exit"
BREAKER_FLATTEN = "breaker_flatten"
DAY_BOUNDARY_FLATTEN = "day_boundary_flatten"

# Float tolerance for "this price is already on the tick grid", in ticks.
# Numerical, not a trading threshold: it only absorbs representation noise.
_GRID_EPS = 1e-6

_POLICIES = {
    "fills.limit_price_improvement": {False},
    "fills.stop_trigger": {"touch"},
    "fills.same_bar_stop_and_target": {"stop"},
    "rounding.stop": {"away_from_entry"},
    "rounding.target": {"toward_entry"},
    "day_boundary.policy": {"flatten"},
}


def _sign(direction: str) -> int:
    if direction == LONG:
        return 1
    if direction == SHORT:
        return -1
    raise ValueError(f"direction must be {LONG!r} or {SHORT!r}, not {direction!r}")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecConfig:
    k_entry_expire_bars: int
    entry_through_ticks: int
    target_through_ticks: int
    stop_on_fill_bar: bool
    target_on_fill_bar: bool

    @classmethod
    def from_config(cls, cfg: dict) -> "ExecConfig":
        # Only the policies the simulator implements are accepted. Anything
        # else raises rather than quietly running the default.
        for dotted, allowed in _POLICIES.items():
            section, key = dotted.split(".")
            value = cfg[section][key]
            if value not in allowed:
                raise ValueError(f"execution {dotted}: {value!r} is not implemented "
                                 f"(allowed: {sorted(map(str, allowed))})")
        fill_bar = set(cfg["fills"]["fill_bar_exits"])
        if not fill_bar <= {"stop", "target"}:
            raise ValueError(f"execution fills.fill_bar_exits: unknown {fill_bar}")
        out = cls(k_entry_expire_bars=int(cfg["entry"]["k_entry_expire_bars"]),
                  entry_through_ticks=int(cfg["entry"]["trade_through_ticks"]),
                  target_through_ticks=int(cfg["targets"]["trade_through_ticks"]),
                  stop_on_fill_bar="stop" in fill_bar,
                  target_on_fill_bar="target" in fill_bar)
        if out.k_entry_expire_bars < 1 or out.entry_through_ticks < 1 \
                or out.target_through_ticks < 1:
            raise ValueError(f"execution config: counts must be >= 1 ({out})")
        return out


@dataclass(frozen=True)
class CostModel:
    """One instrument's tick grid, point value, fees and market-order slippage."""
    symbol: str
    tick_size: float
    point_value: float
    fee_per_side: float                  # commission + pass-through, per contract
    slippage_ticks: float
    slippage_kinds: frozenset

    @classmethod
    def from_config(cls, costs_cfg: dict, symbol_cfg: dict) -> "CostModel":
        sym = symbol_cfg["symbol"]
        spec = symbol_cfg["contract_spec"]
        tick = float(spec["tick_size"])
        point = float(spec["tick_value"]) / tick
        if "point_value" in spec and not math.isclose(point, float(spec["point_value"]),
                                                      rel_tol=1e-9):
            raise ValueError(f"{sym}: point_value {spec['point_value']} != tick_value / "
                             f"tick_size = {point}")
        slip = costs_cfg["slippage"]
        mult = float(slip["thin_session_multiplier"])
        if mult != 1.0:
            # docs/phase4_plan.md makes 2-4x a sensitivity case, but which
            # sessions count as thin is not defined anywhere yet.
            raise ValueError(f"costs slippage.thin_session_multiplier {mult}: which "
                             "sessions are thin is undefined; only 1.0 is supported")
        row = costs_cfg["symbols"][sym]
        return cls(symbol=sym, tick_size=tick, point_value=point,
                   fee_per_side=float(row["ibkr_commission_side"])
                   + float(row["estimated_exchange_regulatory_side"]),
                   slippage_ticks=float(row["slippage_ticks"]) * mult,
                   slippage_kinds=frozenset(slip["applies_to"]))

    # -- tick grid ----------------------------------------------------------
    def ticks(self, price: float) -> float:
        return price / self.tick_size

    def on_grid(self, price: float) -> bool:
        n = self.ticks(price)
        return abs(n - round(n)) < _GRID_EPS

    def to_ticks(self, price: float, how: str = "exact") -> int:
        """Integer ticks. `how`: exact (must be on grid), floor or ceil.
        A price within float noise of the grid is snapped, never pushed a
        whole tick by floor/ceil."""
        n = self.ticks(price)
        if abs(n - round(n)) < _GRID_EPS:
            return int(round(n))
        if how == "exact":
            raise ValueError(f"{self.symbol}: price {price!r} is not on the tick grid "
                             f"({self.tick_size})")
        return int(math.floor(n) if how == "floor" else math.ceil(n))

    def price(self, ticks: int | float) -> float:
        return round(ticks * self.tick_size, 10)

    def round_stop(self, price: float, direction: str, cfg: ExecConfig) -> float:
        # away from entry: a long's stop rounds down, a short's up
        return self.price(self.to_ticks(price, "floor" if _sign(direction) > 0 else "ceil"))

    def round_target(self, price: float, direction: str, cfg: ExecConfig) -> float:
        # toward entry: a long's target rounds down, a short's up
        return self.price(self.to_ticks(price, "floor" if _sign(direction) > 0 else "ceil"))

    # -- costs --------------------------------------------------------------
    def slippage(self, kind: str) -> float:
        """Adverse price slip for one fill of this kind. Limit orders: 0."""
        return self.slippage_ticks * self.tick_size if kind in self.slippage_kinds else 0.0

    def fees(self, contracts: int) -> float:
        return self.fee_per_side * contracts


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fill:
    """One fill, kept whole so realized slippage can be measured later."""
    kind: str
    ts: pd.Timestamp              # open time of the 1-minute bar it filled in
    side: int                     # +1 buy, -1 sell
    contracts: int
    reference_price: float        # the price the model aimed for (limit, stop, bar close)
    order_price: float | None     # the working order's price; None for a flatten
    fill_price: float
    fees_usd: float


@dataclass(frozen=True)
class EntryResult:
    # filled | expired | cancelled_invalidated | cancelled_day_boundary | unresolved
    status: str
    fill: Fill | None = None
    bar_pos: int | None = None    # positional index into the bars passed in
    detail: str = ""


@dataclass(frozen=True)
class ExitResult:
    status: str                   # target | stop | day_boundary | unresolved
    fill: Fill | None = None
    bar_pos: int | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def entry_window(entry_frame: pd.DataFrame, decision_idx: int, cfg: ExecConfig,
                 timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """[live_from, live_until) for the entry limit, on 1-minute bar open times.

    The order goes in at the decision bar's close and stays live for the
    next K entry-timeframe bars that EXIST in the frame, so a bar missing for
    lack of trades doesn't shorten it. `live_until` is None when the frame
    ends first: the window's end is then unknown.
    """
    tf = pd.Timedelta(FREQ_ALIASES[timeframe])
    ts = entry_frame["ts"]
    live_from = ts.iloc[decision_idx] + tf
    last = decision_idx + cfg.k_entry_expire_bars
    live_until = ts.iloc[last] + tf if last < len(entry_frame) else None
    return live_from, live_until


def _arrays(bars: pd.DataFrame, cost: CostModel) -> dict[str, np.ndarray]:
    out = {}
    for col in ("open", "high", "low", "close"):
        n = bars[col].to_numpy(dtype=float) / cost.tick_size
        r = np.round(n)
        if len(n) and np.abs(n - r).max() >= _GRID_EPS:
            bad = bars.loc[np.abs(n - r) >= _GRID_EPS, col].iloc[0]
            raise ValueError(f"{cost.symbol}: bar {col} {bad!r} is off the tick grid")
        out[col] = r.astype(np.int64)
    return out


def simulate_entry(bars: pd.DataFrame, *, direction: str, limit: float, stop: float,
                   live_from: pd.Timestamp, live_until: pd.Timestamp | None,
                   trade_date: date, contracts: int, cost: CostModel,
                   cfg: ExecConfig) -> EntryResult:
    """Walk the entry limit over 1-minute `bars` (canonical schema plus
    trade_date). `trade_date` is the decision bar's trade date.

    Within a bar the fill is tested before the close, because the price path
    reaches the limit before it can close beyond the stop (the limit sits
    between the market and the stop). Invalidation-before-fill (Part 2) can
    therefore only cancel an order the trade-through rule kept unfilled.
    """
    s = _sign(direction)
    lim = cost.to_ticks(limit)
    stp = cost.to_ticks(stop, "floor" if s > 0 else "ceil")
    if (lim - stp) * s <= 0:
        raise ValueError(f"stop {stop} is not beyond the entry {limit} for a {direction}")
    known_end = live_until is not None and not pd.isna(live_until)

    in_window = bars["ts"] >= live_from
    if known_end:
        in_window &= bars["ts"] < live_until
    pos = np.flatnonzero(in_window.to_numpy())
    if len(pos) == 0:
        return EntryResult("expired" if known_end else "unresolved",
                           detail="no bars in the entry window")
    w = bars.iloc[pos]
    a = _arrays(w, cost)
    through = cfg.entry_through_ticks
    filled = (a["low"] <= lim - through) if s > 0 else (a["high"] >= lim + through)
    invalid = (a["close"] < stp) if s > 0 else (a["close"] > stp)
    new_day = (w["trade_date"] != trade_date).to_numpy()

    first = {k: (int(np.argmax(v)) if v.any() else len(w))
             for k, v in (("fill", filled), ("invalid", invalid), ("day", new_day))}
    # A bar in the next trade date is never live; at the same bar, a fill beats
    # the close-beyond-stop test (see docstring).
    if first["day"] <= min(first["fill"], first["invalid"]) and first["day"] < len(w):
        return EntryResult("cancelled_day_boundary", bar_pos=int(pos[first["day"]]),
                           detail="entry window reached the next trade date")
    if first["fill"] < len(w) and first["fill"] <= first["invalid"]:
        i = first["fill"]
        price = cost.price(lim)
        fill = Fill(ENTRY_LIMIT, w["ts"].iloc[i], s, contracts, price, price,
                    price + s * cost.slippage(ENTRY_LIMIT), cost.fees(contracts))
        return EntryResult("filled", fill, int(pos[i]))
    if first["invalid"] < len(w):
        i = first["invalid"]
        return EntryResult("cancelled_invalidated", bar_pos=int(pos[i]),
                           detail=f"closed at {w['close'].iloc[i]} beyond stop {stop}")
    return EntryResult("expired" if known_end else "unresolved")


# ---------------------------------------------------------------------------
# exits
# ---------------------------------------------------------------------------

def market_exit(ts: pd.Timestamp, reference_price: float, direction: str, contracts: int,
                kind: str, cost: CostModel) -> Fill:
    """Close a position at market: the reference price less adverse slippage."""
    s = _sign(direction)
    return Fill(kind, ts, -s, contracts, reference_price, None,
                cost.price(cost.to_ticks(reference_price)) - s * cost.slippage(kind),
                cost.fees(contracts))


def simulate_exit(bars: pd.DataFrame, entry: EntryResult, *, direction: str,
                  stop: float, target: float | None, cost: CostModel,
                  cfg: ExecConfig) -> ExitResult:
    """Walk a filled position's stop-market and target limit from the fill
    bar onward. `target=None` means no resting target (4.2's trailing state).
    """
    if entry.status != "filled":
        raise ValueError(f"no position: entry status {entry.status!r}")
    s = _sign(direction)
    contracts = entry.fill.contracts
    stp = cost.to_ticks(cost.round_stop(stop, direction, cfg))
    tgt = cost.to_ticks(cost.round_target(target, direction, cfg)) if target is not None else None
    ent = cost.to_ticks(entry.fill.order_price)
    if (ent - stp) * s <= 0:
        raise ValueError(f"stop {stop} is not beyond the entry {entry.fill.order_price}")

    start = entry.bar_pos
    w = bars.iloc[start:]
    a = _arrays(w, cost)
    n = len(w)
    raw = w["raw_symbol"].to_numpy()
    td = w["trade_date"].to_numpy()

    hit_stop = (a["low"] <= stp) if s > 0 else (a["high"] >= stp)
    if not cfg.stop_on_fill_bar:
        hit_stop[0] = False
    if tgt is not None:
        through = cfg.target_through_ticks
        hit_target = (a["high"] >= tgt + through) if s > 0 else (a["low"] <= tgt - through)
        if not cfg.target_on_fill_bar:
            hit_target[0] = False
    else:
        hit_target = np.zeros(n, dtype=bool)
    # last bar of the trade date: the next bar starts a new one
    day_end = np.zeros(n, dtype=bool)
    day_end[:-1] = td[1:] != td[:-1]
    contract_change = np.zeros(n, dtype=bool)
    contract_change[1:] = raw[1:] != raw[0]

    def first(mask):
        return int(np.argmax(mask)) if mask.any() else n

    i_stop, i_tgt, i_day, i_raw = (first(hit_stop), first(hit_target), first(day_end),
                                   first(contract_change))
    i = min(i_stop, i_tgt, i_day)
    if i_raw <= i and i_raw < n:
        raise ValueError(f"contract changed from {raw[0]} to {raw[i_raw]} inside an open "
                         f"trade at {w['ts'].iloc[i_raw]}; positions must close first")
    if i == n:
        return ExitResult("unresolved", detail="data ended with the position open")
    ts = w["ts"].iloc[i]
    if i == i_stop:                              # adverse first, including stop+target
        ref = cost.price(stp)
        # gap: a stop resting when the bar opened beyond it fills at the open.
        # On the fill bar the stop didn't exist yet at the open.
        gapped = i > 0 and ((a["open"][i] < stp) if s > 0 else (a["open"][i] > stp))
        base = cost.price(a["open"][i]) if gapped else ref
        fill = Fill(STOP_MARKET, ts, -s, contracts, ref, ref,
                    base - s * cost.slippage(STOP_MARKET), cost.fees(contracts))
        return ExitResult("stop", fill, start + i, "gapped through" if gapped else "")
    if i == i_tgt:
        ref = cost.price(tgt)
        fill = Fill(TARGET_LIMIT, ts, -s, contracts, ref, ref,
                    ref - s * cost.slippage(TARGET_LIMIT), cost.fees(contracts))
        return ExitResult("target", fill, start + i)
    fill = market_exit(ts, float(w["close"].iloc[i]), direction, contracts,
                       DAY_BOUNDARY_FLATTEN, cost)
    return ExitResult("day_boundary", fill, start + i)


def round_trip(entry_fill: Fill, exit_fill: Fill, *, direction: str, stop: float,
               cost: CostModel) -> dict:
    """P&L of one closed trade. R is measured against the planned risk to the
    stop order's price, so slippage and fees show up as R lost."""
    s = _sign(direction)
    contracts = entry_fill.contracts
    gross = (exit_fill.fill_price - entry_fill.fill_price) * s * cost.point_value * contracts
    fees = entry_fill.fees_usd + exit_fill.fees_usd
    risk = abs(entry_fill.order_price - stop) * cost.point_value * contracts
    return {"gross_usd": gross, "fees_usd": fees, "net_usd": gross - fees,
            "risk_usd": risk, "r_gross": gross / risk, "r_net": (gross - fees) / risk}
