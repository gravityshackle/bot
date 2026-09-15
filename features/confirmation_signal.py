"""S17 -- Soloway two-stage close confirmation (Yellow Alert / Red Alert).

A stricter ALTERNATIVE to S4's buffer breakout, not an addition to it. The spec
is explicit that the two modes must not both run on the same setup: pick one
per symbol via `breakout.mode` and A/B them in backtest. `assert_single_mode()`
exists so that rule is enforced in code rather than remembered.

What makes it stricter is the reference price. S4 confirms against the level
plus a buffer; S17 confirms against the *piercing bar's own extreme*. A bar can
clear a level by a buffer without ever exceeding the high of the bar that first
poked through, so S17 fires later and less often.

Three outcomes from a Yellow Alert, and all three must be handled or setups
leak:
  confirmed  -- a later bar closes beyond the piercing bar's extreme
  failed     -- price closes back through the ORIGINAL level first, which is an
                S8 failed breakout and therefore an opposite-direction setup
  expired    -- neither happened within K_confirm bars; discard as stale
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.schema import Params

LONG, SHORT = "long", "short"


class ModeConflict(RuntimeError):
    """S4 and S17 were both applied to one setup."""


def assert_single_mode(params: Params) -> str:
    mode = str(params.get("breakout.mode"))
    if mode not in ("buffer", "confirmation_signal"):
        raise ValueError(f"breakout.mode must be buffer|confirmation_signal, got {mode!r}")
    return mode


def require_mode(params: Params, wanted: str) -> None:
    """Guard at the entry of each breakout implementation."""
    mode = assert_single_mode(params)
    if mode != wanted:
        raise ModeConflict(
            f"breakout.mode is {mode!r} but {wanted!r} logic was invoked. S17 is "
            "an alternative to S4, never both on the same setup -- pick one per "
            "symbol and A/B them in backtest."
        )


@dataclass(frozen=True)
class Alert:
    pierce_idx: int
    pierce_ts: pd.Timestamp
    direction: str            # the breakout direction being tested
    level: float
    pierce_extreme: float     # high[P] for up, low[P] for down
    outcome: str              # "confirmed" | "failed" | "expired" | "pending"
    resolve_idx: int | None = None
    resolve_ts: pd.Timestamp | None = None
    bars_to_resolve: int | None = None


def scan(bars: pd.DataFrame, level: float, params: Params) -> pd.DataFrame:
    """Walk one level, emitting every Yellow Alert and its resolution.

    A new Yellow Alert is not opened while one is still pending on the same
    side -- the first piercing bar is the reference, and re-arming on each
    subsequent poke would keep resetting the bar the confirmation is measured
    against.
    """
    k = int(params.get("confirmation_signal.k_confirm_bars"))
    high, low, close = (bars["high"].to_numpy(), bars["low"].to_numpy(),
                        bars["close"].to_numpy())
    ts = bars["ts"]
    alerts: list[Alert] = []
    pending: dict[str, int] = {}          # direction -> piercing bar index

    for i in range(len(bars)):
        # resolve anything open first, so a bar can close one alert before
        # being eligible to open another
        for direction in (LONG, SHORT):
            if direction not in pending:
                continue
            p = pending[direction]
            if i == p:
                continue
            extreme = high[p] if direction == LONG else low[p]
            crossed_back = close[i] < level if direction == LONG else close[i] > level
            confirmed = (close[i] > extreme) if direction == LONG else (close[i] < extreme)

            if confirmed:
                alerts.append(Alert(p, ts.iloc[p], direction, level, extreme,
                                    "confirmed", i, ts.iloc[i], i - p))
                del pending[direction]
            elif crossed_back:
                # S17 failure case: back through the original level, which S8
                # then reads as an opposite-direction setup
                alerts.append(Alert(p, ts.iloc[p], direction, level, extreme,
                                    "failed", i, ts.iloc[i], i - p))
                del pending[direction]
            elif i - p >= k:
                alerts.append(Alert(p, ts.iloc[p], direction, level, extreme,
                                    "expired", i, ts.iloc[i], i - p))
                del pending[direction]

        # Yellow Alert: the first bar to CROSS the level (S17's word). Merely
        # being beyond it is not a crossing -- if price is trading above a
        # level, every bar has high > level, and arming on that re-opens an
        # alert the instant the previous one resolves. Measured on real MES 5m
        # data that produced a confirmation roughly every five bars against a
        # single static level. A crossing requires the previous bar to have
        # closed on the other side.
        prev_close = close[i - 1] if i > 0 else None
        if prev_close is None:
            continue
        if LONG not in pending and high[i] > level and prev_close <= level:
            pending[LONG] = i
        if SHORT not in pending and low[i] < level and prev_close >= level:
            pending[SHORT] = i

    for direction, p in pending.items():
        extreme = high[p] if direction == LONG else low[p]
        alerts.append(Alert(p, ts.iloc[p], direction, level, extreme, "pending"))

    if not alerts:
        return pd.DataFrame(columns=["pierce_idx", "pierce_ts", "direction",
                                     "level", "pierce_extreme", "outcome",
                                     "resolve_idx", "resolve_ts",
                                     "bars_to_resolve"])
    return pd.DataFrame([a.__dict__ for a in alerts]).sort_values(
        ["pierce_idx", "direction"]).reset_index(drop=True)


def confirmations(bars: pd.DataFrame, level: float, params: Params) -> pd.DataFrame:
    """Only the confirmed breakouts -- the tradeable S17 output."""
    out = scan(bars, level, params)
    return out[out["outcome"] == "confirmed"].reset_index(drop=True)


def magnitude_factor(bars_to_confirm: int, params: Params) -> float:
    """Scoring input: faster confirmation reads as more conviction.

    min(K_confirm_max / bars_taken, 1.0), per the confidence-scoring spec.
    """
    k = float(params.get("confirmation_signal.k_confirm_bars"))
    if not bars_to_confirm or bars_to_confirm <= 0:
        return 1.0
    return min(k / float(bars_to_confirm), 1.0)
