"""Trigger detection: S4 breakout, S7 rejection, S8 failed breakout,
S9 breakout/retest, S10 range reclaim, S11 momentum, S19 three-tail,
S20 engulfing. Plus the S14 trend filter they depend on.

Two structural points.

**Level-dependent vs level-free.** Rejection, engulfing and three-tail are pure
candle geometry -- they say nothing about *where* they happened. Breakout,
failed breakout, retest, range reclaim and momentum are defined against a
specific price level. The level-dependent ones therefore take a single `level`
and are run once per active level by the caller; whether a level-free pattern
occurred somewhere meaningful is Stage 1 gate 2's job, not this module's.

**Everything fires on a CLOSED bar.** S4 is explicit that an intrabar wick
through a level is not a breakout and only a close beyond counts, and that
distinction is what makes S8's failed breakout meaningful. `breakouts()` keeps
both, so the difference is never lost.

**A breakout is a transition, not a state.** See `breakouts()` -- read as a
per-bar predicate, S4 would make every bar of an uptrend a breakout of every
level beneath it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.schema import (
    BODY,
    BODY_RATIO,
    CLV,
    LOWER_WICK,
    UPPER_WICK,
    Params,
    tick_size,
)

LONG, SHORT = "long", "short"


@dataclass(frozen=True)
class TriggerEvent:
    idx: int
    ts: pd.Timestamp
    kind: str            # "rejection" | "engulfing" | "three_tail" | ...
    direction: str       # LONG | SHORT -- the trade it implies, not the move
    level: float = float("nan")
    price: float = float("nan")
    meta: dict = field(default_factory=dict)


def _events_frame(events: list[TriggerEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["idx", "ts", "kind", "direction", "level",
                                     "price", "meta"])
    return pd.DataFrame([{
        "idx": e.idx, "ts": e.ts, "kind": e.kind, "direction": e.direction,
        "level": e.level, "price": e.price, "meta": e.meta,
    } for e in events])


# --------------------------------------------------------------------------
# shared tolerances (S4 buffer, S6 test zone, S19 cluster)
# --------------------------------------------------------------------------

def _band(atr: pd.Series, params: Params, min_ticks_key: str,
          atr_key: str) -> pd.Series:
    """The spec's recurring max(N ticks, k x ATR) construction."""
    ticks = int(params.get(min_ticks_key)) * tick_size(params)
    return np.maximum(ticks, float(params.get(atr_key)) * atr)


def breakout_buffer(atr: pd.Series, params: Params) -> pd.Series:
    return _band(atr, params, "breakout.buffer_min_ticks",
                 "breakout.buffer_atr_multiple")


def test_zone(atr: pd.Series, params: Params) -> pd.Series:
    return _band(atr, params, "test_zone.min_ticks", "test_zone.atr_multiple")


def cluster_tolerance(atr: pd.Series, params: Params) -> pd.Series:
    return _band(atr, params, "three_tail.cluster_tolerance_min_ticks",
                 "three_tail.cluster_tolerance_atr_multiple")


# --------------------------------------------------------------------------
# S14 trend filter
# --------------------------------------------------------------------------

def trend_bias(htf: pd.DataFrame, params: Params) -> pd.Series:
    """bullish / bearish / neutral on the higher timeframe.

    S14: bullish needs close > EMA AND the EMA sloping up over the lookback --
    a flat EMA is explicitly not a trend. Anything that fails both directions
    is neutral. What the engine DOES with neutral is unresolved (see
    docs/open_questions.md); this function only reports it.
    """
    period = int(params.get("trend.ema_period"))
    lookback = int(params.get("trend.slope_lookback_bars"))
    ema = htf["close"].ewm(span=period, adjust=False).mean()
    rising = ema > ema.shift(lookback)
    falling = ema < ema.shift(lookback)
    out = pd.Series("neutral", index=htf.index, dtype="object")
    out[(htf["close"] > ema) & rising] = "bullish"
    out[(htf["close"] < ema) & falling] = "bearish"
    out[ema.isna() | ema.shift(lookback).isna()] = "unknown"
    return out.rename("trend_bias")


# --------------------------------------------------------------------------
# S4 breakout / breakdown
# --------------------------------------------------------------------------

def breakouts(bars: pd.DataFrame, level: float, atr: pd.Series,
              params: Params) -> pd.DataFrame:
    """Classify each bar against one level, as both STATE and EVENT.

    S4 reads literally as "a close beyond the level by a buffer", which as a
    per-bar predicate makes every bar of an uptrend a breakout of every level
    beneath it -- and then S8 manufactures a failed breakout out of any
    oscillation. That cannot be the intent: S8 says price "closes beyond a
    level, THEN within K bars closes back", and S9 says a confirmed breakout is
    followed by a pullback. Both describe a discrete transition.

    So this returns both readings and they are used for different jobs:

      up_close / down_close   STATE  -- is this bar closed beyond the level
      up_break / down_break   EVENT  -- the first close beyond after not being
                                        beyond; this is what S8/S9 consume

    The first bar can never be an event: with no prior bar there is no
    transition to observe, and treating a series that simply opens beyond a
    level as a breakout of it would invent a signal out of where the data
    happens to start.
    """
    buf = breakout_buffer(atr, params)
    up_close = (bars["close"] > (level + buf)).fillna(False)
    down_close = (bars["close"] < (level - buf)).fillna(False)

    prev_up = up_close.shift(1)
    prev_down = down_close.shift(1)
    up_break = up_close & (prev_up == False)      # noqa: E712 -- NaN must not pass
    down_break = down_close & (prev_down == False)  # noqa: E712

    pierced_up = (bars["high"] > level) & ~up_close
    pierced_down = (bars["low"] < level) & ~down_close
    return pd.DataFrame({
        "up_close": up_close,
        "down_close": down_close,
        "up_break": up_break.fillna(False),
        "down_break": down_break.fillna(False),
        "wick_up_only": pierced_up.fillna(False),
        "wick_down_only": pierced_down.fillna(False),
        "buffer": buf,
    }, index=bars.index)


def in_test_zone(bars: pd.DataFrame, level: float, atr: pd.Series,
                 params: Params) -> pd.Series:
    """S6. Note this only starts the bot WATCHING -- it never triggers entry."""
    tol = test_zone(atr, params)
    dist = (bars[["high", "low"]].sub(level).abs().min(axis=1))
    touching = (bars["low"] <= level) & (bars["high"] >= level)
    return (touching | (dist <= tol)).fillna(False)


# --------------------------------------------------------------------------
# S7 rejection candle
# --------------------------------------------------------------------------

def rejection(bars: pd.DataFrame, feats: pd.DataFrame, atr: pd.Series,
              params: Params) -> pd.Series:
    """LONG / SHORT / None per bar (S7).

    All three conditions must hold: CLV threshold, wick at least
    wick_body_ratio times the body, and a body large enough in ATR terms to
    not be a doji.
    """
    clv_t = float(params.get("rejection.clv_threshold"))
    ratio = float(params.get("rejection.wick_body_ratio"))
    min_body = float(params.get("rejection.min_body_atr_multiple")) * atr

    body = feats[BODY]
    big_enough = body >= min_body
    bull = (feats[CLV] >= clv_t) & (feats[LOWER_WICK] >= ratio * body) & big_enough
    bear = (feats[CLV] <= -clv_t) & (feats[UPPER_WICK] >= ratio * body) & big_enough

    out = pd.Series([None] * len(bars), index=bars.index, dtype="object")
    out[bull.fillna(False)] = LONG
    out[bear.fillna(False)] = SHORT
    return out.rename("rejection")


# --------------------------------------------------------------------------
# S20 engulfing
# --------------------------------------------------------------------------

def engulfing(bars: pd.DataFrame, feats: pd.DataFrame, atr: pd.Series,
              params: Params) -> pd.Series:
    """LONG / SHORT / None per bar (S20).

    Body-to-body, not wick-to-wick. Two strength conditions, and S20 is
    explicit that the combination is not optional:

      relative  body[i]   >= strength_multiplier x body[i-1]
      absolute  body[i-1] >= min_prior_body_atr_multiple x ATR

    The relative test alone has no floor. 1.3x of a one-tick doji is still
    almost nothing, so any ordinary bar following a doji "engulfs" it -- and on
    real data that degenerate case, not genuine conviction, was the majority of
    all engulfing firings. The absolute floor is on the PRIOR body: what has to
    be meaningful is the body being swallowed, since that is what makes the
    reversal informative.
    """
    mult = float(params.get("engulfing.strength_multiplier"))
    min_prior = float(params.get("engulfing.min_prior_body_atr_multiple")) * atr
    o, c = bars["open"], bars["close"]
    po, pc = o.shift(1), c.shift(1)
    body, pbody = feats[BODY], feats[BODY].shift(1)
    strong = (body >= mult * pbody) & (pbody >= min_prior)

    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po) & strong
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po) & strong

    out = pd.Series([None] * len(bars), index=bars.index, dtype="object")
    out[bull.fillna(False)] = LONG
    out[bear.fillna(False)] = SHORT
    return out.rename("engulfing")


# --------------------------------------------------------------------------
# S19 three tail theory
# --------------------------------------------------------------------------

def tail_bars(bars: pd.DataFrame, feats: pd.DataFrame, atr: pd.Series,
              params: Params) -> pd.DataFrame:
    """Which bars qualify as upper-side / lower-side tail bars (S19)."""
    ratio = float(params.get("three_tail.wick_body_ratio"))
    max_body = float(params.get("three_tail.max_body_atr_multiple")) * atr
    body = feats[BODY]
    small_body = body <= max_body
    return pd.DataFrame({
        "upper": ((feats[UPPER_WICK] >= ratio * body) & small_body).fillna(False),
        "lower": ((feats[LOWER_WICK] >= ratio * body) & small_body).fillna(False),
    }, index=bars.index)


def three_tail(bars: pd.DataFrame, feats: pd.DataFrame, atr: pd.Series,
               params: Params) -> pd.DataFrame:
    """Clusters of same-side tail bars aligned at one price (S19).

    Alignment is what separates a real cluster from three unrelated wicks: the
    extremes must fall within cluster_tolerance of each other. Fires on the bar
    that completes the cluster, using only the lookback window ending there.
    """
    n = int(params.get("three_tail.lookback_bars"))
    need = int(params.get("three_tail.min_tails_required"))
    tol = cluster_tolerance(atr, params)
    tails = tail_bars(bars, feats, atr, params)

    events: list[TriggerEvent] = []
    highs, lows = bars["high"].to_numpy(), bars["low"].to_numpy()
    for i in range(n - 1, len(bars)):
        window = slice(i - n + 1, i + 1)
        t = float(tol.iloc[i])
        if np.isnan(t):
            continue
        for side, prices, direction in (("upper", highs, SHORT),
                                        ("lower", lows, LONG)):
            idxs = np.flatnonzero(tails[side].to_numpy()[window]) + (i - n + 1)
            if len(idxs) < need or i not in idxs:
                continue
            anchor = prices[i]
            aligned = [j for j in idxs if abs(prices[j] - anchor) <= t]
            if len(aligned) >= need:
                events.append(TriggerEvent(
                    idx=i, ts=bars["ts"].iloc[i], kind="three_tail",
                    direction=direction, level=float(np.mean(prices[aligned])),
                    price=float(bars["close"].iloc[i]),
                    meta={"side": side, "count": len(aligned),
                          "bars": [int(j) for j in aligned],
                          "tolerance": t}))
    return _events_frame(events)


# --------------------------------------------------------------------------
# S8 failed breakout / S10 range reclaim
# --------------------------------------------------------------------------

def failed_breakouts(bars: pd.DataFrame, level: float, atr: pd.Series,
                     params: Params, *, window_key="failed_breakout.window_bars",
                     kind="failed_breakout", only: str | None = None
                     ) -> pd.DataFrame:
    """Close beyond a level, then back through it within K bars (S8).

    The resulting trade is the OPPOSITE direction to the breakout: a failed
    breakout above resistance is a short.

    `only` restricts which breakout leg counts: "up", "down", or None for both.
    A discrete S/R level can fail from either side, so S8 leaves it None; a
    range boundary cannot, which is what `range_reclaims()` uses it for.

    S10 range reclaim is the same math applied to a range edge, so it shares
    this implementation rather than duplicating it.
    """
    if only not in (None, "up", "down"):
        raise ValueError(f"only must be None|'up'|'down', got {only!r}")
    k = int(params.get(window_key))
    b = breakouts(bars, level, atr, params)
    events: list[TriggerEvent] = []
    close = bars["close"].to_numpy()
    up_ok = only in (None, "up")
    down_ok = only in (None, "down")

    for i in range(len(bars)):
        if up_ok and b["up_break"].iloc[i]:
            for j in range(i + 1, min(i + 1 + k, len(bars))):
                if close[j] < level:
                    events.append(TriggerEvent(
                        idx=j, ts=bars["ts"].iloc[j], kind=kind, direction=SHORT,
                        level=level, price=float(close[j]),
                        meta={"breakout_idx": i, "bars_to_fail": j - i}))
                    break
        elif down_ok and b["down_break"].iloc[i]:
            for j in range(i + 1, min(i + 1 + k, len(bars))):
                if close[j] > level:
                    events.append(TriggerEvent(
                        idx=j, ts=bars["ts"].iloc[j], kind=kind, direction=LONG,
                        level=level, price=float(close[j]),
                        meta={"breakout_idx": i, "bars_to_fail": j - i}))
                    break
    return _events_frame(events)


def range_reclaims(bars: pd.DataFrame, edge: float, atr: pd.Series,
                   params: Params, side: str) -> pd.DataFrame:
    """S10 -- S8's math applied to a range boundary, on that boundary's side.

    `side` is which boundary `edge` is: "high" or "low". It is required, not
    inferred, because nothing about a bare price says which edge it is.

    Sharing S8's mechanism does not mean accepting either direction. A reclaim
    of the range HIGH is an escape above it that reverts back below (a short);
    a reclaim of the range LOW is an escape below that reverts back above (a
    long). A close below the range high without ever exceeding it is ordinary
    trade inside the range, not a reclaim of anything -- counting it roughly
    doubled the real reclaim count on live data.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    return failed_breakouts(bars, edge, atr, params,
                            window_key="range_reclaim.window_bars",
                            kind="range_reclaim",
                            only="up" if side == "high" else "down")


# --------------------------------------------------------------------------
# S9 breakout / retest
# --------------------------------------------------------------------------

def breakout_retests(bars: pd.DataFrame, level: float, atr: pd.Series,
                     params: Params, rejection_dir: pd.Series) -> pd.DataFrame:
    """Confirmed breakout, pullback into the test zone, rejection there (S9).

    Invalidated if price closes back through the level in the failure
    direction before the retest -- that is an S8 failed breakout instead, and
    the two must not both fire on the same sequence.
    """
    k_max = int(params.get("breakout_retest.max_bars_to_retest"))
    b = breakouts(bars, level, atr, params)
    zone = in_test_zone(bars, level, atr, params)
    close = bars["close"].to_numpy()
    events: list[TriggerEvent] = []

    for i in range(len(bars)):
        for up in (True, False):
            if not (b["up_break"].iloc[i] if up else b["down_break"].iloc[i]):
                continue
            want = LONG if up else SHORT
            for j in range(i + 1, min(i + 1 + k_max, len(bars))):
                failed = close[j] < level if up else close[j] > level
                if failed:
                    break            # S8 territory, not a retest
                if zone.iloc[j] and rejection_dir.iloc[j] == want:
                    events.append(TriggerEvent(
                        idx=j, ts=bars["ts"].iloc[j], kind="breakout_retest",
                        direction=want, level=level, price=float(close[j]),
                        meta={"breakout_idx": i, "bars_to_retest": j - i}))
                    break
    return _events_frame(events)


# --------------------------------------------------------------------------
# S11 momentum continuation
# --------------------------------------------------------------------------

def momentum_continuation(bars: pd.DataFrame, feats: pd.DataFrame,
                          minor_level: float, params: Params,
                          bias: pd.Series, volume_expanded: pd.Series
                          ) -> pd.DataFrame:
    """Requires ALL of S11: trend aligned, strong body, close beyond a minor
    level in the trend direction, and volume expansion.

    Unlike the reversal triggers this one is explicitly allowed to fire off a
    MINOR level (S11's own definition), which is why Stage 1 gate 2 exempts it.

    **An event, not a state.** S11 is explicit that this fires on the bar where
    the condition first becomes true and must not re-fire while it stays true,
    using the identical shift-a-state pattern as `breakouts()`. Written as a
    bare predicate, `close > minor_level` remains true for every bar that stays
    beyond the level, so any later strong-bodied, volume-expanded bar re-fires
    it indefinitely. Measured on real data that produced 2,000-4,300 firings
    per instrument against 150-460 for the other selective triggers -- the
    order-of-magnitude gap was the symptom.

    So the STATE is "closed beyond the minor level" and the event is its
    transition, exactly as S4 treats a level; the body, volume and trend
    filters are then applied to the transition bar. As with `breakouts()`, the
    first bar can never be an event: there is no prior bar to transition from.
    """
    min_ratio = float(params.get("momentum.min_body_ratio"))
    need_trend = bool(params.get("momentum.requires_trend_alignment"))
    need_vol = bool(params.get("momentum.requires_volume_expansion"))

    strong = feats[BODY_RATIO] >= min_ratio
    vol_ok = volume_expanded if need_vol else pd.Series(True, index=bars.index)

    up = (bars["close"] > minor_level).fillna(False)
    down = (bars["close"] < minor_level).fillna(False)
    # noqa: E712 below -- NaN must not pass as "was not beyond"
    up_cross = up & (up.shift(1) == False)      # noqa: E712
    down_cross = down & (down.shift(1) == False)  # noqa: E712

    bull_ok = (bias == "bullish") if need_trend else pd.Series(True, index=bars.index)
    bear_ok = (bias == "bearish") if need_trend else pd.Series(True, index=bars.index)

    long_hit = (strong & vol_ok & up_cross & bull_ok).fillna(False)
    short_hit = (strong & vol_ok & down_cross & bear_ok).fillna(False)

    events = [
        TriggerEvent(idx=i, ts=bars["ts"].iloc[i], kind="momentum",
                     direction=LONG if long_hit.iloc[i] else SHORT,
                     level=minor_level, price=float(bars["close"].iloc[i]),
                     meta={"body_ratio": float(feats[BODY_RATIO].iloc[i])})
        for i in range(len(bars)) if long_hit.iloc[i] or short_hit.iloc[i]
    ]
    return _events_frame(events)


# --------------------------------------------------------------------------
# level-free patterns, as one frame
# --------------------------------------------------------------------------

def candle_triggers(bars: pd.DataFrame, feats: pd.DataFrame, atr: pd.Series,
                    params: Params) -> pd.DataFrame:
    """S7, S19 and S20 -- the patterns that need no level to be detected.

    Whether each one occurred AT a level is Stage 1 gate 2's decision.
    """
    out = pd.DataFrame(index=bars.index)
    out["rejection"] = rejection(bars, feats, atr, params)
    out["engulfing"] = engulfing(bars, feats, atr, params)
    tt = three_tail(bars, feats, atr, params)
    out["three_tail"] = pd.Series([None] * len(bars), index=bars.index,
                                  dtype="object")
    if not tt.empty:
        out.loc[tt["idx"].to_numpy(), "three_tail"] = tt["direction"].to_numpy()
    return out
