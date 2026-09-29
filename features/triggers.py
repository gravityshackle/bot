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

from features.confirmation import wick_dominates
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
              params: Params, prior: tuple[bool, bool] = (False, False)) -> pd.DataFrame:
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
                                        beyond
      up_start / down_start   EXCURSION -- the first close beyond since price
                                        last closed back through the LEVEL;
                                        this is what S8/S9/S10 consume

    A break can repeat inside one excursion: price closes beyond the buffer,
    drifts back inside it without closing back through the level, and breaks
    again. That is still one excursion, and exit spec Part 0 anchors the stop
    on "the extreme of the failed excursion", so S8/S9 key on its start. Keyed
    on every break, one excursion fired a failed breakout (or a retest) per
    break on the same bar, each later copy with a stop that ignored the
    excursion's first extreme (found on real MES data, 2026-09-28). The close
    that ends an excursion is the one S8 calls a failure: strictly back through
    the level. A series that opens mid-excursion (bar 0 already beyond) counts
    its first bars as that excursion, so a later re-break is not a start.
    `prior` = (up, down) says whether an excursion was already running on the
    bar before `bars` begins (`excursion_before`): a window can open back inside
    the buffer mid-excursion, with nothing in it to show the excursion began.

    The first bar can never be an event: with no prior bar there is no
    transition to observe, and treating a series that simply opens beyond a
    level as a breakout of it would invent a signal out of where the data
    happens to start.
    """
    buf = breakout_buffer(atr, params)
    close = bars["close"].to_numpy(dtype=float)
    up_close, down_close, up_break, down_break, up_start, down_start = \
        _breakout_states(close, buf.to_numpy(dtype=float), level, prior)
    pierced_up = (bars["high"].to_numpy(dtype=float) > level) & ~up_close
    pierced_down = (bars["low"].to_numpy(dtype=float) < level) & ~down_close
    return pd.DataFrame({
        "up_close": up_close,
        "down_close": down_close,
        "up_break": up_break,
        "down_break": down_break,
        "up_start": up_start,
        "down_start": down_start,
        "wick_up_only": pierced_up,
        "wick_down_only": pierced_down,
        "buffer": buf,
    }, index=bars.index)


def _breakout_states(close: np.ndarray, buf: np.ndarray, level: float,
                     prior: tuple[bool, bool] = (False, False)):
    """`breakouts()`'s arithmetic on plain arrays. The detectors call this
    directly: building a DataFrame per level window was most of their cost.
    A comparison against a NaN buffer (ATR not seeded) is False."""
    up_close = close > level + buf
    down_close = close < level - buf
    up_break = np.zeros(len(close), dtype=bool)
    down_break = np.zeros(len(close), dtype=bool)
    up_break[1:] = up_close[1:] & ~up_close[:-1]          # bar 0 is never an event
    down_break[1:] = down_close[1:] & ~down_close[:-1]
    up_start = up_break & _first_in_excursion(up_close, close < level, prior[0])
    down_start = down_break & _first_in_excursion(down_close, close > level, prior[1])
    return up_close, down_close, up_break, down_break, up_start, down_start


def _first_in_excursion(beyond: np.ndarray, back_through: np.ndarray,
                        running: bool = False) -> np.ndarray:
    """True where `beyond` holds for the first time since `back_through` last
    did (or since the series began). `running`: an excursion was already
    under way before the first bar, so until the first `back_through` nothing
    is a first.

    Each `back_through` bar (or bar 0) starts a group; the count is the
    running number of `beyond` bars inside the group.
    """
    n = len(beyond)
    if n == 0:
        return np.zeros(0, dtype=bool)
    seen = np.cumsum(beyond)                     # beyond bars up to and including i
    before = seen - beyond                       # ... strictly before i
    start = np.maximum.accumulate(np.where(back_through, np.arange(n), 0))
    count = seen - before[start]
    if running:
        count = count + (np.cumsum(back_through) == 0)
    return (count == 1) & beyond


def excursion_before(close: np.ndarray, buffer: np.ndarray, level: float,
                     pos: int) -> tuple[bool, bool]:
    """(up, down): is an excursion beyond `level` under way on bar pos - 1?

    Scans back from pos - 1 to the nearest bar that settles it: a close
    beyond by the buffer (under way) or a close back through the level
    (ended). Usually a few bars. Both sides are scanned separately.
    """
    up = down = None
    for i in range(pos - 1, -1, -1):
        c, b = close[i], buffer[i]
        if up is None:
            if c < level:
                up = False
            elif c > level + b:
                up = True
        if down is None:
            if c > level:
                down = False
            elif c < level - b:
                down = True
        if up is not None and down is not None:
            break
    return bool(up), bool(down)


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
    bull = ((feats[CLV] >= clv_t)
            & wick_dominates(feats[LOWER_WICK], body, ratio) & big_enough)
    bear = ((feats[CLV] <= -clv_t)
            & wick_dominates(feats[UPPER_WICK], body, ratio) & big_enough)

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
        "upper": (wick_dominates(feats[UPPER_WICK], body, ratio)
                  & small_body).fillna(False),
        "lower": (wick_dominates(feats[LOWER_WICK], body, ratio)
                  & small_body).fillna(False),
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
                     kind="failed_breakout", only: str | None = None,
                     prior: tuple[bool, bool] = (False, False)) -> pd.DataFrame:
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
    return _events_frame(_failed_breakout_events(
        bars["close"].to_numpy(dtype=float),
        breakout_buffer(atr, params).to_numpy(dtype=float),
        bars["ts"].to_numpy(), level, int(params.get(window_key)),
        kind=kind, only=only, prior=prior))


def _failed_breakout_events(close: np.ndarray, buf: np.ndarray, ts: np.ndarray,
                            level: float, k: int, *, kind: str = "failed_breakout",
                            only: str | None = None,
                            prior: tuple[bool, bool] = (False, False)
                            ) -> list[TriggerEvent]:
    """S8/S10 on plain arrays; positions are relative to the arrays given."""
    up_brk, dn_brk = _breakout_states(close, buf, level, prior)[4:]
    up_ok = only in (None, "up")
    down_ok = only in (None, "down")
    n = len(close)
    events: list[TriggerEvent] = []
    # one failed breakout per EXCURSION, anchored to its first close beyond
    for i in map(int, np.flatnonzero(up_brk | dn_brk)):
        if up_ok and up_brk[i]:
            for j in range(i + 1, min(i + 1 + k, n)):
                if close[j] < level:
                    events.append(TriggerEvent(
                        idx=j, ts=ts[j], kind=kind, direction=SHORT,
                        level=level, price=float(close[j]),
                        meta={"breakout_idx": i, "bars_to_fail": j - i}))
                    break
        elif down_ok and dn_brk[i]:
            for j in range(i + 1, min(i + 1 + k, n)):
                if close[j] > level:
                    events.append(TriggerEvent(
                        idx=j, ts=ts[j], kind=kind, direction=LONG,
                        level=level, price=float(close[j]),
                        meta={"breakout_idx": i, "bars_to_fail": j - i}))
                    break
    return events


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
                     params: Params, rejection_dir: pd.Series,
                     prior: tuple[bool, bool] = (False, False)) -> pd.DataFrame:
    """Confirmed breakout, pullback into the test zone, rejection there (S9).

    Invalidated if price closes back through the level in the failure
    direction before the retest -- that is an S8 failed breakout instead, and
    the two must not both fire on the same sequence.
    """
    return _events_frame(_breakout_retest_events(
        bars["close"].to_numpy(dtype=float), bars["high"].to_numpy(dtype=float),
        bars["low"].to_numpy(dtype=float),
        breakout_buffer(atr, params).to_numpy(dtype=float),
        test_zone(atr, params).to_numpy(dtype=float),
        rejection_dir.to_numpy(dtype=object), bars["ts"].to_numpy(), level,
        int(params.get("breakout_retest.max_bars_to_retest")), prior=prior))


def _breakout_retest_events(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                            buf: np.ndarray, tol: np.ndarray, rej: np.ndarray,
                            ts: np.ndarray, level: float, k_max: int, *,
                            prior: tuple[bool, bool] = (False, False)
                            ) -> list[TriggerEvent]:
    """S9 on plain arrays; positions are relative to the arrays given. The
    test zone is `in_test_zone()`'s rule: the bar straddles the level, or its
    nearer extreme is within the tolerance (a NaN tolerance never is)."""
    up_brk, dn_brk = _breakout_states(close, buf, level, prior)[4:]
    dist = np.minimum(np.abs(high - level), np.abs(low - level))
    zone = ((low <= level) & (high >= level)) | (dist <= tol)
    n = len(close)
    events: list[TriggerEvent] = []
    for i in map(int, np.flatnonzero(up_brk | dn_brk)):   # one per excursion, as S8
        for up in (True, False):
            if not (up_brk[i] if up else dn_brk[i]):
                continue
            want = LONG if up else SHORT
            for j in range(i + 1, min(i + 1 + k_max, n)):
                failed = close[j] < level if up else close[j] > level
                if failed:
                    break            # S8 territory, not a retest
                if zone[j] and rej[j] == want:
                    events.append(TriggerEvent(
                        idx=j, ts=ts[j], kind="breakout_retest",
                        direction=want, level=level, price=float(close[j]),
                        meta={"breakout_idx": i, "bars_to_retest": j - i}))
                    break
    return events


# --------------------------------------------------------------------------
# S11 momentum continuation
# --------------------------------------------------------------------------

def momentum_continuation(bars: pd.DataFrame, feats: pd.DataFrame,
                          minor_level: float, params: Params,
                          bias: pd.Series, volume_expanded: pd.Series,
                          *, side: str) -> pd.DataFrame:
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

    **The cross must match the level's side** (spec S11), the same rule S10
    applies to range edges. `side` is which kind of swing `minor_level` is:
    a swing "high" counts only an upward close through it (long), a swing
    "low" only a downward one (short). It is required, not inferred, because
    a bare price does not say which it is. Direction-blind, a close exactly
    at a swing high's extreme followed by a lower close read as a downward
    cross, which produced a short off a swing high (4 of 3,249 real events).
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    return _events_frame(_momentum_events(
        bars["close"].to_numpy(dtype=float), feats[BODY_RATIO].to_numpy(dtype=float),
        np.asarray(pd.Series(volume_expanded).fillna(False), dtype=bool),
        np.asarray(bias, dtype=object), bars["ts"].to_numpy(), minor_level, params,
        side=side))


def _momentum_events(close: np.ndarray, body_ratio: np.ndarray, vol: np.ndarray,
                     bias: np.ndarray, ts: np.ndarray, minor_level: float,
                     params: Params, *, side: str) -> list[TriggerEvent]:
    """S11 on plain arrays; positions are relative to the arrays given. A NaN
    body ratio or bias never passes, as with the fillna(False) it replaces."""
    min_ratio = float(params.get("momentum.min_body_ratio"))
    need_trend = bool(params.get("momentum.requires_trend_alignment"))
    need_vol = bool(params.get("momentum.requires_volume_expansion"))
    n = len(close)
    strong = body_ratio >= min_ratio
    vol_ok = vol if need_vol else np.ones(n, dtype=bool)
    up = close > minor_level
    down = close < minor_level
    up_cross = np.zeros(n, dtype=bool)
    down_cross = np.zeros(n, dtype=bool)
    up_cross[1:] = up[1:] & ~up[:-1]            # bar 0 has no prior bar to cross from
    down_cross[1:] = down[1:] & ~down[:-1]
    bull_ok = (bias == "bullish") if need_trend else np.ones(n, dtype=bool)
    bear_ok = (bias == "bearish") if need_trend else np.ones(n, dtype=bool)
    if side == "high":
        longs = strong & vol_ok & up_cross & bull_ok
        shorts = np.zeros(n, dtype=bool)
    else:
        longs = np.zeros(n, dtype=bool)
        shorts = strong & vol_ok & down_cross & bear_ok
    return [TriggerEvent(idx=i, ts=ts[i], kind="momentum",
                         direction=LONG if longs[i] else SHORT,
                         level=minor_level, price=float(close[i]),
                         meta={"body_ratio": float(body_ratio[i])})
            for i in map(int, np.flatnonzero(longs | shorts))]
