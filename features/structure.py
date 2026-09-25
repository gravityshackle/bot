"""S1 -- fractal swing highs and lows, and major/minor classification.

Two properties matter more than the arithmetic here.

**No repainting.** A pivot at bar i is not a swing until N further bars have
CLOSED. Every pivot therefore carries `confirmed_idx = i + N`, and consumers
must filter on that, not on the pivot's own index. A backtest that treats a
pivot as known at bar i is using information that did not exist for another N
bars, and no amount of downstream care recovers from that.

**Backward-looking majority.** Depth is measured from the PRIOR opposite swing
(spec S1, resolved). A swing high's depth is that high minus the preceding
confirmed swing low. This is deliberate: exit spec S15 sets the target to the
"next major level" and S16's R:R gate must evaluate it at signal time. Measured
forward, a pivot's majority would depend on a swing that has not formed yet, so
the target would be undefined exactly when the gate needs it.

Note that a pivot and its classification confirm together: the prior opposite
swing already exists by definition, so `is_major` is known at `confirmed_idx`
with no extra delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.schema import Params

PIVOT_COLUMNS = ["idx", "ts", "price", "kind", "confirmed_idx", "confirmed_ts"]


def find_pivots(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Fractal pivots: strictly higher (lower) than the N bars on either side.

    Strict inequality is per spec, which means a flat plateau of equal highs
    yields no pivot at all -- correct, since there is no single turning point.
    """
    if n < 1:
        raise ValueError("pivot N must be >= 1")
    if len(df) < 2 * n + 1:
        return pd.DataFrame(columns=PIVOT_COLUMNS)

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    ts = df["ts"].to_numpy()
    rows = []

    for i in range(n, len(df) - n):
        left_h, right_h = high[i - n:i], high[i + 1:i + 1 + n]
        if high[i] > left_h.max() and high[i] > right_h.max():
            rows.append((i, ts[i], high[i], "high", i + n, ts[i + n]))
        left_l, right_l = low[i - n:i], low[i + 1:i + 1 + n]
        if low[i] < left_l.min() and low[i] < right_l.min():
            rows.append((i, ts[i], low[i], "low", i + n, ts[i + n]))

    out = pd.DataFrame(rows, columns=PIVOT_COLUMNS)
    if out.empty:
        return out
    # order by when each pivot became KNOWN, then by bar -- this is the order a
    # live system would see them in
    return out.sort_values(["confirmed_idx", "idx"]).reset_index(drop=True)


def align_htf(ltf: pd.DataFrame, htf: pd.DataFrame, values: pd.Series,
              htf_freq: str, name: str = "htf_value") -> pd.Series:
    """Align ANY higher-timeframe series onto lower-timeframe bars, causally.

    Resampled bars are stamped with their OPEN time, so an HTF bar opening at
    t0 is not complete until t0 + freq. An LTF bar at time t may therefore only
    use HTF bars whose CLOSE is <= t. Aligning on open time instead would leak
    the currently-forming HTF bar into every LTF bar inside it.

    This is the single implementation of that rule. ATR was the first caller,
    but nothing here is ATR-specific: the trend bias (S14) and any other
    higher-timeframe value must cross timeframes the same way, and a second
    copy of this merge is exactly how one of them would quietly start aligning
    on open time instead.

    `values` may hold labels as well as numbers -- an object series of
    "bullish"/"bearish"/"neutral" aligns identically.
    """
    step = pd.Timedelta(htf_freq)
    right = pd.DataFrame({
        "htf_close_ts": htf["ts"] + step,
        "value": values.to_numpy(),
    }).dropna().sort_values("htf_close_ts")

    left = pd.DataFrame({"ts": ltf["ts"]}).sort_values("ts")
    merged = pd.merge_asof(left, right, left_on="ts", right_on="htf_close_ts",
                           direction="backward")
    return pd.Series(merged["value"].to_numpy(), index=ltf.index, name=name)


def htf_atr_at(ltf: pd.DataFrame, htf: pd.DataFrame, htf_atr: pd.Series,
               htf_freq: str) -> pd.Series:
    """S1's HTF ATR alignment -- `align_htf()` under its original name."""
    return align_htf(ltf, htf, htf_atr, htf_freq, name="htf_atr")


def classify_swings(pivots: pd.DataFrame, atr_at_pivot: pd.Series,
                    atr_multiple: float) -> pd.DataFrame:
    """Add depth and is_major, measuring depth from the prior opposite swing.

    `atr_at_pivot` is the HTF ATR in force when each pivot confirmed, indexed
    to match `pivots`.
    """
    out = pivots.copy()
    if out.empty:
        out["depth"] = []
        out["is_major"] = []
        out["prior_opposite_idx"] = []
        return out

    depths = np.full(len(out), np.nan)
    prior_idx = np.full(len(out), -1, dtype="int64")
    last_high = last_low = None          # (row position, price)

    for pos, row in enumerate(out.itertuples()):
        if row.kind == "high":
            if last_low is not None:
                depths[pos] = row.price - last_low[1]
                prior_idx[pos] = last_low[0]
            last_high = (row.idx, row.price)
        else:
            if last_high is not None:
                depths[pos] = last_high[1] - row.price
                prior_idx[pos] = last_high[0]
            last_low = (row.idx, row.price)

    out["depth"] = depths
    out["prior_opposite_idx"] = prior_idx
    atr = np.asarray(atr_at_pivot, dtype="float64")
    with np.errstate(invalid="ignore"):
        major = (np.abs(depths) >= atr_multiple * atr)

    # Nullable boolean, not plain bool: "unclassifiable" is a third state and
    # must not collapse into False. The first pivot of a series has no prior
    # opposite swing to measure against, and a bar with no HTF ATR yet has no
    # threshold -- calling either of those "minor" would quietly demote real
    # structure and change which levels S15 can target.
    out["is_major"] = pd.array(major, dtype="boolean")
    out.loc[out["prior_opposite_idx"] < 0, "is_major"] = pd.NA
    out.loc[np.isnan(atr), "is_major"] = pd.NA
    out.loc[np.isnan(depths), "is_major"] = pd.NA
    return out


def swings(ltf: pd.DataFrame, params: Params, *, n: int | None = None,
           htf: pd.DataFrame | None = None,
           htf_atr: pd.Series | None = None) -> pd.DataFrame:
    """Full S1 pipeline for one timeframe."""
    # Validate arguments before touching the data, so the failure does not
    # depend on whether this particular frame happened to contain a pivot.
    if htf is None or htf_atr is None:
        raise ValueError("swings() needs an HTF ATR series (spec S1 uses "
                         "ATR(14, HTF), not the entry timeframe)")

    n = int(n if n is not None else params.get("swings.pivot_n_ltf"))
    pivots = find_pivots(ltf, n)
    if pivots.empty:
        return classify_swings(pivots, pd.Series(dtype="float64"), 1.0)

    aligned = htf_atr_at(ltf, htf, htf_atr, str(params.get("timeframes.htf")))

    # ATR as of each pivot's CONFIRMATION bar, not the pivot bar -- that is when
    # the classification actually becomes available
    atr_at_pivot = aligned.iloc[pivots["confirmed_idx"].to_numpy()].reset_index(drop=True)
    return classify_swings(pivots, atr_at_pivot,
                           float(params.get("swings.major_atr_multiple")))


def last_confirmed_swings(pivots: pd.DataFrame, as_of_idx: int,
                          kind: str | None = None,
                          major_only: bool = False) -> pd.DataFrame:
    """Swings visible at bar `as_of_idx` -- the only safe accessor for backtest.

    Filters on confirmed_idx, so a pivot is invisible until it has actually
    survived its N bars.
    """
    out = pivots[pivots["confirmed_idx"] <= as_of_idx]
    if kind:
        out = out[out["kind"] == kind]
    if major_only:
        # Unclassified (pd.NA) is not major. Made explicit rather than left to
        # comparison semantics, so a future dtype change cannot silently start
        # admitting unclassified swings as targets.
        out = out[out["is_major"].fillna(False).astype(bool)]
    return out


DEAD_IDX = "dead_idx"


def mark_swing_deaths(pivots: pd.DataFrame, close: pd.Series, *,
                      major_buffer: pd.Series,
                      minor_buffer: pd.Series) -> pd.DataFrame:
    """Add `dead_idx`: the first bar after each pivot that breaks it.

    Spec S1 liveness. A swing high is dead once a bar closes above
    `high + buffer`, a swing low once one closes below `low - buffer`, with
    the buffer read on that closing bar. The threshold is whatever breaks the
    level for the trigger defined against it, so it differs by swing type:

      major  S4's breakout buffer (`triggers.breakout_buffer()`), because
             S8/S9 are built on a buffered breakout. With a bare close, a
             close inside the buffer killed the swing without being a
             breakout, and the real breakout that followed was of a dead
             level (19% of deaths, 14% of S8/S9 on swing levels).
      minor  zero, a bare close, because S11 momentum is a bare close. With
             the major buffer, a sub-buffer cross did not kill the level, so
             momentum re-fired on a later re-cross of the same swing.

    Unclassified swings (`is_major` NA) are levels under neither rule, so no
    death is computed for them (NA). They are excluded downstream anyway.
    Only closes are read, so a wick never kills a level. Otherwise NA means
    it had not died by the end of the data.

    `close` and both buffers are on the pivots' own frame, positionally.
    Both buffers are required keywords, so neither threshold can default.
    """
    for name, buf in (("major_buffer", major_buffer),
                      ("minor_buffer", minor_buffer)):
        if len(buf) != len(close):
            raise ValueError(f"{name} has {len(buf)} bars but close has "
                             f"{len(close)}; all must be the pivots' own frame")
    out = pivots.copy()
    c = close.to_numpy(dtype="float64")
    b_major = np.asarray(major_buffer, dtype="float64")
    b_minor = np.asarray(minor_buffer, dtype="float64")
    dead = []
    for idx, price, kind, major in zip(out["idx"], out["price"], out["kind"],
                                       out["is_major"]):
        if pd.isna(major):
            dead.append(pd.NA)
            continue
        i0 = int(idx) + 1
        after = c[i0:]
        buf = (b_major if bool(major) else b_minor)[i0:]
        hit = np.flatnonzero(after > price + buf if kind == "high"
                             else after < price - buf)
        dead.append(int(idx) + 1 + int(hit[0]) if len(hit) else pd.NA)
    out[DEAD_IDX] = pd.array(dead, dtype="Int64")
    return out


def live_major_swings(pivots: pd.DataFrame, as_of_idx: int,
                      kind: str | None = None) -> pd.DataFrame:
    """Major swings that are live levels at bar `as_of_idx` (spec S1).

    Confirmed by `as_of_idx`, classified major, and not yet closed beyond. A
    swing is dead ON the bar that closes beyond it: that bar has closed, so
    the break is known when that bar is decided. There is no recency window.
    A dead swing does not come back as the opposite side's level, which is a
    deliberate v1 simplification.
    """
    if DEAD_IDX not in pivots.columns:
        raise ValueError("pivots carry no dead_idx; run mark_swing_deaths() "
                         "first, or 'live' silently means every swing ever")
    out = last_confirmed_swings(pivots, as_of_idx, kind=kind, major_only=True)
    dead = out[DEAD_IDX]
    return out[(dead.isna() | (dead > as_of_idx)).to_numpy(dtype=bool)]
