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


def htf_atr_at(ltf: pd.DataFrame, htf: pd.DataFrame, htf_atr: pd.Series,
               htf_freq: str) -> pd.Series:
    """Align a higher-timeframe ATR onto lower-timeframe bars, causally.

    Resampled bars are stamped with their OPEN time, so an HTF bar opening at
    t0 is not complete until t0 + freq. An LTF bar at time t may therefore only
    use HTF bars whose CLOSE is <= t. Aligning on open time instead would leak
    the currently-forming HTF bar into every LTF bar inside it.
    """
    step = pd.Timedelta(htf_freq)
    right = pd.DataFrame({
        "htf_close_ts": htf["ts"] + step,
        "htf_atr": htf_atr.to_numpy(),
    }).dropna().sort_values("htf_close_ts")

    left = pd.DataFrame({"ts": ltf["ts"]}).sort_values("ts")
    merged = pd.merge_asof(left, right, left_on="ts", right_on="htf_close_ts",
                           direction="backward")
    return pd.Series(merged["htf_atr"].to_numpy(), index=ltf.index, name="htf_atr")


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
