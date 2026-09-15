"""S18 -- time count exhaustion.

A context flag, never a trigger. S18 is explicit: this must not fire a trade on
its own. It only adjusts confidence on other triggers -- down on momentum
continuation in the exhausted direction, up on reversal types against it.

Two details that are easy to get wrong:

**Flat bars hold the count.** `close[i] == close[i-1]` neither resets nor
extends it. So up, up, flat, up is a count of 3, not 1 (reset) and not 4
(extend). The flat bar is transparent.

**Per timeframe, never blended.** A daily exhaustion count and an intraday one
answer different questions, so they are computed independently and returned
separately. Summing or averaging them would produce a number that means
nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.schema import Params

UP, DOWN, FLAT = 1, -1, 0


def bar_direction(close: pd.Series) -> pd.Series:
    """+1 / -1 / 0 versus the prior close (S18)."""
    d = close.diff()
    out = pd.Series(FLAT, index=close.index, dtype="int64")
    out[d > 0] = UP
    out[d < 0] = DOWN
    out[d.isna()] = FLAT
    return out.rename("bar_direction")


def time_count(close: pd.Series) -> pd.DataFrame:
    """Consecutive same-direction periods ending at each bar.

    Returns count and the direction it is counting, so a consumer can ask
    "exhausted in which direction" rather than just "how long".
    """
    direction = bar_direction(close).to_numpy()
    counts = np.zeros(len(direction), dtype="int64")
    dirs = np.zeros(len(direction), dtype="int64")

    run_dir, run_len = FLAT, 0
    for i, d in enumerate(direction):
        if d == FLAT:
            pass                      # hold: neither reset nor extend
        elif d == run_dir:
            run_len += 1
        else:
            run_dir, run_len = d, 1
        counts[i], dirs[i] = run_len, run_dir

    return pd.DataFrame({"time_count": counts, "count_direction": dirs},
                        index=close.index)


def exhaustion(close: pd.Series, params: Params) -> pd.DataFrame:
    """Attach the exhaustion flag and the direction being exhausted."""
    threshold = int(params.get("time_count.exhaustion_threshold"))
    out = time_count(close)
    out["exhausted"] = (out["time_count"] >= threshold) & (out["count_direction"] != FLAT)
    return out


def exhaustion_by_timeframe(frames: dict[str, pd.DataFrame],
                            params: Params) -> dict[str, pd.DataFrame]:
    """Compute S18 independently on each timeframe.

    `frames` maps a timeframe label to its bars. Returned per label and never
    combined -- see the module docstring.
    """
    return {tf: exhaustion(df["close"], params) for tf, df in frames.items()}


def confidence_adjustment(exhausted: bool, count_direction: int,
                          trigger_kind: str, trade_direction: str) -> str:
    """How S18 should modify a trigger's confidence -- advisory only.

    Returns "reduce", "increase" or "none". The Signal Engine applies the
    actual weighting; this keeps the S18 reasoning in one place instead of
    scattered through scoring.
    """
    if not exhausted or count_direction == FLAT:
        return "none"
    exhausted_up = count_direction == UP
    trade_long = trade_direction == "long"

    if trigger_kind == "momentum":
        # continuation in the exhausted direction is the weaker case
        return "reduce" if (exhausted_up == trade_long) else "none"

    if trigger_kind in ("rejection", "three_tail", "failed_breakout",
                        "range_reclaim", "engulfing"):
        # a reversal fading the exhausted move is the stronger case
        return "increase" if (exhausted_up != trade_long) else "none"

    return "none"
