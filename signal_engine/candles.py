"""Level-free candle triggers -- S7 rejection, S20 engulfing, S19 three-tail --
each read on the frame its role names, delivered on the entry frame.

These used to be bundled as `features.triggers.candle_triggers()`, which took
one frame and ran all three on it. That was right while every role resolved to
the entry frequency and wrong the moment `three_tail` did not: S19 is a 10min
pattern by Soloway's own usage, and on 5min bars it finds different clusters
(MES: 310 events on 5min against 92 on 10min over the same 66 sessions). The
bundle had no way to be told otherwise.

It lives here rather than in `features/` because choosing a frame per detector
is orchestration: the detectors stay timeframe-agnostic, and this module is
where the timeframe roles meet them.

Whether a pattern occurred AT a level is Stage 1 gate 2's decision, not this
module's. `three_tail_ts` is kept for that: it is the 10min bar the cluster
completed on, which is where gate 2 has to look.
"""
from __future__ import annotations

import pandas as pd

from features import triggers
from features.schema import ATR
from signal_engine.timeframes import ENTRY, TimeframeSet

THREE_TAIL = "three_tail"
# One bar can complete an upper AND a lower cluster. That is contradictory
# evidence, not a direction, and it is labelled as such rather than resolved.
BOTH = "both"


def _per_bar(events: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """An events frame as a LONG / SHORT / BOTH / None series on `bars`.

    Writing directions straight into a per-bar column is last-wins, and
    `three_tail()` emits the lower (LONG) side after the upper, so a
    two-sided bar used to come out LONG. On real 10min data that was 4-11%
    of S19 bars on most instruments and half of MET's.
    """
    out = pd.Series([None] * len(bars), index=bars.index, dtype="object")
    if events.empty:
        return out
    dirs = events.groupby("idx")["direction"].agg(
        lambda d: d.iloc[0] if d.nunique() == 1 else BOTH)
    out.loc[dirs.index.to_numpy()] = dirs.to_numpy()
    return out


def candle_triggers(tfs: TimeframeSet) -> pd.DataFrame:
    """rejection, engulfing, three_tail, three_tail_ts -- on the entry frame.

    S7 and S20 run on the entry frame. S19 runs on the `three_tail` role's
    frame and crosses to entry through `TimeframeSet.align_events()`, so it
    fires once, on the first entry bar at which its bar has closed.
    """
    p = tfs.params
    entry = tfs.frame(ENTRY)
    out = pd.DataFrame(index=entry.index)
    out["rejection"] = triggers.rejection(entry, entry, entry[ATR], p)
    out["engulfing"] = triggers.engulfing(entry, entry, entry[ATR], p)

    tt = tfs.frame(THREE_TAIL)
    events = triggers.three_tail(tt, tt, tt[ATR], p)
    out[THREE_TAIL], out[f"{THREE_TAIL}_ts"] = tfs.align_events(
        THREE_TAIL, _per_bar(events, tt), name=THREE_TAIL)
    return out
