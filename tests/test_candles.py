"""Level-free candle triggers (S7/S19/S20) across their own timeframes.

S7 and S20 are read on the entry frame; S19 on the `three_tail` role, which
Soloway puts at 10min. The frames are hand-built and handed to a TimeframeSet
directly, so the 10min chart can hold a tail cluster the 5min chart does not --
the configuration in which running all three on one frame gives a wrong answer
rather than merely a different one.

The load-bearing test is `test_s19_is_read_from_the_three_tail_frame`: under
the old bundled `candle_triggers()` there was no way to express it at all.
"""
from __future__ import annotations

import pandas as pd

from features import confirmation, triggers
from features.schema import ATR, load_params
from signal_engine import candles
from signal_engine.timeframes import TimeframeSet

CT = "America/Chicago"
P = load_params("MES")          # tick 0.25, three_tail lookback 6, need 3

NEUTRAL = (100.0, 101.0, 99.5, 100.8)     # no qualifying wick either side
LOW_TAIL = (100.0, 100.5, 97.0, 100.2)    # lower wick 3.0 on a 0.2 body


def frame(rows, freq, start="2026-03-02 09:00", drop=(), next_session=None):
    """Bars with anatomy, CLV and a flat ATR of 4 attached -- the columns a
    TimeframeSet frame carries. `drop` removes bars by CT clock time (no
    prints); bars at or after `next_session` belong to the following trade
    date, which is how a session boundary looks to the frames."""
    ts = pd.date_range(pd.Timestamp(start), periods=len(rows), freq=freq,
                       tz=CT)
    day = pd.Timestamp(start).normalize()
    later = (ts.strftime("%H:%M") >= next_session) if next_session         else [False] * len(ts)
    df = pd.DataFrame({
        "ts": ts.tz_convert("UTC"), "raw_symbol": "MESM6",
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": 100,
        "trade_date": [(day + pd.Timedelta(days=int(n))).date() for n in later],
    })
    df = df[~ts.strftime("%H:%M").isin(drop)].reset_index(drop=True)
    df = pd.concat([df, confirmation.candle_anatomy(df)], axis=1)
    df[confirmation.CLV] = confirmation.clv(df)
    df[ATR] = 4.0
    return df


def tfset(entry, tt, tt_freq="10min"):
    frames = {"5min": entry, tt_freq: tt}
    return TimeframeSet(symbol="MES", params=P, symbol_cfg={},
                        roles={"entry": "5min", "three_tail": tt_freq},
                        frames=frames)


# 10min: tails at 09:30, 09:40, 09:50 -> S19 completes on the 09:50 bar,
# which CLOSES at 10:00. 5min: nothing but neutral bars.
TT_ROWS = [NEUTRAL] * 3 + [LOW_TAIL] * 3 + [NEUTRAL] * 2
CLUSTER_CLOSE = pd.Timestamp("2026-03-02 10:00", tz=CT).tz_convert("UTC")


def fired(out):
    return out.index[out["three_tail"].notna()]


# --------------------------------------------------------------------------
# which frame each pattern is read from
# --------------------------------------------------------------------------

def test_s19_is_read_from_the_three_tail_frame():
    """A cluster that exists only on the 10min chart must still fire.

    Run on the entry frame -- which is all the old bundled function could do --
    this is silently empty, and nothing raises.
    """
    entry = frame([NEUTRAL] * 30, "5min")
    assert triggers.three_tail(entry, entry, entry[ATR], P).empty, \
        "fixture: the entry frame alone must hold no cluster"

    out = candles.candle_triggers(tfset(entry, frame(TT_ROWS, "10min")))
    hits = fired(out)
    assert len(hits) == 1
    assert out.loc[hits[0], "three_tail"] == triggers.LONG


def test_s7_and_s20_are_read_from_the_entry_frame():
    entry = frame([NEUTRAL, (99.5, 101.2, 95.0, 100.8),         # rejection
                   (101.5, 101.6, 99.0, 99.2),                  # bear bar
                   (99.0, 102.3, 98.8, 102.2)] + [NEUTRAL] * 4,  # engulfs it
                  "5min")
    out = candles.candle_triggers(tfset(entry, frame(TT_ROWS, "10min")))
    assert out["rejection"].equals(triggers.rejection(entry, entry, entry[ATR], P))
    assert out["engulfing"].equals(triggers.engulfing(entry, entry, entry[ATR], P))
    assert out["rejection"].notna().any() and out["engulfing"].notna().any(), \
        "fixture must actually exercise both"


def test_output_is_indexed_on_the_entry_frame():
    entry = frame([NEUTRAL] * 30, "5min")
    out = candles.candle_triggers(tfset(entry, frame(TT_ROWS, "10min")))
    assert out.index.equals(entry.index)
    assert list(out.columns) == ["rejection", "engulfing", "three_tail",
                                 "three_tail_ts"]


# --------------------------------------------------------------------------
# crossing the timeframe boundary
# --------------------------------------------------------------------------

def test_s19_never_lands_before_its_bar_closes():
    """The cluster completes on the 09:50 bar; it is not known until 10:00.

    Same rule as `structure.align_htf()` -- an entry bar at t sees only 10min
    bars with open + 10min <= t -- so the event lands on the entry bar stamped
    10:00, and the 09:50 / 09:55 bars inside the forming 10min bar see nothing.
    """
    entry = frame([NEUTRAL] * 30, "5min")
    out = candles.candle_triggers(tfset(entry, frame(TT_ROWS, "10min")))
    hit = fired(out)[0]
    assert entry.loc[hit, "ts"] == CLUSTER_CLOSE
    assert out.loc[hit, "three_tail_ts"] == CLUSTER_CLOSE - pd.Timedelta("10min")


def test_s19_fires_once_rather_than_as_a_state():
    """Two 5min bars sit inside every 10min bar. Forward-filling the aligned
    value -- what `align()` does for a state like S14's bias -- would fire the
    event on both, and on every bar until the next 10min close. That is the S11
    failure again: an event read as a state."""
    entry = frame([NEUTRAL] * 30, "5min")
    out = candles.candle_triggers(tfset(entry, frame(TT_ROWS, "10min")))
    assert len(fired(out)) == 1


def test_s19_lands_on_the_first_entry_bar_after_a_quiet_spell():
    """No prints for a while after the close -- routine in a thin market --
    means no entry bars, not a stale pattern: price has not moved. The event
    lands on the first entry bar that does exist, still inside its session."""
    entry = frame([NEUTRAL] * 30, "5min", drop=("10:00", "10:05", "10:10"))
    out = candles.candle_triggers(tfset(entry, frame(TT_ROWS[:6], "10min")))
    hit = fired(out)
    assert len(hit) == 1
    assert entry.loc[hit[0], "ts"] == CLUSTER_CLOSE + pd.Timedelta("15min")


def test_s19_does_not_carry_into_the_next_session():
    """The cluster completes on the last bar of its session. The next entry
    bar is the following session's open, and delivering it there would
    present yesterday's pattern as a fresh one."""
    entry = frame([NEUTRAL] * 30, "5min", next_session="10:00")
    out = candles.candle_triggers(tfset(entry, frame(TT_ROWS[:6], "10min")))
    assert fired(out).empty


def test_s19_two_sided_bar_is_both_not_whichever_side_ran_last():
    """Regression, found on real 10min data: a bar can complete an upper and
    a lower cluster at once. Written per bar, the LONG side overwrote the
    SHORT one every time -- MES 9 of 83 S19 bars, MET 151 of 306."""
    spin = (100.0, 103.0, 97.0, 100.2)      # 2.8 up / 3.0 down on a 0.2 body
    tt = frame([NEUTRAL] * 3 + [spin] * 3 + [NEUTRAL] * 2, "10min")
    both = triggers.three_tail(tt, tt, tt[ATR], P)
    assert set(both["direction"]) == {triggers.LONG, triggers.SHORT}, \
        "fixture must complete both sides on one bar"

    out = candles.candle_triggers(tfset(frame([NEUTRAL] * 30, "5min"), tt))
    hit = fired(out)
    assert len(hit) == 1 and out.loc[hit[0], "three_tail"] == candles.BOTH


def test_s19_on_a_shared_frequency_matches_direct_detection():
    """three_tail configured to the entry frequency: nothing crosses a
    boundary, so the result is exactly what three_tail() says on that frame."""
    rows = [NEUTRAL] * 3 + [LOW_TAIL] * 3 + [NEUTRAL] * 4
    entry = frame(rows, "5min")
    tfs = TimeframeSet(symbol="MES", params=P, symbol_cfg={},
                       roles={"entry": "5min", "three_tail": "5min"},
                       frames={"5min": entry})
    out = candles.candle_triggers(tfs)
    direct = triggers.three_tail(entry, entry, entry[ATR], P)
    assert list(fired(out)) == list(direct["idx"])
    assert (out.loc[fired(out), "three_tail_ts"]
            == entry.loc[fired(out), "ts"]).all()
