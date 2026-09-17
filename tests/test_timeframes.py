"""Multi-timeframe orchestration: role resolution and cross-timeframe causality.

The load-bearing test here is `test_alignment_never_shows_a_forming_htf_bar`.
Everything else in this module is bookkeeping; that one is the rule that makes
a backtest reproducible live, and it fails silently when broken -- an HTF value
aligned on open time flatters every signal inside the bar and nothing raises.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml

from data.continuous_contract import trade_date
from features.schema import ATR, Params, load_params
from signal_engine import timeframes as tf

CT = "America/Chicago"


def load_sym(name="MES"):
    with open(f"config/symbols/{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def base_bars(minutes=60 * 24, start="2026-03-02 17:00"):
    """1m bars with a steadily rising price, so any misalignment is visible as
    a value that could not have been known yet."""
    ts = pd.date_range(pd.Timestamp(start), periods=minutes, freq="1min",
                       tz=CT).tz_convert("UTC")
    px = pd.Series(range(minutes), dtype="float64") + 100.0
    df = pd.DataFrame({
        "ts": ts, "raw_symbol": "MESM6",
        "open": px, "high": px + 0.5, "low": px - 0.5, "close": px + 0.25,
        "volume": 100,
    })
    df["trade_date"] = trade_date(df["ts"], CT, "17:00")
    return df


def with_roles(**overrides) -> Params:
    """A MES Params with timeframes.* overridden."""
    p = load_params("MES")
    values = copy.deepcopy(p.values)
    values["timeframes"].update(overrides)
    return Params(values=values, symbol="MES")


@pytest.fixture(scope="module")
def tfs():
    return tf.build("MES", base_bars(), load_sym())


# --------------------------------------------------------------------------
# roles
# --------------------------------------------------------------------------

def test_every_configured_role_is_built(tfs):
    for role in tf.ROLE_CONSUMERS:
        assert len(tfs.frame(role)) > 0
    assert tfs.freq("three_tail") == "10min", "S19 runs on its own spec frame"
    assert tfs.freq("entry") == "5min"


def test_unknown_role_names_the_configured_ones(tfs):
    with pytest.raises(tf.TimeframeError, match="three_tail"):
        tfs.frame("tick")


def test_roles_sharing_a_frequency_share_one_frame():
    """Nothing stops three_tail being configured to the entry frequency.

    When it is, the two roles must be the same object -- two separately
    resampled copies of one frequency can drift under later edits, and the
    whole point of resolving frames centrally is that they cannot.
    """
    p = with_roles(three_tail="5min")
    built = tf.build("MES", base_bars(), load_sym(), p)
    assert built.frame("three_tail") is built.frame("entry")
    assert sorted(built.frames) == ["1D", "1h", "5min"]


def test_s18_timeframes_are_built_even_when_no_role_names_them():
    """time_count.timeframes names frequencies directly, not roles."""
    p = with_roles(htf="4h")          # 1h now only appears in time_count
    built = tf.build("MES", base_bars(), load_sym(), p)
    assert "1h" in built.frames
    assert built.by_freq("1h") is not built.frame("htf")


def test_unresampleable_timeframe_is_refused():
    with pytest.raises(tf.TimeframeError, match="not resampleable"):
        tf.build("MES", base_bars(), load_sym(), with_roles(htf="3s"))


# --------------------------------------------------------------------------
# causality -- the one that matters
# --------------------------------------------------------------------------

def test_alignment_never_shows_a_forming_htf_bar(tfs):
    """An entry bar may only see HTF bars that have CLOSED.

    Carries the HTF bar's own open timestamp across as the value, so the
    assertion is exact rather than approximate: for every entry bar at time t
    the aligned bar must satisfy `open + freq <= t`. Aligning on open time
    instead -- the natural mistake, since resampled bars are stamped with their
    open -- puts the currently-forming bar into every entry bar inside it and
    this fails immediately.
    """
    htf = tfs.frame("htf")
    step = pd.Timedelta(tfs.freq("htf"))
    carried = pd.Series(htf["ts"].to_numpy(), index=htf.index)

    aligned = tfs.align("htf", carried, name="htf_open")
    entry_ts = tfs.entry["ts"]
    seen = pd.to_datetime(pd.Series(aligned.to_numpy(), index=aligned.index),
                          utc=True)

    known = seen.notna()
    assert (seen[known] + step <= entry_ts[known]).all(), \
        "an HTF bar became visible before it closed"

    # and the value is the LATEST such bar, not merely some earlier one
    for pos in range(0, len(entry_ts), 37):
        if not known.iloc[pos]:
            continue
        closed = htf["ts"][htf["ts"] + step <= entry_ts.iloc[pos]]
        assert seen.iloc[pos] == closed.max()


def test_entry_bars_before_the_first_htf_close_are_unknown(tfs):
    """Not-yet-knowable stays NaN rather than collapsing to a value."""
    aligned = tfs.align("htf", tfs.atr("htf"))
    step = pd.Timedelta(tfs.freq("htf"))
    first_close = tfs.frame("htf")["ts"].iloc[0] + step
    early = tfs.entry["ts"] < first_close
    assert early.any(), "fixture must span the first HTF bar"
    assert aligned[early].isna().all()


def test_alignment_preserves_the_entry_index(tfs):
    aligned = tfs.align("htf", tfs.atr("htf"))
    assert len(aligned) == len(tfs.entry)
    assert aligned.index.equals(tfs.entry.index)


def test_aligning_the_entry_role_to_itself_is_a_noop(tfs):
    same = tfs.align("entry", tfs.entry[ATR])
    assert same.equals(tfs.entry[ATR].rename(same.name))


def test_labels_align_the_same_way_as_numbers(tfs):
    """S14's bias is an object series; it must cross timeframes identically."""
    htf = tfs.frame("htf")
    labels = pd.Series(["bullish"] * len(htf), index=htf.index, dtype="object")
    aligned = tfs.align("htf", labels, name="bias")
    assert set(aligned.dropna().unique()) == {"bullish"}
    assert aligned.isna().any(), "early bars have no closed HTF bar yet"


# --------------------------------------------------------------------------
# per-frame features
# --------------------------------------------------------------------------

def test_intraday_frames_carry_volume_expansion(tfs):
    from features.schema import VOLUME_EXPANDED
    assert VOLUME_EXPANDED in tfs.entry.columns
    assert VOLUME_EXPANDED in tfs.frame("three_tail").columns


def test_daily_frames_omit_volume_expansion_rather_than_faking_it(tfs):
    """S12's baseline is session-matched or hour-of-day; neither exists on
    daily bars, and a meaningless column is worse than an absent one."""
    from features.schema import VOLUME_EXPANDED
    assert VOLUME_EXPANDED not in tfs.frame("daily").columns


def test_every_frame_carries_atr(tfs):
    for freq in tfs.frames:
        assert ATR in tfs.frames[freq].columns
