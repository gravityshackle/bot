"""S2 prior day/week high-low, S3 gap zones, S5 consolidation ranges.

This is the first module where the Phase 1 session decisions actually bite.
"The day's range" means two different things in this repo:

  MES / MNQ / MYM      RTH only, 08:30-15:15 CT (use_eth_range: false)
  MCL / MGC / SIL / MET  the full session (use_eth_range: true)

so `scope_mask()` is the single place that distinction is resolved. Everything
in S2 and S3 goes through it. Getting it wrong would not raise -- it would just
produce prior-day levels drawn from the wrong bars, and five trigger types
would inherit that silently.

Causality holds by construction: a trade date's levels are built only from
*completed prior* periods. For the RTH instruments, trade date D-1's cash
session closes at 15:15 CT, well before D opens at 17:00 CT. For the continuous
four, D-1 ends exactly at the day boundary where D begins. Either way, nothing
attached to a bar in D depends on anything inside D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.confirmation import classify_session
from features.schema import Params

PRIOR_DAY_HIGH = "prior_day_high"
PRIOR_DAY_LOW = "prior_day_low"
PRIOR_WEEK_HIGH = "prior_week_high"
PRIOR_WEEK_LOW = "prior_week_low"
IN_RANGE = "in_range"
RANGE_HIGH = "range_high"
RANGE_LOW = "range_low"


# --------------------------------------------------------------------------
# session scope -- the one place RTH-vs-continuous is decided
# --------------------------------------------------------------------------

def scope_mask(bars: pd.DataFrame, symbol_cfg: dict) -> pd.Series:
    """Which bars count toward this symbol's daily/weekly range.

    S2 says prior day H/L is RTH-only by default, unless the symbol config
    flags use_eth_range. After Phase 1 that flag is true for MCL/MGC/SIL/MET,
    which were measured to have no isolable day session, so for those four
    every bar counts.
    """
    if symbol_cfg["session"].get("use_eth_range", False):
        return pd.Series(True, index=bars.index)
    return (classify_session(bars, symbol_cfg) == "rth").rename(None)


def week_key(trade_dates: pd.Series) -> pd.Series:
    """ISO year-week. Sunday-evening bars already belong to Monday's trade
    date, so keying off trade_date puts them in the right week automatically."""
    d = pd.to_datetime(pd.Series(list(trade_dates)))
    iso = d.dt.isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2))


# --------------------------------------------------------------------------
# S2 prior day / prior week
# --------------------------------------------------------------------------

def period_extremes(bars: pd.DataFrame, symbol_cfg: dict,
                    key: pd.Series) -> pd.DataFrame:
    """High/low/open/close per period, using only in-scope bars."""
    mask = scope_mask(bars, symbol_cfg)
    scoped = bars[mask]
    if scoped.empty:
        return pd.DataFrame(columns=["high", "low", "first_open", "last_close",
                                     "n_bars"])
    k = key[mask]
    g = scoped.groupby(k, sort=True)
    return pd.DataFrame({
        "high": g["high"].max(),
        "low": g["low"].min(),
        "first_open": g["open"].first(),
        "last_close": g["close"].last(),
        "n_bars": g.size(),
    })


def liquid_sessions(extremes: pd.DataFrame, params: Params) -> pd.Series:
    """Which sessions are liquid enough to serve as someone's "prior day" (S2).

    A session qualifies when its bar count is at least `liquidity_floor_ratio`
    of the rolling median trade-date bar count. The median excludes the session
    being judged -- a thin session must not drag down the bar it is measured
    against.

    The earliest sessions have no prior history to form a median from. They
    qualify: "not yet knowable" is not the same as "thin", and demoting them
    would leave the start of every series with no prior-day levels at all.
    """
    ratio = float(params.get("prior_levels.liquidity_floor_ratio"))
    window = int(params.get("prior_levels.liquidity_median_window_sessions"))
    n = extremes["n_bars"].astype("float64")
    median = n.shift(1).rolling(window, min_periods=1).median()
    return ((n >= ratio * median) | median.isna()).rename("is_liquid")


def prior_liquid_period(extremes: pd.DataFrame, params: Params) -> dict:
    """Map each period to the most recent PRIOR liquid one (S2).

    "Prior day" is the previous *liquid* session, not simply the previous trade
    date. The two only differ where a thin session sits between two real ones,
    but there it matters: MET trades through weekends, so a naive shift makes
    Monday's prior-day levels come from Sunday's 81-144 bar session instead of
    Friday's ~734 bar one, and every trigger keyed to those levels inherits it.

    This is a general rule; MET is just where it bites hardest.
    """
    ok = liquid_sessions(extremes, params)
    out: dict = {}
    last = None
    for pos, period in enumerate(extremes.index):
        out[period] = last          # assigned before this period qualifies
        if bool(ok.iloc[pos]):
            last = period
    return out


def attach_prior_levels(bars: pd.DataFrame, symbol_cfg: dict,
                        params: Params) -> pd.DataFrame:
    """Attach prior-day and prior-week H/L to every bar.

    Prior-day levels come from the most recent *liquid* prior session; weekly
    levels shift by one week. Either way a bar never sees its own period.
    """
    out = bars.copy()
    td = out["trade_date"]

    daily = period_extremes(out, symbol_cfg, td)
    prior = prior_liquid_period(daily, params)
    prior_high = {d: (daily.at[p, "high"] if p is not None else np.nan)
                  for d, p in prior.items()}
    prior_low = {d: (daily.at[p, "low"] if p is not None else np.nan)
                 for d, p in prior.items()}
    out[PRIOR_DAY_HIGH] = td.map(prior_high).astype("float64")
    out[PRIOR_DAY_LOW] = td.map(prior_low).astype("float64")

    wk = week_key(td)
    weekly = period_extremes(out, symbol_cfg, wk)
    prior_weekly = weekly.shift(1)
    out[PRIOR_WEEK_HIGH] = wk.map(prior_weekly["high"]).astype("float64")
    out[PRIOR_WEEK_LOW] = wk.map(prior_weekly["low"]).astype("float64")
    return out


# --------------------------------------------------------------------------
# S3 gap zones
# --------------------------------------------------------------------------

def daily_atr_by_date(daily_bars: pd.DataFrame, period: int) -> pd.Series:
    """ATR(period) on daily bars, re-indexed by trade_date for S3.

    Daily bars carry both a ts and a trade_date; gaps are keyed by trade_date,
    so this converts once, here, rather than letting each caller guess.
    """
    from features.risk_state import atr as _atr
    return pd.Series(_atr(daily_bars, period).to_numpy(),
                     index=pd.Index(daily_bars["trade_date"], name="trade_date"))


def find_gaps(bars: pd.DataFrame, symbol_cfg: dict, daily_atr: pd.Series,
              params: Params) -> pd.DataFrame:
    """Gaps between one session's close and the next session's open.

    Scope follows S2: for the index micros this is the cash-session gap
    (prior RTH close -> next RTH open), which is the gap traders actually
    watch; for the continuous four it spans the daily maintenance halt.

    The threshold uses the PRIOR day's daily ATR -- using the current day's
    would test a gap against a volatility figure that partly depends on the
    gap itself.
    """
    td = bars["trade_date"]
    ext = period_extremes(bars, symbol_cfg, td)
    if len(ext) < 2:
        return pd.DataFrame(columns=["trade_date", "prior_close", "open",
                                     "gap", "zone_low", "zone_high",
                                     "direction", "threshold", "filled_date"])

    # daily_atr must be indexed by trade_date -- see daily_atr_by_date().
    # Reindexing on anything else would line ATR up against the wrong session.
    if not daily_atr.index.equals(ext.index):
        missing = ext.index.difference(daily_atr.index)
        if len(missing) == len(ext):
            raise ValueError(
                "daily_atr must be indexed by trade_date; got "
                f"{type(daily_atr.index).__name__}. Use daily_atr_by_date()."
            )
    mult = float(params.get("gaps.threshold_atr_multiple"))

    prior_close = ext["last_close"].shift(1)
    prior_atr = daily_atr.reindex(ext.index).shift(1)
    gap = ext["first_open"] - prior_close
    threshold = mult * prior_atr

    is_gap = gap.abs() >= threshold
    rows = []
    for date, flag in is_gap.items():
        if not flag or pd.isna(gap.loc[date]):
            continue
        o, pc = float(ext.at[date, "first_open"]), float(prior_close.loc[date])
        rows.append({
            "trade_date": date,
            "prior_close": pc,
            "open": o,
            "gap": float(gap.loc[date]),
            "zone_low": min(o, pc),
            "zone_high": max(o, pc),
            "direction": "up" if gap.loc[date] > 0 else "down",
            "threshold": float(threshold.loc[date]),
            "filled_date": pd.NaT,
        })
    out = pd.DataFrame(rows)
    return _mark_gap_fills(out, bars)


def _mark_gap_fills(gaps: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """A gap is filled once price traverses the WHOLE zone (S3).

    For an up gap the unfilled edge is the prior close below, so the fill is a
    trade down to zone_low; mirrored for a down gap. Touching the near edge is
    not a fill.
    """
    if gaps.empty:
        return gaps
    filled = []
    for row in gaps.itertuples():
        later = bars[bars["trade_date"] > row.trade_date]
        if later.empty:
            filled.append(pd.NaT)
            continue
        if row.direction == "up":
            hit = later[later["low"] <= row.zone_low]
        else:
            hit = later[later["high"] >= row.zone_high]
        filled.append(hit["trade_date"].iloc[0] if not hit.empty else pd.NaT)
    gaps = gaps.copy()
    gaps["filled_date"] = filled
    return gaps


# --------------------------------------------------------------------------
# S5 consolidation range
# --------------------------------------------------------------------------

def consolidation(bars: pd.DataFrame, atr: pd.Series, atr_mean: pd.Series,
                  params: Params) -> pd.DataFrame:
    """Range regime, true when EITHER S5 condition holds.

    Condition A is volatility compression -- identical to the low-vol regime in
    risk_state, so the already-shifted ATR mean is reused rather than
    recomputed, keeping one definition of "compressed".
    Condition B is a narrow observed span over the lookback window.
    """
    n = int(params.get("range_detection.span_lookback_bars"))
    compress = float(params.get("range_detection.atr_compression_ratio"))
    span_mult = float(params.get("range_detection.span_atr_multiple"))

    cond_a = atr < compress * atr_mean
    roll_high = bars["high"].rolling(n, min_periods=n).max()
    roll_low = bars["low"].rolling(n, min_periods=n).min()
    cond_b = (roll_high - roll_low) < span_mult * atr

    in_range = (cond_a.fillna(False) | cond_b.fillna(False))
    return pd.DataFrame({
        IN_RANGE: in_range,
        # boundaries are the compression window's own extremes, defined only
        # where the regime is actually flagged
        RANGE_HIGH: roll_high.where(in_range),
        RANGE_LOW: roll_low.where(in_range),
    }, index=bars.index)


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------

def apply(bars: pd.DataFrame, params: Params, symbol_cfg: dict,
          atr: pd.Series, atr_mean: pd.Series) -> pd.DataFrame:
    """Attach all S2 and S5 columns. S3 gaps are returned separately by
    find_gaps(), since a gap is a zone with a lifetime rather than a per-bar
    value."""
    out = attach_prior_levels(bars, symbol_cfg, params)
    return pd.concat([out, consolidation(bars, atr, atr_mean, params)], axis=1)
