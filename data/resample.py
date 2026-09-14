"""1m bars -> higher timeframes, aligned to each symbol's own session.

The subtle part is the bucket origin. Resampling on a naive UTC grid puts the
boundaries at 00:00/04:00/08:00 UTC, which fall in the middle of a CME session
and cut the day in arbitrary places. A 4h bar has to start when the session
starts, and a daily bar has to be the trade date -- 17:00 CT to 16:00 CT for
the six session instruments, 16:00 CT for MET -- not a UTC calendar day.

So buckets are indexed by elapsed time since that trade date's session open,
computed from local wall-clock time. That is also DST-safe: the session opens
at 17:00 CT whether that is 22:00 or 23:00 UTC, and a fixed UTC origin would
silently drift by an hour twice a year.

Nothing is forward-filled. A missing minute stays missing -- inventing a bar
where nothing traded would put fake price levels into a levels-based strategy.
"""
from __future__ import annotations

import pandas as pd

AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "raw_symbol": "first",
}

# Timeframes the config may request, mapped to pandas offsets.
FREQ_ALIASES = {
    "1min": "1min", "5min": "5min", "10min": "10min", "15min": "15min",
    "30min": "30min", "1h": "60min", "2h": "120min", "4h": "240min",
    "1D": "1D", "1d": "1D", "daily": "1D",
}


def session_open_utc(trade_dates: pd.Series, tz: str, boundary: str) -> pd.Series:
    """The UTC instant each trade date's session begins.

    With a 17:00 boundary, trade date D starts at 17:00 local on D-1. Built by
    localizing the wall-clock time per date, so DST is handled by the tz
    database rather than by an assumed fixed offset.
    """
    hh, mm = (int(x) for x in boundary.split(":"))
    d = pd.to_datetime(pd.Series(list(trade_dates)))
    local_naive = (d - pd.Timedelta(days=1)) + pd.Timedelta(hours=hh, minutes=mm)
    localized = local_naive.dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
    return localized.dt.tz_convert("UTC")


def resample(bars: pd.DataFrame, timeframe: str, symbol_cfg: dict) -> pd.DataFrame:
    """Aggregate canonical 1m bars to `timeframe`, session-aligned.

    Requires a `trade_date` column (added by data.pipeline.build_continuous).
    Returns the canonical columns plus trade_date and n_base_bars, the count of
    source bars in each bucket -- downstream can use it to drop partial bars at
    session edges rather than treating a two-minute stub as a full 4h bar.
    """
    if timeframe not in FREQ_ALIASES:
        raise ValueError(f"unknown timeframe {timeframe!r}; known: {sorted(FREQ_ALIASES)}")
    if bars.empty:
        return bars.copy()
    if "trade_date" not in bars.columns:
        raise ValueError("bars must carry trade_date; build them via data.pipeline")

    tz = symbol_cfg["session"]["timezone"]
    boundary = symbol_cfg["day_boundary"]
    freq = FREQ_ALIASES[timeframe]

    b = bars.sort_values("ts").reset_index(drop=True)

    if freq == "1D":
        # A daily bar IS the trade date. Never a UTC calendar day.
        out = (b.groupby("trade_date", observed=True)
                 .agg(**{k: (k, v) for k, v in AGG.items()},
                      n_base_bars=("close", "size"),
                      ts=("ts", "first"))
                 .reset_index())
        return out[["ts", "raw_symbol", "open", "high", "low", "close",
                    "volume", "trade_date", "n_base_bars"]]

    origin = session_open_utc(b["trade_date"], tz, boundary)
    elapsed = (b["ts"].reset_index(drop=True) - origin.reset_index(drop=True))
    step = pd.Timedelta(freq)
    bucket_no = (elapsed // step)
    b = b.assign(_bucket_ts=origin.reset_index(drop=True) + bucket_no * step)

    out = (b.groupby(["trade_date", "_bucket_ts"], observed=True)
             .agg(**{k: (k, v) for k, v in AGG.items()},
                  n_base_bars=("close", "size"))
             .reset_index()
             .rename(columns={"_bucket_ts": "ts"}))
    out = out.sort_values("ts").reset_index(drop=True)
    return out[["ts", "raw_symbol", "open", "high", "low", "close",
                "volume", "trade_date", "n_base_bars"]]


def resample_all(bars: pd.DataFrame, symbol_cfg: dict,
                 timeframes: list[str]) -> dict[str, pd.DataFrame]:
    return {tf: resample(bars, tf, symbol_cfg) for tf in timeframes}


def check_ohlc_integrity(base: pd.DataFrame, agg: pd.DataFrame) -> list[str]:
    """Aggregation must conserve the extremes and the volume of its source."""
    problems = []
    if base.empty or agg.empty:
        return problems
    if abs(int(base["volume"].sum()) - int(agg["volume"].sum())) > 0:
        problems.append(
            f"volume not conserved: base {int(base['volume'].sum()):,} != "
            f"aggregated {int(agg['volume'].sum()):,}")
    if agg["high"].max() != base["high"].max():
        problems.append(f"high extreme lost: {agg['high'].max()} != {base['high'].max()}")
    if agg["low"].min() != base["low"].min():
        problems.append(f"low extreme lost: {agg['low'].min()} != {base['low'].min()}")
    bad = agg[(agg["high"] < agg["low"])
              | (agg["high"] < agg[["open", "close"]].max(axis=1) - 1e-9)
              | (agg["low"] > agg[["open", "close"]].min(axis=1) + 1e-9)]
    if len(bad):
        problems.append(f"{len(bad)} aggregated bars violate OHLC ordering")
    return problems
