"""S15 -- ATR, stop distance, and volatility regime.

ATR is load-bearing far beyond this module: it sets the breakout buffer (S4),
the test-zone tolerance (S6), the rejection-candle body filter (S7), the gap
threshold (S3), the three-tail cluster tolerance (S19), the stop distance, and
through the stop it sets position size. An ATR that is wrong by 20% is a
position size wrong by 20% on every trade, with nothing reporting an error.

So two choices are made explicitly rather than by habit:

1. **Wilder's smoothing**, not a simple mean. "ATR(14)" without qualification
   means Wilder's RMA -- that is what the indicator is. A simple rolling mean
   gives materially different values (it reacts faster and decays faster), and
   silently swapping one for the other would shift every derived threshold.
2. **The regime baseline excludes the current bar.** ATR(14).rolling_mean(50)
   compared against ATR(14) must not include today in today's own baseline,
   or the comparison is partly self-referential.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.schema import ATR, ATR_MEAN, TRUE_RANGE, VOL_REGIME, Params


def true_range(df: pd.DataFrame) -> pd.Series:
    """max(H-L, |H-prevC|, |L-prevC|). First bar has no prior close -> H-L."""
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    tr.iloc[0] = hl.iloc[0] if len(df) else np.nan
    return tr.rename(TRUE_RANGE)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's running moving average -- an EWMA with alpha = 1/period.

    Seeded with the simple mean of the first `period` values, which is the
    conventional definition and keeps values comparable with charting
    platforms.
    """
    s = series.astype("float64")
    if len(s) < period:
        return pd.Series(np.nan, index=s.index, dtype="float64")

    vals = np.asarray(s, dtype="float64")
    # own the buffer: pandas 3 hands back read-only views from to_numpy()
    arr = np.full(len(s), np.nan, dtype="float64")
    prev = float(np.nanmean(vals[:period]))
    arr[period - 1] = prev
    for i in range(period, len(s)):
        prev = (prev * (period - 1) + vals[i]) / period
        arr[i] = prev
    return pd.Series(arr, index=s.index)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return wilder_rma(true_range(df), period).rename(ATR)


def volatility_regime(atr_series: pd.Series, mean_window: int,
                      high_ratio: float, low_ratio: float
                      ) -> tuple[pd.Series, pd.Series]:
    """Classify each bar high / normal / low against a trailing ATR mean.

    The mean is shifted by one bar so the current ATR is never part of the
    baseline it is measured against.
    """
    baseline = atr_series.shift(1).rolling(mean_window, min_periods=mean_window).mean()
    regime = pd.Series("normal", index=atr_series.index, dtype="object")
    regime[atr_series > high_ratio * baseline] = "high"
    regime[atr_series < low_ratio * baseline] = "low"
    regime[baseline.isna() | atr_series.isna()] = "unknown"
    return regime.rename(VOL_REGIME), baseline.rename(ATR_MEAN)


def apply(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Attach true_range, atr, atr_mean and vol_regime."""
    period = int(params.get("atr.period"))
    out = df.copy()
    out[TRUE_RANGE] = true_range(out)
    out[ATR] = wilder_rma(out[TRUE_RANGE], period)
    regime, baseline = volatility_regime(
        out[ATR],
        int(params.get("atr.regime_mean_window")),
        float(params.get("atr.high_vol_ratio")),
        float(params.get("atr.low_vol_ratio")),
    )
    out[ATR_MEAN] = baseline
    out[VOL_REGIME] = regime
    return out


def stop_distance(atr_value: float, params: Params) -> float:
    """Invalidation buffer, S15: 0.10-0.25 x ATR (config default 0.20)."""
    return float(params.get("stops.buffer_atr_multiple")) * atr_value


def reward_risk(entry: float, stop: float, target: float, is_long: bool) -> float:
    """S16. Returns NaN on a zero-width risk leg rather than dividing by zero."""
    risk = (entry - stop) if is_long else (stop - entry)
    reward = (target - entry) if is_long else (entry - target)
    if risk <= 0:
        return float("nan")
    return reward / risk
