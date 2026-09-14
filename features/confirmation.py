"""S12 volume expansion and S13 close location value.

The futures-specific part is S12's baseline. Comparing an RTH bar against an
average that includes overnight bars badly understates the baseline, because
overnight volume is a fraction of day volume -- so almost every RTH bar would
look "expanded". The spec's answer is a session-matched average.

That answer needs extending here. After Phase 1, only MES/MNQ/MYM have an
RTH/ETH split at all; MCL, MGC, SIL and MET were measured and found to have no
isolable day session, so "session-matched" has nothing to match against for
four of the seven. Those use an hour-of-day baseline instead -- the same
substitute the spec already prescribes for MET -- which preserves the intent
(compare like with like) for instruments whose volume seasonality is diurnal
rather than sessional.

Baselines exclude the current bar. A bar must not contribute to the average it
is being tested against.
"""
from __future__ import annotations

import pandas as pd

from features.schema import (
    BODY,
    BODY_RATIO,
    CLV,
    LOWER_WICK,
    SESSION,
    UPPER_WICK,
    VOLUME_BASELINE,
    VOLUME_EXPANDED,
    VOLUME_RATIO,
    Params,
)


def clv(df: pd.DataFrame) -> pd.Series:
    """S13: ((close - low) - (high - close)) / (high - low), bounded [-1, 1].

    Zero-range bars are treated as 0 rather than dropped -- a bar that opened,
    closed, high'd and low'd at one price has no directional information, and
    0 is exactly "no information" on this scale.
    """
    rng = df["high"] - df["low"]
    out = ((df["close"] - df["low"]) - (df["high"] - df["close"]))
    return (out / rng).where(rng > 0, 0.0).rename(CLV)


def candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Body, wicks and body ratio -- shared by S7, S11 and S19."""
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    rng = df["high"] - df["low"]
    return pd.DataFrame({
        BODY: body,
        UPPER_WICK: upper,
        LOWER_WICK: lower,
        BODY_RATIO: (body / rng).where(rng > 0, 0.0),
    }, index=df.index)


def classify_session(df: pd.DataFrame, symbol_cfg: dict) -> pd.Series:
    """Label each bar rth / eth / continuous per the symbol's own config."""
    s = symbol_cfg["session"]
    if s["mode"] == "continuous" or not s.get("rth_open"):
        return pd.Series("continuous", index=df.index, dtype="object").rename(SESSION)

    local = df["ts"].dt.tz_convert(s["timezone"])
    minutes = local.dt.hour * 60 + local.dt.minute
    o_h, o_m = (int(x) for x in s["rth_open"].split(":"))
    c_h, c_m = (int(x) for x in s["rth_close"].split(":"))
    start, end = o_h * 60 + o_m, c_h * 60 + c_m
    in_rth = (minutes >= start) & (minutes < end)
    return pd.Series(["rth" if v else "eth" for v in in_rth],
                     index=df.index, dtype="object").rename(SESSION)


def volume_baseline(df: pd.DataFrame, session: pd.Series, mode: str,
                    lookback: int) -> pd.Series:
    """Trailing average volume of comparable bars, excluding the current bar.

    session_matched : previous `lookback` bars of the SAME session label
    hour_of_day     : previous `lookback` bars in the same local hour
    """
    if mode == "session_matched":
        group = session
    elif mode == "hour_of_day":
        group = df["_local_hour"]
    else:
        raise ValueError(f"unknown volume baseline mode {mode!r}")

    return (df["volume"]
            .groupby(group, sort=False)
            .transform(lambda s: s.shift(1).rolling(lookback, min_periods=1).mean())
            .rename(VOLUME_BASELINE))


def apply(df: pd.DataFrame, params: Params, symbol_cfg: dict) -> pd.DataFrame:
    """Attach CLV, candle anatomy, session label and volume expansion."""
    out = df.copy()
    out[CLV] = clv(out)
    out = pd.concat([out, candle_anatomy(out)], axis=1)
    out[SESSION] = classify_session(out, symbol_cfg)

    mode = symbol_cfg["volume_baseline"]
    if mode == "hour_of_day":
        out["_local_hour"] = out["ts"].dt.tz_convert(
            symbol_cfg["session"]["timezone"]).dt.hour

    lookback = int(params.get("volume_expansion.lookback_bars"))
    mult = float(params.get("volume_expansion.multiplier"))
    out[VOLUME_BASELINE] = volume_baseline(out, out[SESSION], mode, lookback)
    out[VOLUME_RATIO] = out["volume"] / out[VOLUME_BASELINE]
    out[VOLUME_EXPANDED] = out[VOLUME_RATIO] >= mult
    # an unknown baseline is not confirmation
    out.loc[out[VOLUME_BASELINE].isna() | (out[VOLUME_BASELINE] <= 0),
            VOLUME_EXPANDED] = False
    return out.drop(columns=[c for c in ["_local_hour"] if c in out.columns])
