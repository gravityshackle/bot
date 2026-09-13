"""Canonical OHLCV schema that every data source normalizes to.

Databento (historical) and IBKR (live/paper) must both emit exactly this shape,
so that the Feature -> Signal layers cannot tell which source they are running
against. Phase 5 of the build order validates backtest/paper parity by replaying
the same dates through both; that only works if normalization happens here and
nowhere else.

Bars are UNADJUSTED individual-contract prices. Continuous stitching and any
back-adjustment happen downstream in data/continuous_contract.py, never here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

# Canonical column order. `ts` is the bar's OPEN time, UTC, tz-aware.
OHLCV_COLUMNS = ["ts", "raw_symbol", "open", "high", "low", "close", "volume"]

OHLCV_DTYPES = {
    "raw_symbol": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}


@dataclass(frozen=True)
class ContractMeta:
    """Per-contract facts needed by the roll logic."""
    raw_symbol: str          # e.g. "MESZ5"
    root: str                # e.g. "MES"
    month_code: str          # e.g. "Z"
    year: int                # e.g. 2025
    expiration: pd.Timestamp  # UTC, from Databento definition schema

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.year, MONTH_CODES.index(self.month_code))


MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
MONTH_CODE_TO_NUM = {c: i + 1 for i, c in enumerate(MONTH_CODES)}


class SchemaError(ValueError):
    """Raised when a source emits bars that violate the canonical contract."""


def parse_raw_symbol(raw: str, root: str) -> tuple[str, int]:
    """Split a CME globex symbol into (month_code, year).

    Handles both single-digit ("MESZ5") and two-digit ("MESZ25") year forms.
    Single-digit years are resolved to the decade nearest the current year,
    which is the CME convention and is unambiguous for any contract listed
    within +/- 5 years of today.
    """
    tail = raw[len(root):]
    if not tail or tail[0] not in MONTH_CODE_TO_NUM:
        raise SchemaError(f"cannot parse month code from {raw!r} (root={root!r})")
    month_code, digits = tail[0], tail[1:]
    if not digits.isdigit():
        raise SchemaError(f"cannot parse year from {raw!r}")
    if len(digits) >= 2:
        year = 2000 + int(digits[-2:])
    else:
        current = datetime.now().year
        decade, digit = current - (current % 10), int(digits)
        # pick the candidate decade whose year is closest to now
        year = min((decade - 10 + digit, decade + digit, decade + 10 + digit),
                   key=lambda y: abs(y - current))
    return month_code, year


def normalize(df: pd.DataFrame, *, raw_symbol: str | None = None) -> pd.DataFrame:
    """Coerce a source-native frame into the canonical schema."""
    out = df.copy()
    if raw_symbol is not None and "raw_symbol" not in out.columns:
        out["raw_symbol"] = raw_symbol

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    out = out[OHLCV_COLUMNS]
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for col, dt in OHLCV_DTYPES.items():
        out[col] = out[col].astype(dt)
    return out.sort_values(["raw_symbol", "ts"]).reset_index(drop=True)


def validate(
    df: pd.DataFrame,
    *,
    tick_size: float | None = None,
    symbol: str = "",
    strict: bool = True,
) -> list[str]:
    """Check canonical bars for the corruption modes that matter downstream.

    Returns a list of human-readable problems. A wrong bar here silently
    poisons every feature computed from it, so this runs on every fetch rather
    than only in tests.
    """
    problems: list[str] = []
    tag = f"[{symbol}] " if symbol else ""

    if df.empty:
        return [f"{tag}frame is empty"]

    if list(df.columns) != OHLCV_COLUMNS:
        problems.append(f"{tag}column order/set is not canonical: {list(df.columns)}")

    if df["ts"].dt.tz is None:
        problems.append(f"{tag}ts is tz-naive; must be UTC-aware")

    for sym, g in df.groupby("raw_symbol", sort=False):
        s = f"{tag}{sym}: "
        if not g["ts"].is_monotonic_increasing:
            problems.append(s + "timestamps not monotonic increasing")
        dupes = int(g["ts"].duplicated().sum())
        if dupes:
            problems.append(s + f"{dupes} duplicate timestamps")

        o, h, l, c = g["open"], g["high"], g["low"], g["close"]
        if (bad := int((h < l).sum())):
            problems.append(s + f"{bad} bars with high < low")
        if (bad := int((h < o.combine(c, np.maximum) - 1e-9).sum())):
            problems.append(s + f"{bad} bars with high below open/close")
        if (bad := int((l > o.combine(c, np.minimum) + 1e-9).sum())):
            problems.append(s + f"{bad} bars with low above open/close")
        if (bad := int((g[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())):
            problems.append(s + f"{bad} bars with non-positive prices")
        if (bad := int((g["volume"] < 0).sum())):
            problems.append(s + f"{bad} bars with negative volume")

        if tick_size:
            # Unadjusted prices must land exactly on the tick grid. If this
            # fails, either the data is wrong or something back-adjusted it
            # upstream -- both are serious and neither should pass silently.
            grid = (g[["open", "high", "low", "close"]] / tick_size)
            off = (grid - grid.round()).abs().max().max()
            if off > 1e-6:
                problems.append(
                    s + f"prices off the {tick_size} tick grid (max residual {off:.3g}) "
                        "-- data may already be adjusted"
                )

    if strict and problems:
        raise SchemaError(f"{len(problems)} validation problem(s):\n  - " +
                          "\n  - ".join(problems))
    return problems
