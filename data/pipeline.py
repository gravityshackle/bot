"""Fetch -> trade_date -> roll map -> stitched continuous series.

One entry point so the backtest and the Phase 1 validation plots build their
series through identical code. If these ever diverge, Phase 5's backtest/paper
parity check becomes meaningless.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data.continuous_contract import (
    ContinuousSeries,
    daily_volume_by_contract,
    RolloverConfig,
    build_roll_map,
    stitch,
    trade_date,
)
from data.sources.databento_client import fetch_ohlcv

CONFIG_DIR = Path("config")


def load_symbol_config(symbol: str) -> dict:
    p = CONFIG_DIR / "symbols" / f"{symbol}.yaml"
    if not p.exists():
        p = CONFIG_DIR / "symbols" / "full_size" / f"{symbol}.yaml"
    with open(p) as fh:
        return yaml.safe_load(fh)


def load_data_config() -> dict:
    with open(CONFIG_DIR / "data.yaml") as fh:
        return yaml.safe_load(fh)


def build_continuous(symbol: str, data_cfg: dict | None = None
                     ) -> tuple[ContinuousSeries, dict, pd.DataFrame]:
    """Return (continuous series, symbol config, all raw per-contract bars)."""
    data_cfg = data_cfg or load_data_config()
    scfg = load_symbol_config(symbol)

    result = fetch_ohlcv(data_cfg, scfg)
    bars = result.bars
    if bars.empty:
        raise RuntimeError(f"{symbol}: no bars available")

    # Trade date honours each symbol's own boundary -- 17:00 CT for the six
    # session-based instruments, 16:00 CT for MET.
    bars = bars.copy()
    bars["trade_date"] = trade_date(
        bars["ts"], scfg["session"]["timezone"], scfg["day_boundary"])

    roll_cfg = RolloverConfig.from_symbol_config(scfg)
    roll_map, warnings = build_roll_map(bars, result.metas, roll_cfg)
    series = stitch(bars, roll_map, symbol=symbol, warnings=warnings)
    series.warnings.extend(result.problems)
    series.warnings.extend(coverage_warnings(series, bars, data_cfg))
    return series, scfg, bars


def coverage_warnings(series: ContinuousSeries, all_bars: pd.DataFrame,
                      data_cfg: dict) -> list[str]:
    """Assert the stitched series actually tracks the liquid contract.

    A continuous series built from the wrong contract is still a well-formed
    frame -- monotonic timestamps, valid OHLC, tick-aligned prices -- so every
    structural check passes while the prices belong to something nobody trades.
    Volume share is what catches it, so it is checked on every build rather
    than left to the validation script.
    """
    out: list[str] = []
    thresh = float(data_cfg.get("validation", {})
                   .get("min_active_volume_share_warn", 0.50))

    total_root = int(all_bars["volume"].sum())
    if total_root <= 0:
        return out
    captured = int(series.bars["volume"].sum()) / total_root
    if captured < thresh:
        out.append(
            f"{series.symbol}: stitched series holds only {captured:.1%} of this "
            f"root's total volume -- the roll map is almost certainly tracking "
            f"the wrong contract(s)")

    vol = daily_volume_by_contract(all_bars)
    day_total = vol.sum(axis=1)
    active = series.bars.groupby("trade_date", observed=True)["raw_symbol"].first()
    shares = {d: vol.at[d, a] / day_total.loc[d]
              for d, a in active.items()
              if d in vol.index and day_total.loc[d] > 0}
    if shares:
        s = pd.Series(shares)
        bad = s[s < thresh]
        # One low-share session per roll is expected: the crossover is detected
        # on that session and the roll takes effect on the next one.
        if len(bad) > max(len(series.roll_map), 0) + 1:
            out.append(
                f"{series.symbol}: {len(bad)} session(s) below {thresh:.0%} "
                f"volume share (worst {s.min():.1%} on {s.idxmin()}) against "
                f"{len(series.roll_map)} roll(s) -- more than roll transitions "
                "alone explain")
    return out


def seam_report(series: ContinuousSeries, scfg: dict,
                all_bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Decompose the visible step at each roll into its two causes.

    The step you SEE in an unadjusted series is not all roll artifact:

        visible_step = contract_spread + market_move

    contract_spread is the simultaneous price difference between the outgoing
    and incoming contracts at the roll moment -- that is what offset_difference
    records and what back-adjustment removes. market_move is the genuine price
    change between the last bar of the old session and the first bar of the
    new one (a full weekend for a Monday roll), and adjustment must NOT remove
    it.

    Comparing offset_difference against the visible step directly -- as an
    earlier version of this function did -- makes every Monday roll look like a
    mismatch when nothing is wrong.
    """
    rows = []
    b = series.bars
    tick = scfg["contract_spec"]["tick_size"]

    closes = None
    if all_bars is not None:
        closes = (all_bars.sort_values("ts")
                  .groupby(["trade_date", "raw_symbol"], observed=True)["close"]
                  .last().unstack())

    for row in series.roll_map.itertuples():
        before = b[b["trade_date"] < row.effective_date]
        after = b[b["trade_date"] >= row.effective_date]
        if before.empty or after.empty:
            continue
        last, first = before["close"].iloc[-1], after["close"].iloc[0]
        visible = first - last

        spread = float("nan")
        if closes is not None:
            decide = before["trade_date"].iloc[-1]
            try:
                spread = float(closes.at[decide, row.to_symbol]
                               - closes.at[decide, row.from_symbol])
            except KeyError:
                pass

        rows.append({
            "effective_date": row.effective_date,
            "from": row.from_symbol,
            "to": row.to_symbol,
            "reason": row.reason,
            "last_close_before": last,
            "first_close_after": first,
            "visible_step": visible,
            "contract_spread": spread,
            "market_move": visible - spread,
            "recorded_offset": row.offset_difference,
            "offset_ok": abs(spread - row.offset_difference) < 1e-6,
            "ticks": visible / tick,
        })
    return pd.DataFrame(rows)


def largest_non_roll_jumps(series: ContinuousSeries, n: int = 5) -> pd.DataFrame:
    """Biggest bar-to-bar close changes that are NOT at a roll boundary.

    Also reports the time gap preceding each jump. A large move across a
    multi-hour gap is a weekend reopen or maintenance halt -- expected. A large
    move across a one-minute gap is a real intraday event (or bad data), and is
    the only kind worth investigating.
    """
    b = series.bars.sort_values("ts").reset_index(drop=True)
    d = b["close"].diff().abs()
    gap_min = b["ts"].diff().dt.total_seconds() / 60.0
    roll_dates = set(series.roll_map["effective_date"]) if not series.roll_map.empty else set()
    # exclude the first bar of each roll date -- that gap is the intended seam
    is_roll_edge = b["trade_date"].isin(roll_dates) & (
        b["trade_date"] != b["trade_date"].shift())
    d = d.where(~is_roll_edge)
    idx = d.nlargest(n).index
    return pd.DataFrame({
        "ts": b.loc[idx, "ts"],
        "raw_symbol": b.loc[idx, "raw_symbol"],
        "jump": d.loc[idx],
        "gap_minutes": gap_min.loc[idx],
        "kind": [("session_gap" if g > 5 else "intraday") for g in gap_min.loc[idx]],
        "prev_close": b["close"].shift().loc[idx],
        "close": b.loc[idx, "close"],
    }).reset_index(drop=True)
