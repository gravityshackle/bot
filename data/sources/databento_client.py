"""Databento historical OHLCV, normalized to the canonical schema.

Two things this module refuses to do:

1. Fetch without pricing first. Databento bills by volume and parent symbology
   ("MES.FUT") expands to every listed contract, including dozens of nearly
   dead far-dated ones. get_cost() is cheap and is always called first, with a
   hard abort above config cost_control.abort_above_usd.
2. Hand back bars it has not validated. Everything goes through
   sources.base.validate(), including the tick-grid check -- if prices ever
   arrive off the exchange tick grid, something upstream has adjusted them and
   the whole unadjusted-levels design is void.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from data.sources.base import (
    ContractMeta,
    SchemaError,
    normalize,
    parse_raw_symbol,
    validate,
)


class CostLimitExceeded(RuntimeError):
    """Estimated spend exceeded the configured ceiling; nothing was fetched."""


class ConfirmationRequired(RuntimeError):
    """A billable fetch was attempted without explicit approval.

    A cost estimate is not consent. Any request that would actually bill has to
    be approved by the operator first -- this exists because a moving cache key
    once caused a full re-pull to run silently just because the date changed.
    Callers pass confirm=True only after a human has seen the price.
    """


@dataclass
class FetchResult:
    bars: pd.DataFrame
    metas: dict[str, ContractMeta]
    cost_usd: float
    from_cache: bool
    problems: list[str]


def _client(cfg: dict):
    import databento as db

    load_dotenv()
    key = os.getenv(cfg["source"]["api_key_env"])
    if not key:
        raise RuntimeError(
            f"{cfg['source']['api_key_env']} not set. Put it in .env "
            "(gitignored) -- never in a committed file."
        )
    return db.Historical(key)


def resolve_window(cfg: dict, end: pd.Timestamp | None = None
                   ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive start / exclusive end, UTC."""
    w = cfg["window"]
    if end is None:
        if w["end"] == "last_complete_session":
            # yesterday UTC: today's session may still be open
            end = (pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=1))
        else:
            end = pd.Timestamp(w["end"], tz="UTC")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo \
        else pd.Timestamp(end, tz="UTC")
    start = end - pd.DateOffset(months=int(w["months"]))
    return start.normalize(), end.normalize()


def estimate_cost(cfg: dict, symbol_cfg: dict, start, end) -> float:
    """Price a request without sending it."""
    client = _client(cfg)
    src = cfg["source"]
    return float(client.metadata.get_cost(
        dataset=src["dataset"],
        symbols=[symbol_cfg["databento"]["parent_symbol"]],
        stype_in=src["stype_in"],
        schema=src["schema"],
        start=start.isoformat(),
        end=end.isoformat(),
    ))


def fetch_definitions(cfg: dict, symbol_cfg: dict, start, end
                      ) -> dict[str, ContractMeta]:
    """Real expiration dates per contract -- the calendar backstop needs these.

    Derived from the definition schema rather than guessed from the month code,
    since expiry conventions differ per product (MCL ~3 business days before
    the 25th of the prior month; MGC end-of-month; MET last Friday).
    """
    client = _client(cfg)
    src = cfg["source"]
    root = symbol_cfg["databento"]["parent_symbol"].split(".")[0]

    store = client.timeseries.get_range(
        dataset=src["dataset"],
        symbols=[symbol_cfg["databento"]["parent_symbol"]],
        stype_in=src["stype_in"],
        schema="definition",
        start=start.isoformat(),
        end=end.isoformat(),
    )
    df = store.to_df()
    if df.empty:
        return {}

    metas: dict[str, ContractMeta] = {}
    for raw, grp in df.groupby("raw_symbol"):
        try:
            code, year = parse_raw_symbol(str(raw), root)
        except SchemaError:
            continue          # spreads and other non-outright legs
        exp = pd.to_datetime(grp["expiration"].iloc[0], utc=True)
        metas[str(raw)] = ContractMeta(raw_symbol=str(raw), root=root,
                                       month_code=code, year=year, expiration=exp)
    return metas


def _cache_path(cfg: dict, symbol: str, start, end, kind: str) -> Path:
    d = Path(cfg["cache"]["dir"]) / symbol
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}"
    return d / f"{kind}_{tag}.parquet"


def fetch_ohlcv(cfg: dict, symbol_cfg: dict, *, end=None,
                dry_run: bool = False, confirm: bool = False) -> FetchResult:
    """Fetch (or load cached) 1m bars for every listed contract of one root.

    Reading from cache is always free and never needs confirmation. A fetch
    that would bill requires confirm=True when cost_control
    .require_explicit_confirmation is set.
    """
    symbol = symbol_cfg["symbol"]
    src = cfg["source"]
    start, end_ts = resolve_window(cfg, end)
    bars_path = _cache_path(cfg, symbol, start, end_ts, "ohlcv1m")
    meta_path = _cache_path(cfg, symbol, start, end_ts, "meta")

    if cfg["cache"]["reuse_existing"] and bars_path.exists() and meta_path.exists():
        bars = pd.read_parquet(bars_path)
        metas = _metas_from_frame(pd.read_parquet(meta_path))
        return FetchResult(bars, metas, 0.0, True,
                           validate(bars, tick_size=symbol_cfg["contract_spec"]["tick_size"],
                                    symbol=symbol, strict=False))

    cost = 0.0
    if cfg["cost_control"]["estimate_before_fetch"]:
        cost = estimate_cost(cfg, symbol_cfg, start, end_ts)
        ceiling = float(cfg["cost_control"]["abort_above_usd"])
        if cost > ceiling:
            raise CostLimitExceeded(
                f"{symbol}: estimated ${cost:.2f} exceeds ceiling ${ceiling:.2f} "
                f"for {start:%Y-%m-%d}..{end_ts:%Y-%m-%d}. Narrow the window or "
                "raise cost_control.abort_above_usd deliberately."
            )
    if dry_run:
        return FetchResult(pd.DataFrame(), {}, cost, False, [])

    if cfg["cost_control"].get("require_explicit_confirmation", True) and not confirm:
        raise ConfirmationRequired(
            f"{symbol}: this would BILL about ${cost:.2f} for "
            f"{start:%Y-%m-%d}..{end_ts:%Y-%m-%d} (no cache entry for that "
            "window). Re-run with confirm=True only after the operator has "
            "approved the spend."
        )

    metas = fetch_definitions(cfg, symbol_cfg, start, end_ts)

    store = _client(cfg).timeseries.get_range(
        dataset=src["dataset"],
        symbols=[symbol_cfg["databento"]["parent_symbol"]],
        stype_in=src["stype_in"],
        schema=src["schema"],
        start=start.isoformat(),
        end=end_ts.isoformat(),
    )
    df = store.to_df()
    if df.empty:
        return FetchResult(pd.DataFrame(), metas, cost, False, [f"{symbol}: no bars returned"])

    df = df.reset_index().rename(columns={"ts_event": "ts", "symbol": "raw_symbol"})
    if "raw_symbol" not in df.columns:
        raise SchemaError(f"{symbol}: Databento returned no symbol mapping")

    # Drop spreads/combos -- parent symbology includes them and they are not
    # outright contracts. Anything the month-code parser rejects is not a leg
    # we trade.
    root = symbol_cfg["databento"]["parent_symbol"].split(".")[0]
    keep = []
    for raw in df["raw_symbol"].unique():
        try:
            parse_raw_symbol(str(raw), root)
            keep.append(raw)
        except SchemaError:
            pass
    df = df[df["raw_symbol"].isin(keep)]

    bars = normalize(df)
    problems = validate(bars, tick_size=symbol_cfg["contract_spec"]["tick_size"],
                        symbol=symbol, strict=False)

    bars.to_parquet(bars_path, index=False)
    _metas_to_frame(metas).to_parquet(meta_path, index=False)
    return FetchResult(bars, metas, cost, False, problems)


def _metas_to_frame(metas: dict[str, ContractMeta]) -> pd.DataFrame:
    return pd.DataFrame([
        {"raw_symbol": m.raw_symbol, "root": m.root, "month_code": m.month_code,
         "year": m.year, "expiration": m.expiration}
        for m in metas.values()
    ])


def _metas_from_frame(df: pd.DataFrame) -> dict[str, ContractMeta]:
    return {
        r.raw_symbol: ContractMeta(
            raw_symbol=r.raw_symbol, root=r.root, month_code=r.month_code,
            year=int(r.year), expiration=pd.Timestamp(r.expiration))
        for r in df.itertuples()
    }
