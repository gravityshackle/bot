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
    is_outright,
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
        if not is_outright(str(raw), root):
            continue          # spreads and other non-outright legs
        exp = pd.to_datetime(grp["expiration"].iloc[0], utc=True)
        # the year comes from the data (this contract's expiration), never
        # the clock -- see parse_raw_symbol
        code, year = parse_raw_symbol(str(raw), root, near_year=exp.year)
        metas[str(raw)] = ContractMeta(raw_symbol=str(raw), root=root,
                                       month_code=code, year=year, expiration=exp)
    return metas


def _cache_path(cfg: dict, symbol: str, start, end, kind: str) -> Path:
    d = Path(cfg["cache"]["dir"]) / symbol
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}"
    return d / f"{kind}_{tag}.parquet"


def year_chunks(start: pd.Timestamp, end: pd.Timestamp
                ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) at each 1 January, half-open and contiguous.

    A window inside one calendar year is a single chunk equal to the window,
    so its cache file keeps the name it always had.
    """
    out, a = [], start
    while a < end:
        b = min(pd.Timestamp(year=a.year + 1, month=1, day=1, tz=a.tz), end)
        out.append((a, b))
        a = b
    return out


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file and an atomic rename, so a crash mid-write can
    never leave a truncated file that later reads as a valid cache hit."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _with_one_retry(fn, what: str):
    """One retry on a dropped stream; a second failure propagates.

    The first 5-year attempt died on "Response ended prematurely" while
    streaming definitions. A retry of a partly streamed request can bill that
    part again, which is why it is one retry and not a loop.
    """
    from databento.common.error import BentoError
    try:
        return fn()
    except BentoError as exc:
        print(f"  {what}: {exc} -- retrying once", flush=True)
        return fn()


def _fetch_chunk(cfg: dict, symbol_cfg: dict, start, end
                 ) -> tuple[pd.DataFrame, dict[str, ContractMeta]]:
    symbol, src = symbol_cfg["symbol"], cfg["source"]
    what = f"{symbol} {start:%Y-%m-%d}..{end:%Y-%m-%d}"
    metas = _with_one_retry(lambda: fetch_definitions(cfg, symbol_cfg, start, end),
                            f"{what} definitions")

    def bars_request():
        return _client(cfg).timeseries.get_range(
            dataset=src["dataset"],
            symbols=[symbol_cfg["databento"]["parent_symbol"]],
            stype_in=src["stype_in"],
            schema=src["schema"],
            start=start.isoformat(),
            end=end.isoformat(),
        ).to_df()
    df = _with_one_retry(bars_request, f"{what} bars")
    if df.empty:
        return pd.DataFrame(columns=["ts", "raw_symbol", "open", "high", "low",
                                     "close", "volume"]), metas

    df = df.reset_index().rename(columns={"ts_event": "ts", "symbol": "raw_symbol"})
    if "raw_symbol" not in df.columns:
        raise SchemaError(f"{symbol}: Databento returned no symbol mapping")
    # Drop spreads/combos -- parent symbology includes them and they are not
    # outright contracts.
    root = symbol_cfg["databento"]["parent_symbol"].split(".")[0]
    keep = [raw for raw in df["raw_symbol"].unique() if is_outright(str(raw), root)]
    return normalize(df[df["raw_symbol"].isin(keep)]), metas


def _merge_metas(into: dict[str, ContractMeta], more: dict[str, ContractMeta],
                 symbol: str) -> None:
    for raw, m in more.items():
        have = into.get(raw)
        if have is not None and have.expiration != m.expiration:
            raise SchemaError(
                f"{symbol}: contract {raw} has two expirations across chunks "
                f"({have.expiration} vs {m.expiration}); refusing to guess")
        into[raw] = m


def fetch_ohlcv(cfg: dict, symbol_cfg: dict, *, end=None, dry_run: bool = False,
                confirm: bool = False, assemble: bool = True) -> FetchResult:
    """Fetch (or load cached) 1m bars for every listed contract of one root.

    The window is fetched in calendar-year chunks (`year_chunks`), each saved
    to disk atomically the moment it arrives. A re-run reads saved chunks for
    free and fetches only the missing ones, so a failure or a killed process
    loses at most the chunk in flight. A dropped stream is retried once.

    Reading from cache is always free and never needs confirmation. Anything
    that would bill -- the chunks not yet on disk -- is priced first, refused
    above cost_control.abort_above_usd IN TOTAL, and requires confirm=True
    when cost_control.require_explicit_confirmation is set.

    `assemble=False` downloads and caches without concatenating the window in
    memory; it returns no bars. The bulk pull uses it.
    """
    symbol = symbol_cfg["symbol"]
    start, end_ts = resolve_window(cfg, end)
    tick = symbol_cfg["contract_spec"]["tick_size"]
    chunks = year_chunks(start, end_ts)
    paths = [(_cache_path(cfg, symbol, a, b, "ohlcv1m"), _cache_path(cfg, symbol, a, b, "meta"))
             for a, b in chunks]
    reuse = cfg["cache"]["reuse_existing"]
    pending = [i for i, (bp, mp) in enumerate(paths)
               if not (reuse and bp.exists() and mp.exists())]

    cost = 0.0
    if pending and cfg["cost_control"]["estimate_before_fetch"]:
        cost = sum(estimate_cost(cfg, symbol_cfg, *chunks[i]) for i in pending)
        ceiling = float(cfg["cost_control"]["abort_above_usd"])
        if cost > ceiling:
            raise CostLimitExceeded(
                f"{symbol}: estimated ${cost:.2f} exceeds ceiling ${ceiling:.2f} "
                f"for {start:%Y-%m-%d}..{end_ts:%Y-%m-%d} ({len(pending)} chunk(s) "
                "not on disk). Narrow the window or raise "
                "cost_control.abort_above_usd deliberately."
            )
    if dry_run and pending:
        return FetchResult(pd.DataFrame(), {}, cost, False, [])

    if pending and cfg["cost_control"].get("require_explicit_confirmation", True) \
            and not confirm:
        raise ConfirmationRequired(
            f"{symbol}: this would BILL about ${cost:.2f} for "
            f"{start:%Y-%m-%d}..{end_ts:%Y-%m-%d} ({len(pending)} chunk(s) not "
            "on disk). Re-run with confirm=True only after the operator has "
            "approved the spend."
        )

    problems: list[str] = []
    for i in pending:
        a, b = chunks[i]
        bars, metas = _fetch_chunk(cfg, symbol_cfg, a, b)
        problems += validate(bars, tick_size=tick, symbol=symbol, strict=False) \
            if len(bars) else []
        bp, mp = paths[i]
        _atomic_parquet(bars, bp)                  # bars first: a hit needs both
        _atomic_parquet(_metas_to_frame(metas), mp)

    if not assemble:
        return FetchResult(pd.DataFrame(), {}, cost, not pending, problems)

    frames, metas_all = [], {}
    for bp, mp in paths:
        frames.append(pd.read_parquet(bp))
        _merge_metas(metas_all, _metas_from_frame(pd.read_parquet(mp)), symbol)
    frames = [f for f in frames if len(f)]
    bars = (normalize(pd.concat(frames, ignore_index=True)) if frames
            else pd.DataFrame(columns=["ts", "raw_symbol", "open", "high", "low",
                                       "close", "volume"]))
    problems = validate(bars, tick_size=tick, symbol=symbol, strict=False) if len(bars) else []
    return FetchResult(bars, metas_all, cost, not pending, problems)


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
