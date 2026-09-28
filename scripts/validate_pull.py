"""Post-pull validation of one instrument's full cached window (Phase 1 checks).

Reads only from cache (free, no network). Checks:
  - every chunk: validate() (tick grid, OHLC sanity, monotonic timestamps),
    bars inside the chunk's own half-open window
  - contract metadata: consistent across chunks, and the year-and-month
    order (ContractMeta.sort_key, which drives the roll logic) matches the
    order of real expirations -- the check the clock-free year fix exists for
  - the continuous series: roll map with how each roll fired, every seam
    offset verified, minority-contract (roll-quality) sessions, and the
    largest non-roll jumps

Usage: python scripts/validate_pull.py SYM      (one process per instrument)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from data.continuous_contract import trade_date  # noqa: E402
from data.pipeline import (  # noqa: E402
    build_continuous,
    coverage_warnings,
    largest_non_roll_jumps,
    load_data_config,
    load_symbol_config,
    seam_report,
)
from data.sources.base import validate  # noqa: E402
from data.sources.databento_client import (  # noqa: E402
    _cache_path,
    _metas_from_frame,
    resolve_window,
    year_chunks,
)


def main(sym: str) -> int:
    cfg = load_data_config()
    scfg = load_symbol_config(sym)
    tick = scfg["contract_spec"]["tick_size"]
    start, end = resolve_window(cfg)
    bad = []

    # ---- per chunk ----------------------------------------------------------
    metas_seen: dict = {}
    conflicts: dict = {}                    # raw -> [older expirations...]
    for a, b in year_chunks(start, end):
        bars = pd.read_parquet(_cache_path(cfg, sym, a, b, "ohlcv1m"))
        metas = _metas_from_frame(pd.read_parquet(_cache_path(cfg, sym, a, b, "meta")))
        probs = validate(bars, tick_size=tick, symbol=sym, strict=False) if len(bars) else []
        inside = bool(((bars["ts"] >= a) & (bars["ts"] < b)).all()) if len(bars) else True
        if probs or not inside:
            bad.append(f"chunk {a:%Y}: {len(probs)} problems, inside window {inside} {probs[:2]}")
        for raw, m in metas.items():         # chunks oldest first: newest wins
            if raw in metas_seen and metas_seen[raw].expiration != m.expiration:
                conflicts.setdefault(raw, []).append(metas_seen[raw].expiration)
            metas_seen[raw] = m

    # ---- contract ordering -----------------------------------------------------
    by_key = sorted(metas_seen.values(), key=lambda m: m.sort_key)
    by_exp = sorted(metas_seen.values(), key=lambda m: m.expiration)
    order_ok = [m.raw_symbol for m in by_key] == [m.raw_symbol for m in by_exp]
    year_gap = [m.raw_symbol for m in metas_seen.values() if abs(m.year - m.expiration.year) > 1]
    if not order_ok:
        bad.append("contract sort order != expiration order")
    if year_gap:
        bad.append(f"year more than 1 from expiration year: {year_gap[:5]}")

    # ---- continuous series ------------------------------------------------------
    series, scfg, all_bars = build_continuous(sym, cfg)

    # ---- conflicting definitions: re-check the loader's rule from raw bars --
    within = pd.Timedelta(hours=float(cfg["definitions"]["conflict_last_bar_within_hours"]))
    last_bar = all_bars.groupby("raw_symbol")["ts"].max()
    conflict_notes = []
    for raw, older in conflicts.items():
        chosen = metas_seen[raw].expiration
        lb = last_bar.get(raw)
        ok = lb is not None and lb <= chosen and chosen - lb <= within
        conflict_notes.append(f"{raw}: newest {chosen:%Y-%m-%d %H:%M} over "
                              f"{', '.join(f'{t:%Y-%m-%d}' for t in older)}; last bar "
                              f"{lb:%Y-%m-%d %H:%M} -> {'confirmed' if ok else 'NOT CONFIRMED'}"
                              if lb is not None else f"{raw}: no bars -> NOT CONFIRMED")
        if not ok:
            bad.append(f"{raw}: conflicting definition not confirmed by bars")
    s = series.bars
    rm = series.roll_map
    reasons = Counter(str(r).split("(")[0] for r in rm["reason"])
    seams = seam_report(series, scfg, all_bars)
    offsets_ok = int(seams["offset_ok"].sum()) if len(seams) else 0
    cov = coverage_warnings(series, all_bars, cfg)
    jumps = largest_non_roll_jumps(series, n=3)

    ab = all_bars.copy()
    ab["td"] = trade_date(ab["ts"], scfg["session"]["timezone"], scfg["day_boundary"])
    dv = ab.groupby(["td", "raw_symbol"])["volume"].sum().unstack(fill_value=0)
    held = s.groupby("trade_date")["raw_symbol"].agg(lambda x: x.mode()[0])
    share = pd.Series({d: dv.loc[d, h] / dv.loc[d].sum() for d, h in held.items()
                       if d in dv.index and dv.loc[d].sum() > 0})
    minority = int((share < 0.5).sum())

    if offsets_ok != len(rm):
        bad.append(f"{len(rm) - offsets_ok} roll offset(s) do not verify")

    print(f"== {sym}: {len(all_bars):,} raw bars, {all_bars['raw_symbol'].nunique()} contracts, "
          f"series {len(s):,} bars over {s['trade_date'].nunique()} trade dates "
          f"({s['ts'].min():%Y-%m-%d}..{s['ts'].max():%Y-%m-%d})", flush=True)
    print(f"   chunks: 6 checked | contracts {len(metas_seen)}, sort order matches expiration order: "
          f"{order_ok}", flush=True)
    print(f"   rolls {len(rm)}: {dict(reasons)} | seam offsets verified {offsets_ok}/{len(rm)} | "
          f"minority-contract sessions {minority} (~{minority / max(1, len(rm)):.1f} per roll)", flush=True)
    for note in conflict_notes:
        print(f"   definition conflict: {note}", flush=True)
    for w in cov[:3]:
        print(f"   coverage: {w}", flush=True)
    for j in jumps.itertuples():
        print(f"   jump {j.ts:%Y-%m-%d %H:%M} {j.jump:g} ({j.jump / tick:.0f} ticks) "
              f"{j.kind} after {j.gap_minutes:.0f}min gap", flush=True)
    print(f"   -> {'OK' if not bad else 'PROBLEMS: ' + '; '.join(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
