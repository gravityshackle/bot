"""Phase 1 visual validation: plot each continuous contract against its rolls.

Three panels per instrument:

  1. Unadjusted continuous close, coloured per contract, roll boundaries marked.
     Seams here are CORRECT -- the series holds real traded prices. What the eye
     is checking is that seams appear ONLY at roll lines.
  2. The same series difference-back-adjusted, which should be visually
     continuous across every roll. If panel 2 still shows a step, the offset is
     wrong.
  3. Daily volume per contract, which is what drove the roll decision. The roll
     line should land just after the incoming contract's volume overtakes the
     outgoing one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.continuous_contract import daily_volume_by_contract  # noqa: E402
from data.pipeline import (  # noqa: E402
    build_continuous,
    largest_non_roll_jumps,
    load_data_config,
    seam_report,
)

SYMBOLS = ["MES", "MNQ", "MYM", "MCL", "MGC", "SIL", "MET"]
OUT = Path("plots")


def plot_symbol(symbol: str, data_cfg: dict) -> dict:
    series, scfg, all_bars = build_continuous(symbol, data_cfg)
    b = series.bars
    tick = scfg["contract_spec"]["tick_size"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(
        f"{symbol}  —  continuous contract (UNADJUSTED) — "
        f"{b['ts'].min():%Y-%m-%d} to {b['ts'].max():%Y-%m-%d}",
        fontsize=13, fontweight="bold")

    # ---- panel 1: unadjusted, coloured by contract -----------------------
    ax = axes[0]
    for i, (sym, g) in enumerate(b.groupby("raw_symbol", sort=False)):
        ax.plot(g["ts"], g["close"], lw=0.6, label=sym,
                color=plt.cm.tab10(i % 10))
    ax.set_ylabel("price (unadjusted)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)

    seams = seam_report(series, scfg, all_bars)
    for _, r in seams.iterrows():
        for a in axes:
            a.axvline(_to_dt(r["effective_date"]), color="crimson",
                      ls="--", lw=1.1, alpha=0.85)
        ax.annotate(
            f"{r['from']}→{r['to']}\n"
            f"spread {r['contract_spread']:+.4g}\n"
            f"market {r['market_move']:+.4g}\n{r['reason']}",
            xy=(_to_dt(r["effective_date"]), r["first_close_after"]),
            xytext=(8, 14), textcoords="offset points", fontsize=7.5,
            color="crimson",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.85))

    # ---- panel 2: difference-adjusted, should be continuous --------------
    ax = axes[1]
    if not series.roll_map.empty:
        adj = series.to_adjusted("difference")
        ax.plot(adj["ts"], adj["close"], lw=0.6, color="darkgreen")
        ax.set_ylabel("difference-adjusted")
        ax.set_title("back-adjusted on demand — must be continuous at every roll line",
                     fontsize=9, loc="left")
    else:
        ax.text(0.5, 0.5, "no rolls in window — nothing to adjust",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_ylabel("difference-adjusted")
    ax.grid(alpha=0.25)

    # ---- panel 3: daily volume per contract ------------------------------
    ax = axes[2]
    vol = daily_volume_by_contract(all_bars)
    active = vol.sum().nlargest(6).index
    for i, sym in enumerate(active):
        ax.plot([_to_dt(d) for d in vol.index], vol[sym], lw=1.0,
                label=sym, color=plt.cm.tab10(list(b["raw_symbol"].unique()).index(sym)
                                              % 10 if sym in set(b["raw_symbol"]) else i % 10))
    ax.set_ylabel("daily volume")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{symbol}_continuous.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)

    jumps = largest_non_roll_jumps(series, n=3)
    return {
        "symbol": symbol, "path": path, "bars": len(b),
        "contracts": b["raw_symbol"].nunique(),
        "rolls": len(series.roll_map), "seams": seams,
        "jumps": jumps, "tick": tick, "warnings": series.warnings,
        "days": b["trade_date"].nunique(),
    }


def _to_dt(d):
    import pandas as pd
    return pd.Timestamp(d).tz_localize("UTC")


def main() -> int:
    data_cfg = load_data_config()
    results = []
    for s in SYMBOLS:
        try:
            results.append(plot_symbol(s, data_cfg))
        except Exception as exc:
            print(f"{s}: FAILED {type(exc).__name__}: {exc}")
    print()
    for r in results:
        print(f"=== {r['symbol']} === {r['bars']:,} bars / {r['days']} sessions / "
              f"{r['contracts']} contracts / {r['rolls']} roll(s) -> {r['path']}")
        if len(r["seams"]):
            for _, s in r["seams"].iterrows():
                print(f"    roll {s['effective_date']}  {s['from']}->{s['to']}  "
                      f"visible {s['visible_step']:+.4g} = spread {s['contract_spread']:+.4g} "
                      f"+ market {s['market_move']:+.4g}  "
                      f"offset {'OK' if s['offset_ok'] else 'WRONG'}  [{s['reason']}]")
        else:
            print("    no rolls in window")
        j = r["jumps"]
        if len(j):
            biggest = j["jump"].iloc[0]
            print(f"    largest non-roll bar jump: {biggest:.4g} "
                  f"({biggest / r['tick']:.0f} ticks) {j['kind'].iloc[0]} "
                  f"after {j['gap_minutes'].iloc[0]:.0f}min gap at {j['ts'].iloc[0]}")
        for w in r["warnings"][:3]:
            print(f"    ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
