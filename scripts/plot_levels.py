"""Phase 2 visual validation: S2 prior levels, S3 gaps, S5 ranges on real charts.

The specific thing to eyeball here is session scope, since this is the first
module where the Phase 1 RTH decision actually does work:

  MES / MNQ / MYM       prior-day H/L must bracket only the SHADED RTH band
  MCL / MGC / SIL / MET prior-day H/L must bracket the whole session

On the index charts the prior-day lines should visibly ignore overnight
extremes that poke outside the shaded band. If a line tracks an overnight spike
on one of those three, the scope mask is wrong.

Gap zones are drawn from the session that created them until the session that
filled them; still-open gaps run to the right edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pipeline import build_continuous, load_data_config  # noqa: E402
from data.resample import resample  # noqa: E402
from features import levels, risk_state  # noqa: E402
from features.schema import load_params  # noqa: E402

SYMBOLS = ["MES", "MNQ", "MYM", "MCL", "MGC", "SIL", "MET"]
OUT = Path("plots")
SESSIONS_SHOWN = 10


def build(symbol: str, cfg: dict):
    series, scfg, _ = build_continuous(symbol, cfg)
    p = load_params(symbol)
    ltf = resample(series.bars, "5min", scfg)
    daily = resample(series.bars, "1D", scfg)

    a = risk_state.atr(ltf, int(p.get("atr.period")))
    _, amean = risk_state.volatility_regime(
        a, int(p.get("atr.regime_mean_window")),
        float(p.get("atr.high_vol_ratio")), float(p.get("atr.low_vol_ratio")))
    enriched = levels.apply(ltf, p, scfg, a, amean)
    gaps = levels.find_gaps(ltf, scfg,
                            levels.daily_atr_by_date(daily, int(p.get("atr.period"))), p)
    return enriched, gaps, scfg, p


def break_gaps(ts: pd.Series, y: pd.Series, max_gap_minutes=30):
    """Insert NaN across session gaps so the line does not span weekends.

    Without this matplotlib draws a straight diagonal through the entire
    weekend, which reads as price action that never happened.
    """
    gap = ts.diff().dt.total_seconds() / 60.0 > max_gap_minutes
    out = y.astype("float64").copy()
    out[gap] = float("nan")
    return out


def plot_symbol(symbol: str, cfg: dict):
    df, gaps, scfg, p = build(symbol, cfg)
    dates = sorted(df["trade_date"].unique())[-SESSIONS_SHOWN:]
    view = df[df["trade_date"].isin(dates)].copy().reset_index(drop=True)
    in_scope = levels.scope_mask(view, scfg)
    rth_only = not scfg["session"]["use_eth_range"]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 2]})

    # session scope band -- what the prior-day levels are allowed to see
    if rth_only:
        for d in dates:
            day = view[(view["trade_date"] == d) & in_scope]
            if day.empty:
                continue
            ax.axvspan(day["ts"].iloc[0], day["ts"].iloc[-1],
                       color="steelblue", alpha=0.07, lw=0)

    # consolidation regime
    rng = view[view[levels.IN_RANGE]]
    for _, block in _contiguous(rng):
        ax.axvspan(block["ts"].iloc[0], block["ts"].iloc[-1],
                   color="mediumpurple", alpha=0.13, lw=0)

    close = break_gaps(view["ts"], view["close"])
    ax.plot(view["ts"], close, lw=0.7, color="#222", zorder=3)

    # S2 levels -- step lines, constant within each session
    ax.step(view["ts"], view[levels.PRIOR_DAY_HIGH], where="post",
            color="crimson", lw=1.2, label="prior day H/L", zorder=4)
    ax.step(view["ts"], view[levels.PRIOR_DAY_LOW], where="post",
            color="crimson", lw=1.2, zorder=4)
    ax.step(view["ts"], view[levels.PRIOR_WEEK_HIGH], where="post",
            color="darkgreen", lw=1.1, ls="--", label="prior week H/L", zorder=4)
    ax.step(view["ts"], view[levels.PRIOR_WEEK_LOW], where="post",
            color="darkgreen", lw=1.1, ls="--", zorder=4)

    # session boundaries
    for d in dates:
        first = view[view["trade_date"] == d]["ts"].iloc[0]
        ax.axvline(first, color="grey", lw=0.5, alpha=0.5, zorder=1)

    # S3 gap zones, drawn from creation until fill
    shown = 0
    for g in gaps.itertuples():
        if g.trade_date not in dates:
            continue
        start = view[view["trade_date"] == g.trade_date]["ts"].iloc[0]
        if pd.notna(g.filled_date) and g.filled_date in dates:
            end = view[view["trade_date"] == g.filled_date]["ts"].iloc[-1]
            face, edge = "darkorange", "darkorange"
        else:
            end = view["ts"].iloc[-1]
            face, edge = "grey", "dimgrey"
        # Light fill with a defined edge: on the index micros nearly every
        # session gaps at a 0.15xATR threshold, so a heavy fill buries the
        # price action entirely.
        ax.add_patch(mpatches.Rectangle(
            (mdates.date2num(start), g.zone_low),
            mdates.date2num(end) - mdates.date2num(start),
            g.zone_high - g.zone_low,
            facecolor=face, alpha=0.10, edgecolor=edge, lw=0.8,
            linestyle="-", zorder=2))
        shown += 1

    scope_txt = "RTH only (08:30-15:15 CT)" if rth_only else "full session"
    ax.set_title(
        f"{symbol} — prior-day / prior-week levels, gap zones, consolidation   "
        f"[scope: {scope_txt}]   last {len(dates)} sessions",
        fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.2)

    handles = [
        plt.Line2D([], [], color="crimson", lw=1.2, label="prior day H/L"),
        plt.Line2D([], [], color="darkgreen", lw=1.1, ls="--", label="prior week H/L"),
        mpatches.Patch(color="darkorange", alpha=0.3, label="gap zone (filled)"),
        mpatches.Patch(color="grey", alpha=0.3, label="gap zone (open)"),
        mpatches.Patch(color="mediumpurple", alpha=0.2, label="consolidation (S5)"),
    ]
    if rth_only:
        handles.append(mpatches.Patch(color="steelblue", alpha=0.15,
                                      label="RTH band (level scope)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=3)

    # ---- panel 2: one session zoomed, proving the scope --------------------
    # The claim under test: the high/low of the SHADED in-scope bars on day D
    # are exactly the prior-day levels drawn on day D+1. For the index micros
    # that means overnight extremes outside the band must be ignored.
    zoom_dates = dates[-2:]
    z = view[view["trade_date"].isin(zoom_dates)]
    zmask = levels.scope_mask(z, scfg)
    d0, d1 = zoom_dates

    ax2.plot(z["ts"], break_gaps(z["ts"], z["close"]), lw=0.9, color="#222", zorder=3)
    day0 = z[(z["trade_date"] == d0)]
    scoped0 = z[(z["trade_date"] == d0) & zmask]
    if not scoped0.empty:
        ax2.axvspan(scoped0["ts"].iloc[0], scoped0["ts"].iloc[-1],
                    color="steelblue", alpha=0.16, lw=0,
                    label="in-scope bars, day D")
        hi, lo = scoped0["high"].max(), scoped0["low"].min()
        ax2.axhline(hi, color="steelblue", lw=1.6, ls=":")
        ax2.axhline(lo, color="steelblue", lw=1.6, ls=":")
        ax2.annotate(f"in-scope H {hi:.2f}", (day0["ts"].iloc[0], hi),
                     fontsize=8, color="steelblue", va="bottom")
        ax2.annotate(f"in-scope L {lo:.2f}", (day0["ts"].iloc[0], lo),
                     fontsize=8, color="steelblue", va="top")
        # full-session extremes, to show what RTH scope excludes
        if rth_only:
            fhi, flo = day0["high"].max(), day0["low"].min()
            ax2.axhline(fhi, color="grey", lw=0.9, ls="--")
            ax2.axhline(flo, color="grey", lw=0.9, ls="--")
            ax2.annotate(f"full-session H {fhi:.2f} (excluded)",
                         (day0["ts"].iloc[0], fhi), fontsize=8, color="grey",
                         va="bottom")

    nxt = z[z["trade_date"] == d1]
    if not nxt.empty:
        ax2.step(nxt["ts"], nxt[levels.PRIOR_DAY_HIGH], where="post",
                 color="crimson", lw=1.8)
        ax2.step(nxt["ts"], nxt[levels.PRIOR_DAY_LOW], where="post",
                 color="crimson", lw=1.8)
        ax2.axvline(nxt["ts"].iloc[0], color="grey", lw=0.8)
        ax2.annotate("day D+1: prior-day levels\nmust equal the dotted lines",
                     (nxt["ts"].iloc[0], nxt["close"].iloc[0]),
                     xytext=(10, 18), textcoords="offset points", fontsize=8,
                     color="crimson")

    ax2.set_title(f"scope proof — day D = {d0}, day D+1 = {d1}", fontsize=10,
                  loc="left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{symbol}_levels.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)

    return {"symbol": symbol, "path": path, "sessions": len(dates),
            "gaps_shown": shown, "gaps_total": len(gaps),
            "scope": scope_txt, "px": float(view["close"].median()),
            "in_range": float(view[levels.IN_RANGE].mean()),
            "view": view, "scfg": scfg}


def _contiguous(df):
    """Yield contiguous blocks of consecutive rows."""
    if df.empty:
        return
    grp = (df.index.to_series().diff() != 1).cumsum()
    for k, block in df.groupby(grp):
        yield k, block


def main() -> int:
    cfg = load_data_config()
    for sym in SYMBOLS:
        try:
            r = plot_symbol(sym, cfg)
        except Exception as exc:
            print(f"{sym}: FAILED {type(exc).__name__}: {exc}")
            continue

        # scope assertion, printed rather than asserted so every symbol reports
        v, scfg = r["view"], r["scfg"]
        mask = levels.scope_mask(v, scfg)
        ok = "n/a (full session)"
        if not scfg["session"]["use_eth_range"]:
            bad = 0
            for d in sorted(v["trade_date"].unique())[1:]:
                prev = v[v["trade_date"] < d]
                if prev.empty:
                    continue
                pd_hi = v.loc[v["trade_date"] == d, levels.PRIOR_DAY_HIGH].iloc[0]
                prev_day = prev["trade_date"].max()
                scoped = v[(v["trade_date"] == prev_day) & mask]
                if scoped.empty or pd.isna(pd_hi):
                    continue
                if abs(pd_hi - scoped["high"].max()) > 1e-9:
                    bad += 1
            ok = "OK" if bad == 0 else f"{bad} MISMATCH"
        print(f"{r['symbol']:5} {r['scope']:<26} px~{r['px']:>9.2f}  "
              f"gaps {r['gaps_shown']}/{r['gaps_total']:<3} "
              f"in_range {r['in_range']:>5.1%}  scope-check: {ok:<14} -> {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
