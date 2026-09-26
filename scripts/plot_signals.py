"""Phase 3 validation plots: triggers, gates and scores on real charts.

For one instrument, runs the full Signal Engine (triggers -> Stage 1 gates ->
Stage 2 score) and draws a deterministic sample of setups on the entry-frame
chart, so a reader can judge by eye whether high-scoring setups look like good
trades and low-scoring ones like bad ones:

  - the 3 highest-scoring candidates
  - the 3 lowest-scoring candidates
  - 2 setups rejected by Stage 1, on different failing gates where possible

Each panel shows the pattern bars (shaded), the marked levels known at the
decision bar, the entry / stop / target plan, every gate's verdict, and the
score with its six components. The bars AFTER the decision are drawn in a
shaded region with the outcome proxy (target or stop first; a bar touching
both counts as the stop) -- for review only. Nothing downstream reads them.

Usage:  python scripts/plot_signals.py MES     -> plots/MES_signals.png
                                                  plots/MES_signals.md
One instrument per process, so memory is released between instruments.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.pipeline import build_continuous, load_data_config  # noqa: E402
from features.schema import tick_size  # noqa: E402
from signal_engine import engine, gates, scoring, timeframes  # noqa: E402

OUT = Path("plots")
BEFORE, AFTER = 40, 30               # entry bars shown either side of a decision

# Reference palette (dataviz skill): neutral candles, categorical slots for
# trade direction, status colours only beside a text label.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"
UP_BODY, DOWN_BODY = "#fcfcfb", "#52514e"
LONG_C, SHORT_C = "#2a78d6", "#eb6834"          # categorical slots 1, 2
GOOD, CRITICAL = "#0ca30c", "#d03b3b"            # status: target, stop
PATTERN_SHADE, AFTER_SHADE = "#cde2fb", "#f0efec"
GATE_ICON = {"pass": "✓", "fail": "✗", "unknown": "?", "not_evaluated": "–"}
COMP_ABBR = [("trigger_quality", "TQ"), ("confirmation_strength", "CS"),
             ("level_confluence", "LC"), ("directional_context", "DC"),
             ("reward_risk_quality", "RR"), ("volatility_fit", "VF")]


def outcome(entry: pd.DataFrame, plan, start: int) -> str:
    """Target or stop first after the decision bar; review only."""
    if plan is None or not plan.risk or plan.risk <= 0 or pd.isna(plan.target):
        return "no plan"
    hi, lo = entry["high"].to_numpy(), entry["low"].to_numpy()
    long_ = plan.target > plan.entry
    for j in range(start + 1, len(entry)):
        if (lo[j] <= plan.stop) if long_ else (hi[j] >= plan.stop):
            return f"stop first (−1R), {j - start} bars"
        if (hi[j] >= plan.target) if long_ else (lo[j] <= plan.target):
            return f"target first (+{plan.rr:.1f}R), {j - start} bars"
    return "unresolved at data end"


def pattern_span(c: gates.Candidate, ctx: gates.GateContext) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Open of the first pattern bar to close of the last, on the role frame."""
    frame = ctx.tfs.frame(c.role)
    step = pd.Timedelta(ctx.tfs.freq(c.role))
    return (frame["ts"].iloc[c.pattern_bars[0]],
            frame["ts"].iloc[c.pattern_bars[-1]] + step)


def pick(reports: list[gates.GateReport], rows: list[dict]) -> list[tuple[str, int]]:
    """(label, position in reports) -- deterministic and spread across time."""
    cand = [(i, r["score"]) for i, r in enumerate(rows)
            if r["is_candidate"] and pd.notna(r["score"])]
    cand.sort(key=lambda t: (-t[1], t[0]))
    chosen: list[tuple[str, int]] = []
    used_bars: list[int] = []

    def far(i: int) -> bool:
        d = reports[i].candidate.decision_idx
        return all(abs(d - u) > BEFORE + AFTER for u in used_bars)

    for label, seq in (("highest score", cand), ("lowest score", cand[::-1])):
        n = 0
        for i, _ in seq:
            if n == 3:
                break
            if far(i):
                chosen.append((label, i)); used_bars.append(reports[i].candidate.decision_idx); n += 1
    # rejected: directional triggers that failed a core gate, one per gate first
    rejected = [(i, r) for i, r in enumerate(reports)
                if not r.is_candidate and r.result(1).status == "pass"]
    seen_gates: set[int] = set()
    for i, r in rejected:
        g = r.failed()[0].gate
        if g not in seen_gates and far(i):
            chosen.append((f"rejected: gate {g}", i))
            used_bars.append(r.candidate.decision_idx); seen_gates.add(g)
        if len(seen_gates) == 2:
            break
    return chosen


def draw_panel(ax, label: str, r: gates.GateReport, row: dict,
               ctx: gates.GateContext) -> str:
    e = ctx.entry
    c = r.candidate
    d = c.decision_idx
    lo_i, hi_i = max(0, d - BEFORE), min(len(e) - 1, d + AFTER)
    w = e.iloc[lo_i:hi_i + 1]
    x = np.arange(lo_i, hi_i + 1)

    ax.set_facecolor(SURFACE)
    ax.axvspan(d + 0.5, hi_i + 0.5, color=AFTER_SHADE, zorder=0)
    t0, t1 = pattern_span(c, ctx)
    in_pat = np.flatnonzero(((w["ts"] >= t0) & (w["ts"] < t1)).to_numpy())
    if c.role != "entry":                    # S19: the 10min bars' span
        in_pat = np.flatnonzero(((w["ts"] + pd.Timedelta("5min") > t0)
                                 & (w["ts"] < t1)).to_numpy())
    if len(in_pat):
        ax.axvspan(x[in_pat[0]] - 0.5, x[in_pat[-1]] + 0.5,
                   color=PATTERN_SHADE, zorder=0.5)

    for xi, (_, b) in zip(x, w.iterrows()):
        up = b["close"] >= b["open"]
        ax.plot([xi, xi], [b["low"], b["high"]], color=INK_2, lw=0.8, zorder=2)
        ax.add_patch(plt.Rectangle((xi - 0.32, min(b["open"], b["close"])), 0.64,
                                   max(abs(b["close"] - b["open"]), 1e-9),
                                   facecolor=UP_BODY if up else DOWN_BODY,
                                   edgecolor=INK_2, lw=0.8, zorder=3))
    ylo, yhi = float(w["low"].min()), float(w["high"].max())

    plan = r.plan
    lines = []
    if plan is not None and plan.risk and plan.risk > 0:
        lines = [("entry", plan.entry, INK, "--"), ("stop", plan.stop, CRITICAL, "-"),
                 ("target", plan.target, GOOD, "-")]
        ylo = min(ylo, plan.stop, plan.target, plan.entry)
        yhi = max(yhi, plan.stop, plan.target, plan.entry)
    pad = (yhi - ylo) * 0.08 or tick_size(ctx.params) * 4
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xlim(lo_i - 0.5, hi_i + 6)

    labels = []                      # (price, text, is_plan) for the margin
    for name, price in gates.marked_levels(ctx, d):
        if ylo - pad <= price <= yhi + pad:
            ax.axhline(price, color=MUTED, lw=0.7, ls=":", zorder=1)
            labels.append((price, name, False))
    for name, price, colour, ls in lines:
        ax.hlines(price, d, hi_i, color=colour, lw=1.4, ls=ls, zorder=4)
        labels.append((price, f"{name} {price:g}", True))
    # Right-margin labels, spaced so none overprints another: plan labels are
    # placed first at their exact height, level labels move out of the way.
    gap = (yhi - ylo + 2 * pad) * 0.035
    placed: list[float] = []
    for price, text, is_plan in sorted(labels, key=lambda t: (not t[2], t[0])):
        y = price
        while any(abs(y - q) < gap for q in placed):
            y += gap * 0.5
        placed.append(y)
        ax.text(hi_i + 0.8, y, text, fontsize=6.5 if is_plan else 6,
                color=INK if is_plan else MUTED, va="center",
                fontweight="bold" if is_plan else "normal")

    colour = LONG_C if c.direction == "long" else SHORT_C
    marker = "^" if c.direction == "long" else "v"
    ypt = e["low"].iloc[d] - pad * 0.4 if c.direction == "long" else e["high"].iloc[d] + pad * 0.4
    ax.scatter([d], [ypt], marker=marker, s=70, color=colour,
               edgecolor=SURFACE, linewidth=1.5, zorder=5)
    ax.axvline(d + 0.5, color=MUTED, lw=0.6, zorder=1)

    gate_line = "  ".join(f"g{g.gate} {GATE_ICON[g.status]}" for g in r.results)
    if r.is_candidate:
        comps = " ".join(f"{a} {row[f's_{k}']:.2f}" for k, a in COMP_ABBR)
        score_line = f"score {row['score']:.1f}   {comps}"
    else:
        f = r.failed()[0]
        score_line = f"not scored — gate {f.gate} {f.status}: {f.detail}"[:95]
    res = outcome(e, plan, d)
    ts = pd.Timestamp(e["ts"].iloc[d]).tz_convert("America/Chicago")
    ax.set_title(f"{label.upper()} · {c.kind} {c.direction} · {ts:%Y-%m-%d %H:%M} CT\n"
                 f"{gate_line}\n{score_line}\nafter decision (review only): {res}",
                 fontsize=7.5, color=INK, loc="left")
    ax.tick_params(labelsize=6, colors=INK_2)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.5)
    return res


def main(symbol: str) -> None:
    series, scfg, _ = build_continuous(symbol, load_data_config())
    tfs = timeframes.build(symbol, series.bars, scfg)
    ctx = gates.GateContext.build(tfs, scfg)
    sc = scoring.ScoreContext.build(ctx)
    cands = engine.candidates(ctx)
    reports = [gates.evaluate(c, ctx) for c in cands]
    rows = [engine.report_row(r, ctx, sc) for r in reports]

    chosen = pick(reports, rows)
    fig, axes = plt.subplots(4, 2, figsize=(15, 18.5), facecolor=SURFACE)
    md = [f"# {symbol} signal validation", "",
          f"{len(reports)} triggers, {sum(r.is_candidate for r in reports)} Stage 1 "
          "candidates. Sample: 3 highest-scoring, 3 lowest-scoring, 2 rejected. "
          "Bars after the decision and the outcome column are review only.", "",
          "| panel | type | dir | decision (CT) | score | TQ | CS | LC | DC | RR | VF | gates | after decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ax, (label, i) in zip(axes.flat, chosen):
        r, row = reports[i], rows[i]
        res = draw_panel(ax, label, r, row, ctx)
        ts = pd.Timestamp(ctx.entry["ts"].iloc[r.candidate.decision_idx]).tz_convert("America/Chicago")
        comps = [f"{row[f's_{k}']:.2f}" if r.is_candidate else "" for k, _ in COMP_ABBR]
        md.append(f"| {label} | {r.candidate.kind} | {r.candidate.direction} | "
                  f"{ts:%Y-%m-%d %H:%M} | "
                  f"{row['score']:.1f} | " if r.is_candidate else
                  f"| {label} | {r.candidate.kind} | {r.candidate.direction} | "
                  f"{ts:%Y-%m-%d %H:%M} | – | ")
        md[-1] += " | ".join(comps) + " | " + " ".join(
            f"g{g.gate}{GATE_ICON[g.status]}" for g in r.results) + f" | {res} |"
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle(f"{symbol} — triggers, Stage 1 gates and Stage 2 scores on the "
                 f"{tfs.freq('entry')} chart", fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.955,
             "▲ long (blue) / ▼ short (orange) at the decision bar · blue band = pattern bars · "
             "dotted = marked levels · entry dashed, stop red, target green · "
             "grey region = after the decision, review only · "
             "✓ pass  ✗ fail  ? unknown  – not evaluated",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{symbol}_signals.png", dpi=110, facecolor=SURFACE)
    (OUT / f"{symbol}_signals.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{symbol}: {len(chosen)} panels -> {OUT / f'{symbol}_signals.png'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
