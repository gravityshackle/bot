"""Signal Engine orchestration: enumerate every trigger, run the gates, log.

Phase 3 is log-only: this produces the record of every trigger that fired,
what each Stage 1 gate said about it, and -- for the candidates that passed
every gate -- the Stage 2 score with every component and the inputs behind
it, for hand review against charts.

## Enumerating level-dependent triggers causally

S8, S9, S10, S11 and S17 are each defined against ONE level, and the
detectors scan a window of bars against a fixed price. The question is which
levels to scan, and over which bars.

A level is scanned only over the bars where it IS a marked level -- its
*live interval* -- plus enough bars after to let a pattern that started
inside it finish. The window extension is the longest pattern span in config,
not a constant. An event is kept only if the bar before its pattern began
lies inside the level's live interval. That is exactly what gate 2 checks for
level-defined triggers, so enumeration and gating cannot disagree about
whether a level was marked in time.

    major swing (S1)      confirmed_idx .. the bar before its death close
    prior day / week (S2) each run of bars carrying that value
    gap edge (S3)         first in-scope bar .. end of the session it fills
    range edge (S5)       each run of bars with that rolling edge (S10 only)
    minor swing (S11)     until a bare close through it, so momentum fires on
                          the FIRST cross only; highs and lows are kept apart
                          because S11's cross direction follows the side

Levels on the same tick are merged, so one price held by a prior-day high and
a swing high is scanned once and named for both.

Which breakout trigger exists is `breakout.mode`: S9 (buffer) or S17
(confirmation_signal), never both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import confirmation_signal, levels, structure, triggers
from features.schema import ATR, BODY_RATIO, VOLUME_EXPANDED, tick_size
from signal_engine import gates, scoring
from signal_engine.gates import Candidate, GateContext, GateReport
from signal_engine.timeframes import TimeframeSet

Interval = tuple[int, int]        # inclusive entry-frame positions


@dataclass(frozen=True)
class LiveLevel:
    price: float
    names: tuple[str, ...]
    intervals: tuple[Interval, ...]

    @property
    def name(self) -> str:
        return " / ".join(self.names)

    def live_at(self, i: int) -> bool:
        return any(a <= i <= b for a, b in self.intervals)


# ==========================================================================
# live intervals
# ==========================================================================

def _runs(values: pd.Series) -> list[tuple[float, Interval]]:
    """(value, (start, end)) for each run of equal, non-null values."""
    v = values.to_numpy(dtype="float64")
    out, start = [], None
    for i in range(len(v) + 1):
        cur = v[i] if i < len(v) else np.nan
        if start is not None and (np.isnan(cur) or cur != v[start]):
            out.append((float(v[start]), (start, i - 1)))
            start = None
        if start is None and i < len(v) and not np.isnan(cur):
            start = i
    return out


def _swing_intervals(pivots: pd.DataFrame, n: int, major: bool,
                     kind: str | None = None
                     ) -> list[tuple[str, float, Interval]]:
    if pivots.empty:
        return []
    flag = pivots["is_major"]
    pick = (flag.fillna(False).astype(bool) if major
            else flag.notna() & ~flag.fillna(True).astype(bool))
    if kind is not None:
        pick = pick & (pivots["kind"] == kind)
    out = []
    for r in pivots[pick.to_numpy(dtype=bool)].itertuples():
        end = n - 1 if pd.isna(r.dead_idx) else int(r.dead_idx) - 1
        if int(r.confirmed_idx) <= end:
            label = f"{'major' if major else 'minor'} swing {r.kind}"
            out.append((label, float(r.price), (int(r.confirmed_idx), end)))
    return out


def _gap_intervals(ctx: GateContext) -> list[tuple[str, float, Interval]]:
    """From the gap's first in-scope bar to the end of the session it fills,
    matching `gates.marked_levels()` (the fill is known per session)."""
    g, e = ctx.gaps, ctx.entry
    if g.empty:
        return []
    days = pd.to_datetime(pd.Series(list(e["trade_date"])))
    out = []
    for z in g.itertuples():
        if pd.isna(z.active_from):
            continue
        start = int((e["ts"] < z.active_from).sum())     # ts is sorted
        if pd.isna(z.filled_date):
            end = len(e) - 1
        else:
            end = int((days <= pd.Timestamp(z.filled_date)).sum()) - 1
        if start <= end:
            out += [("gap edge", float(z.zone_low), (start, end)),
                    ("gap edge", float(z.zone_high), (start, end))]
    return out


def _merge(raw: list[tuple[str, float, Interval]], tick: float) -> list[LiveLevel]:
    """Group by tick, union names and intervals."""
    by_tick: dict[int, dict] = {}
    for name, price, iv in raw:
        k = round(price / tick)
        slot = by_tick.setdefault(k, {"price": price, "names": [], "iv": []})
        if name not in slot["names"]:
            slot["names"].append(name)
        slot["iv"].append(iv)
    out = []
    for slot in by_tick.values():
        merged: list[list[int]] = []
        for a, b in sorted(slot["iv"]):
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out.append(LiveLevel(slot["price"], tuple(slot["names"]),
                             tuple((a, b) for a, b in merged)))
    return sorted(out, key=lambda lv: lv.price)


def marked_level_intervals(ctx: GateContext) -> list[LiveLevel]:
    """Every S1-S4 marked level with the bars it is live over.

    The same sources as `gates.marked_levels()`, minus S5 range edges: those
    are what S10 is defined against and are enumerated separately.
    """
    n, e = len(ctx.entry), ctx.entry
    raw: list[tuple[str, float, Interval]] = []
    for name, col in (("prior day high", levels.PRIOR_DAY_HIGH),
                      ("prior day low", levels.PRIOR_DAY_LOW),
                      ("prior week high", levels.PRIOR_WEEK_HIGH),
                      ("prior week low", levels.PRIOR_WEEK_LOW)):
        if col in e.columns:
            raw += [(name, v, iv) for v, iv in _runs(e[col])]
    raw += _gap_intervals(ctx)
    raw += _swing_intervals(ctx.pivots, n, major=True)
    return _merge(raw, tick_size(ctx.params))


def range_edge_intervals(ctx: GateContext) -> list[tuple[str, LiveLevel]]:
    """(side, level) for each run of a rolling S5 range edge."""
    tick = tick_size(ctx.params)
    out = []
    for side, col in (("high", levels.RANGE_HIGH), ("low", levels.RANGE_LOW)):
        raw = [(f"range {side}", v, iv) for v, iv in _runs(ctx.entry[col])]
        out += [(side, lv) for lv in _merge(raw, tick)]
    return out


def minor_level_intervals(ctx: GateContext) -> list[tuple[str, LiveLevel]]:
    """(side, level) for each minor swing price, merged within a side only.

    S11's cross direction follows the swing's side, so a minor high and a
    minor low on the same tick must stay two levels. Merged, they would be
    one level with no single side to match.
    """
    n, tick = len(ctx.entry), tick_size(ctx.params)
    return [(side, lv) for side in ("high", "low")
            for lv in _merge(_swing_intervals(ctx.pivots, n, major=False,
                                              kind=side), tick)]


# ==========================================================================
# enumeration
# ==========================================================================

def _span(params) -> int:
    """How far past a live interval a pattern that began inside it can run."""
    return max(int(params.get("breakout_retest.max_bars_to_retest")),
               int(params.get("failed_breakout.window_bars")),
               int(params.get("range_reclaim.window_bars")),
               int(params.get("confirmation_signal.k_confirm_bars")))


def _windows(lv: LiveLevel, n: int, span: int):
    for a, b in lv.intervals:
        yield a, b, min(b + 1 + span, n - 1)


def _event_dict(ev: "triggers.TriggerEvent") -> dict:
    """The record `DataFrame.to_dict("records")` gave for one event row."""
    return {"idx": ev.idx, "ts": ev.ts, "kind": ev.kind, "direction": ev.direction,
            "level": ev.level, "price": ev.price, "meta": ev.meta}


def _shift_event(ev: dict, off: int) -> dict:
    ev = dict(ev)
    ev["idx"] = int(ev["idx"]) + off
    meta = dict(ev.get("meta") or {})
    if "breakout_idx" in meta:
        meta["breakout_idx"] = int(meta["breakout_idx"]) + off
    ev["meta"] = meta
    return ev


def _keep(c: Candidate, a: int, b: int) -> bool:
    return a <= c.pattern_bars[0] - 1 <= b


def level_dependent_candidates(ctx: GateContext) -> list[Candidate]:
    """S8, S9/S17, S10 and S11 across every level's live intervals."""
    e, p = ctx.entry, ctx.params
    n, span = len(e), _span(p)
    mode = confirmation_signal.assert_single_mode(p)
    # Whole-frame arrays, built once. Each level window is a NumPy VIEW of
    # these handed to the detector cores: slicing the wide entry frame and
    # round-tripping every event through a DataFrame per window was most of
    # the engine's runtime. The public triggers.* functions wrap the same
    # cores, so both paths apply one implementation of each rule.
    rej_full = triggers.rejection(e, e, e[ATR], p).to_numpy(dtype=object)
    close_full = e["close"].to_numpy(dtype=float)
    high_full = e["high"].to_numpy(dtype=float)
    low_full = e["low"].to_numpy(dtype=float)
    ts_full = e["ts"].to_numpy()
    # also carries an excursion that began before a window into it
    buf_full = triggers.breakout_buffer(e[ATR], p).to_numpy(dtype=float)
    tol_full = triggers.test_zone(e[ATR], p).to_numpy(dtype=float)
    body_full = e[BODY_RATIO].to_numpy(dtype=float)
    vol_full = np.asarray(e[VOLUME_EXPANDED].fillna(False), dtype=bool)
    bias_full = ctx.bias.to_numpy(dtype=object)
    k_fail = int(p.get("failed_breakout.window_bars"))
    k_range = int(p.get("range_reclaim.window_bars"))
    k_retest = int(p.get("breakout_retest.max_bars_to_retest"))
    out: list[Candidate] = []

    def keep_events(events, a, b, name):
        for ev in events:
            c = gates.from_event(_shift_event(_event_dict(ev), a), level_name=name)
            if _keep(c, a, b):
                out.append(c)

    for lv in marked_level_intervals(ctx):
        for a, b, w in _windows(lv, n, span):
            v = slice(a, w + 1)
            prior = triggers.excursion_before(close_full, buf_full, lv.price, a)
            found = triggers._failed_breakout_events(
                close_full[v], buf_full[v], ts_full[v], lv.price, k_fail, prior=prior)
            if mode == "buffer":
                found += triggers._breakout_retest_events(
                    close_full[v], high_full[v], low_full[v], buf_full[v], tol_full[v],
                    rej_full[v], ts_full[v], lv.price, k_retest, prior=prior)
            keep_events(found, a, b, lv.name)
            if mode == "confirmation_signal":
                s_ = e.iloc[a:w + 1].reset_index(drop=True)
                for row in confirmation_signal.confirmations(s_, lv.price, p).to_dict("records"):
                    row = dict(row, pierce_idx=int(row["pierce_idx"]) + a,
                               resolve_idx=int(row["resolve_idx"]) + a)
                    c = gates.from_confirmation(row, level_name=lv.name)
                    if _keep(c, a, b):
                        out.append(c)

    for side, lv in range_edge_intervals(ctx):
        for a, b, w in _windows(lv, n, span):
            v = slice(a, w + 1)
            keep_events(triggers._failed_breakout_events(
                close_full[v], buf_full[v], ts_full[v], lv.price, k_range,
                kind="range_reclaim", only="up" if side == "high" else "down"),
                a, b, lv.name)

    for side, lv in minor_level_intervals(ctx):
        for a, b, _ in _windows(lv, n, 0):
            w = min(b + 1, n - 1)          # momentum is the crossing bar itself
            v = slice(a, w + 1)
            keep_events(triggers._momentum_events(
                close_full[v], body_full[v], vol_full[v], bias_full[v], ts_full[v],
                lv.price, p, side=side), a, b, lv.name)

    # No de-duplication is needed: same-tick levels are merged before scanning,
    # and one level's intervals are disjoint, so an event's anchor bar can lie
    # in only one of them.
    return sorted(out, key=lambda c: (c.decision_idx, c.kind))


def candidates(ctx: GateContext) -> list[Candidate]:
    """Every trigger that fired, level-free and level-dependent."""
    return sorted(gates.level_free_candidates(ctx.tfs)
                  + level_dependent_candidates(ctx),
                  key=lambda c: (c.decision_idx, c.kind))


# ==========================================================================
# the log
# ==========================================================================

SCORE_COLUMNS = ["score"] + [f"s_{f}" for f in
                              scoring.ScoreBreakdown.__dataclass_fields__
                              if f != "score"]


def report_row(r: GateReport, ctx: GateContext,
               sc: "scoring.ScoreContext | None" = None) -> dict:
    c, plan = r.candidate, r.plan
    row = {
        "symbol": ctx.tfs.symbol,
        "ts": ctx.entry["ts"].iloc[c.decision_idx],
        "kind": c.kind, "direction": c.direction,
        "level": c.level, "level_name": c.level_name,
        "role": c.role, "idx": c.idx, "decision_idx": c.decision_idx,
        "is_candidate": r.is_candidate, "risk_evaluated": r.risk_evaluated,
    }
    for g in r.results:
        row[f"g{g.gate}_{g.name}"] = g.status
        row[f"g{g.gate}_detail"] = g.detail
    for f in ("entry", "stop", "target", "rr", "target_source",
              "disagreement_flag"):
        row[f] = getattr(plan, f) if plan is not None else None
    # Stage 2 only for Stage 1 candidates; everything else carries no score
    # at all, never a low one.
    if sc is not None and r.is_candidate:
        row.update(scoring.score(r, sc).row())
    else:
        row.update({k: None for k in SCORE_COLUMNS})
    return row


def run(ctx: GateContext, sc: "scoring.ScoreContext | None" = None
        ) -> pd.DataFrame:
    """One row per trigger: every gate's verdict, and for candidates the full
    score breakdown. `sc` defaults to one built from `ctx`."""
    sc = sc or scoring.ScoreContext.build(ctx)
    rows = [report_row(gates.evaluate(c, ctx), ctx, sc) for c in candidates(ctx)]
    return pd.DataFrame(rows)


def run_symbol(symbol: str, base_bars: pd.DataFrame, symbol_cfg: dict,
               params=None, veto=None) -> pd.DataFrame:
    """Build the frames and context from 1m bars, then `run()`."""
    from signal_engine import timeframes
    tfs: TimeframeSet = timeframes.build(symbol, base_bars, symbol_cfg, params)
    return run(GateContext.build(tfs, symbol_cfg, veto=veto))
