"""Stage 1 hard gates -- "is this a valid setup at all."

Six boolean gates from the confidence-scoring spec, all of which must pass
before a setup reaches Stage 2 scoring. A setup that fails one is not a
low-scoring candidate; it is not a candidate.

    1  structure valid     a recognised, DIRECTIONAL trigger fired
    2  level present       at or near a marked level (S2-S6)
    3  confirmation        volume expansion (S12) on the trigger's own bar
    4  reward/risk         RR >= min_reward_risk, with the Part 0 stop
    5  HTF alignment       continuation triggers only: bias must match
    6  risk-control veto   injected; not evaluated until the risk engine exists

## Four outcomes, not two

A gate is `pass`, `fail`, `unknown` or `not_evaluated`. The last two exist
because this build does not collapse "not yet knowable" into a value that
looks real:

- `unknown`: the inputs do not exist yet, e.g. no volume baseline, HTF bias
  still warming up, or ATR not seeded. Unknown is never a pass.
- `not_evaluated`: the gate was not run. Gate 6 reports this until
  `risk_engine/controls.py` is attached (Phase 3 is log-only). Gates 2-5
  report it when gate 1 already found the trigger non-directional, since
  there is no trade direction to evaluate them against.

A setup is a candidate only when gates 1-5 all `pass` and gate 6 is `pass` or
`not_evaluated`. Every gate is still evaluated when an earlier one fails, so
the log shows everything that is wrong with a setup, not just the first thing.

## Positions, not labels

Every index here is POSITIONAL (`iloc`), on the frame named by the
candidate's `role`. `decision_idx` is always on the entry frame: it is the bar
at which the setup can be acted on. For entry-frame triggers that is the
trigger bar itself. For S19 it is the entry bar its 10min bar landed on
(`TimeframeSet.align_events`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from features import levels, structure, triggers
from features.confirmation_signal import assert_single_mode
from features.schema import (
    ATR,
    ATR_MEAN,
    VOLUME_BASELINE,
    VOLUME_EXPANDED,
    tick_size,
)
from signal_engine import candles
from signal_engine.timeframes import ENTRY, TimeframeSet

LONG, SHORT = triggers.LONG, triggers.SHORT
PASS, FAIL, UNKNOWN, NOT_EVALUATED = "pass", "fail", "unknown", "not_evaluated"

# Spec S14 / scoring gate 5. The reversal list is closed on purpose: a
# breakout/retest against the trend is not thereby a reversal, so every
# breakout/retest is continuation-type and must match the HTF bias.
CONTINUATION = frozenset({"momentum", "breakout_retest", "confirmation_signal"})
REVERSAL = frozenset({"rejection", "three_tail", "failed_breakout",
                      "range_reclaim", "engulfing"})
TRIGGER_KINDS = CONTINUATION | REVERSAL

# Triggers DEFINED against a level. Gate 2 checks that level; the level-free
# patterns are checked by where their bars sit instead.
LEVEL_DEFINED = frozenset({"breakout_retest", "failed_breakout",
                           "range_reclaim", "confirmation_signal"})

# S17 is an alternative to S4's buffer breakout, never both on one symbol.
# breakout.mode decides which of the two breakout-entry triggers exists.
MODE_OF = {"breakout_retest": "buffer",
           "confirmation_signal": "confirmation_signal"}

GATE_NAMES = {1: "structure", 2: "level", 3: "confirmation", 4: "reward_risk",
              5: "htf_alignment", 6: "risk_veto"}


# ==========================================================================
# data
# ==========================================================================

@dataclass(frozen=True)
class Candidate:
    """One trigger firing, in the shape the gates read."""
    kind: str
    direction: str                     # LONG | SHORT | candles.BOTH
    role: str                          # frame the trigger was detected on
    idx: int                           # trigger bar, positional on that frame
    decision_idx: int                  # entry-frame bar it is acted on
    pattern_bars: tuple[int, ...]      # on the role frame; Part 0 stop basis
    level: float = math.nan            # the level it is defined against
    level_name: str = ""
    meta: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class GateResult:
    gate: int
    status: str
    detail: str = ""

    @property
    def name(self) -> str:
        return GATE_NAMES[self.gate]


@dataclass(frozen=True)
class TradePlan:
    """Gate 4's working, kept for scoring and the log even when RR fails."""
    entry: float
    stop: float
    target: float
    risk: float
    rr: float
    invalidation: float                # Part 0 extreme, before the buffer
    target_source: str                 # "major_level" | "2R_fallback"
    target_level_name: str = ""
    disagreement_ratio: float = math.nan
    disagreement_flag: bool = False


@dataclass(frozen=True)
class GateReport:
    candidate: Candidate
    results: tuple[GateResult, ...]
    plan: TradePlan | None = None

    def result(self, gate: int) -> GateResult:
        return next(r for r in self.results if r.gate == gate)

    @property
    def is_candidate(self) -> bool:
        core = all(self.result(g).status == PASS for g in range(1, 6))
        return core and self.result(6).status in (PASS, NOT_EVALUATED)

    @property
    def risk_evaluated(self) -> bool:
        return self.result(6).status != NOT_EVALUATED

    def failed(self) -> list[GateResult]:
        return [r for r in self.results if r.status in (FAIL, UNKNOWN)]


Veto = Callable[[Candidate], GateResult]


@dataclass
class GateContext:
    """Everything the gates read for one symbol, built once."""
    tfs: TimeframeSet
    entry: pd.DataFrame        # entry frame with S2/S5 columns attached
    pivots: pd.DataFrame       # S1 swings on the entry frame, deaths marked
    gaps: pd.DataFrame         # S3 zones, with `active_from` (first knowable ts)
    bias: pd.Series            # S14 bias aligned onto the entry frame
    veto: Veto | None = None

    @property
    def params(self):
        return self.tfs.params

    @classmethod
    def build(cls, tfs: TimeframeSet, symbol_cfg: dict,
              veto: Veto | None = None) -> "GateContext":
        p = tfs.params
        policy = str(p.get("trend.neutral_policy"))
        if policy != "continuation_only":
            raise ValueError(f"trend.neutral_policy {policy!r} is not "
                             "implemented; spec S14 defines continuation_only")
        ent = tfs.entry
        ltf = levels.apply(ent, p, symbol_cfg, ent[ATR], ent[ATR_MEAN])
        if not ltf.index.equals(ent.index):
            raise RuntimeError("levels.apply() changed the entry index")
        htf = tfs.frame("htf")
        piv = structure.mark_swing_deaths(
            structure.swings(ltf, p, htf=htf, htf_atr=tfs.atr("htf")),
            ltf["close"])
        bias = tfs.align("htf", triggers.trend_bias(htf, p), name="trend_bias")
        daily_atr = levels.daily_atr_by_date(tfs.frame("daily"),
                                             int(p.get("atr.period")))
        gaps = levels.find_gaps(ltf, symbol_cfg, daily_atr, p)
        gaps = gaps.assign(active_from=gap_active_from(ltf, gaps, symbol_cfg))
        return cls(tfs=tfs, entry=ltf, pivots=piv, gaps=gaps, bias=bias,
                   veto=veto)


def gap_active_from(entry: pd.DataFrame, gaps: pd.DataFrame,
                    symbol_cfg: dict) -> pd.Series:
    """The first entry-bar timestamp at which each gap is knowable.

    A gap is measured to its session's first IN-SCOPE open. For the RTH
    instruments the trade date starts at 17:00 the evening before, so an
    overnight bar carries the gap's trade date hours before the gap exists.
    Keying on trade date alone would hand those bars a level from the future.
    """
    if gaps.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    scoped = entry[levels.scope_mask(entry, symbol_cfg).to_numpy()]
    first = scoped.groupby("trade_date")["ts"].min()
    return pd.Series([first.get(d, pd.NaT) for d in gaps["trade_date"]],
                     index=gaps.index)


# ==========================================================================
# candidates
# ==========================================================================

def pattern_bars(kind: str, idx: int, meta: dict) -> tuple[int, ...]:
    """The bars that FORMED the pattern -- exit spec Part 0, per trigger.

    The stop goes beyond their extreme: that is where the evidence that
    justified the trade is proven wrong.
    """
    if kind in ("rejection", "momentum"):
        return (idx,)
    if kind == "engulfing":
        return (idx - 1, idx)
    if kind == "three_tail":
        return tuple(int(b) for b in meta["bars"])
    if kind in ("failed_breakout", "range_reclaim", "breakout_retest"):
        return tuple(range(int(meta["breakout_idx"]), idx + 1))
    if kind == "confirmation_signal":
        return tuple(range(int(meta["pierce_idx"]), idx + 1))
    raise ValueError(f"no Part 0 pattern definition for trigger {kind!r}")


def from_event(event: dict, *, role: str = ENTRY,
               decision_idx: int | None = None,
               level_name: str = "") -> Candidate:
    """A Candidate from one `triggers.TriggerEvent` row.

    Positions in `idx` and meta (`breakout_idx`, `bars`) must be on the FULL
    role frame. Detectors run on a slice return slice positions, and the
    caller has to map them back first.
    """
    kind, idx = str(event["kind"]), int(event["idx"])
    meta = dict(event.get("meta") or {})
    if decision_idx is None:
        if role != ENTRY:
            raise ValueError(f"a {role!r}-frame trigger needs the entry bar it "
                             "landed on (TimeframeSet.align_events)")
        decision_idx = idx
    level = event.get("level", math.nan)
    return Candidate(kind=kind, direction=str(event["direction"]), role=role,
                     idx=idx, decision_idx=int(decision_idx),
                     pattern_bars=pattern_bars(kind, idx, meta),
                     level=float(level) if pd.notna(level) else math.nan,
                     level_name=level_name, meta=meta)


def from_confirmation(row: dict, level_name: str = "") -> Candidate:
    """A Candidate from one `confirmation_signal.confirmations()` row (S17)."""
    idx = int(row["resolve_idx"])
    meta = {"pierce_idx": int(row["pierce_idx"]),
            "pierce_extreme": float(row["pierce_extreme"]),
            "bars_to_resolve": int(row["bars_to_resolve"])}
    return Candidate(kind="confirmation_signal", direction=str(row["direction"]),
                     role=ENTRY, idx=idx, decision_idx=idx,
                     pattern_bars=pattern_bars("confirmation_signal", idx, meta),
                     level=float(row["level"]), level_name=level_name,
                     meta=meta)


def level_free_candidates(tfs: TimeframeSet) -> list[Candidate]:
    """S7, S20 and S19 candidates, each read on its own role's frame.

    S19 comes through `candles.candle_triggers()`, so it arrives already
    aligned: one candidate per landed 10min bar, direction BOTH where the bar
    completed clusters on both sides.
    """
    entry, p = tfs.entry, tfs.params
    ct = candles.candle_triggers(tfs)
    out: list[Candidate] = []
    for kind in ("rejection", "engulfing"):
        col = ct[kind].to_numpy()
        for i in np.flatnonzero(pd.notna(col)):
            out.append(from_event({"kind": kind, "idx": int(i),
                                   "direction": col[i], "meta": {}}))

    tt = tfs.frame(candles.THREE_TAIL)
    events = triggers.three_tail(tt, tt, tt[ATR], p)
    dirs = ct[candles.THREE_TAIL].to_numpy()
    src = ct[f"{candles.THREE_TAIL}_ts"]
    tt_pos = pd.Series(np.arange(len(tt)), index=tt["ts"])
    for j in np.flatnonzero(pd.notna(dirs)):
        i = int(tt_pos[src.iloc[j]])
        on_bar = events[events["idx"] == i]
        bars = sorted({b for m in on_bar["meta"] for b in m["bars"]})
        side = on_bar[on_bar["direction"] == dirs[j]]
        level = float(side["level"].iloc[0]) if len(side) else math.nan
        out.append(Candidate(
            kind="three_tail", direction=str(dirs[j]), role=candles.THREE_TAIL,
            idx=i, decision_idx=int(j), pattern_bars=tuple(bars), level=level,
            level_name="tail cluster",
            meta={"events": on_bar.drop(columns=["meta"]).to_dict("records"),
                  "bars": bars}))
    return sorted(out, key=lambda c: (c.decision_idx, c.kind))


# ==========================================================================
# marked levels (S2-S6, plus confirmed major swings)
# ==========================================================================

def marked_levels(ctx: GateContext, i: int) -> list[tuple[str, float]]:
    """Every marked level knowable at entry bar `i`.

    S2 prior day/week come from columns already built only from completed
    prior periods. S5 range edges are the current compression window, which
    ends at bar i. S3 gap edges count from the gap's first in-scope bar until
    the session it fills (inclusive: the fill date is only known to the
    session, not the bar). Major swings (S1) count only while LIVE
    (`structure.live_major_swings()`): confirmed, classified major, and not
    yet closed beyond. Counting every swing in the history left ~470 levels
    live at a typical bar, which made gate 2 pass almost anything.
    """
    row = ctx.entry.iloc[i]
    out: list[tuple[str, float]] = []
    for name, col in (("prior day high", levels.PRIOR_DAY_HIGH),
                      ("prior day low", levels.PRIOR_DAY_LOW),
                      ("prior week high", levels.PRIOR_WEEK_HIGH),
                      ("prior week low", levels.PRIOR_WEEK_LOW),
                      ("range high", levels.RANGE_HIGH),
                      ("range low", levels.RANGE_LOW)):
        if col in row.index and pd.notna(row[col]):
            out.append((name, float(row[col])))

    if not ctx.gaps.empty:
        ts, day = row["ts"], pd.Timestamp(row["trade_date"])
        g = ctx.gaps
        filled = pd.to_datetime(g["filled_date"])
        live = ((g["active_from"] <= ts)
                & (filled.isna() | (filled >= day)))
        for z in g[live.fillna(False)].itertuples():
            out += [("gap edge", float(z.zone_low)),
                    ("gap edge", float(z.zone_high))]

    sw = structure.live_major_swings(ctx.pivots, i)
    for kind, price in zip(sw["kind"], sw["price"]):
        out.append((f"major swing {kind}", float(price)))
    return out


# ==========================================================================
# the gates
# ==========================================================================

def gate_structure(c: Candidate, ctx: GateContext) -> GateResult:
    """1. A recognised trigger, with a direction.

    A two-sided S19 bar is NO TRADE (spec S19): contradictory evidence, not a
    direction, so it fails here, explicitly, rather than defaulting to a side.
    """
    if c.kind not in TRIGGER_KINDS:
        return GateResult(1, FAIL, f"{c.kind!r} is not a Stage 1 trigger type")
    if c.direction == candles.BOTH:
        return GateResult(1, FAIL, "two-sided bar: no trade (spec S19)")
    if c.direction not in (LONG, SHORT):
        return GateResult(1, FAIL, f"no trade direction ({c.direction!r})")
    mode = assert_single_mode(ctx.params)
    if c.kind in MODE_OF and MODE_OF[c.kind] != mode:
        return GateResult(1, FAIL, f"{c.kind} does not exist in breakout.mode "
                                   f"{mode!r} (S17 is an alternative to S4)")
    return GateResult(1, PASS, c.kind)


def _near(price: float, marks: list[tuple[str, float]], tol: float
          ) -> list[str]:
    return [n for n, lv in marks if abs(price - lv) <= tol]


def gate_level(c: Candidate, ctx: GateContext) -> GateResult:
    """2. At or near a marked level -- the S6 test zone of one.

    Exempt: momentum (a minor level satisfies it, per S11), and three-tail
    while `three_tail.requires_nearby_level` is false (scoring gate 2).
    Level-defined triggers are checked on their own level; the level-free
    patterns on whether any of their bars reached a marked level's zone.
    """
    p = ctx.params
    if c.kind == "momentum":
        return GateResult(2, PASS, "exempt: momentum fires off a minor level")
    if c.kind == "three_tail" and not bool(p.get("three_tail.requires_nearby_level")):
        return GateResult(2, PASS, "exempt: three-tail may fire in open space")

    atr = ctx.entry[ATR].iloc[c.decision_idx]
    if pd.isna(atr):
        return GateResult(2, UNKNOWN, "ATR not seeded; no test-zone tolerance")
    tol = float(triggers.test_zone(pd.Series([atr]), p).iloc[0])
    marks = marked_levels(ctx, c.decision_idx)

    if c.kind in LEVEL_DEFINED or c.kind == "three_tail":
        if pd.isna(c.level):
            return GateResult(2, FAIL, f"{c.kind} carries no level")
        hits = _near(c.level, marks, tol)
    else:
        frame = ctx.tfs.frame(c.role)
        hits = []
        for b in c.pattern_bars:
            hi, lo = float(frame["high"].iloc[b]), float(frame["low"].iloc[b])
            for name, lv in marks:
                if (lo <= lv <= hi) or min(abs(hi - lv), abs(lo - lv)) <= tol:
                    hits.append(name)
    if hits:
        return GateResult(2, PASS, ", ".join(sorted(set(hits))))
    return GateResult(2, FAIL, f"no marked level within {tol:g}")


def gate_confirmation(c: Candidate, ctx: GateContext) -> GateResult:
    """3. Volume expansion (S12) on the bar the trigger completes on.

    That bar is on the trigger's own frame, which for S19 is the 10min bar,
    not the entry bar it landed on. CLV never substitutes, not even for a
    rejection candle that already passed its own CLV test.

    `VOLUME_EXPANDED` is False when there is no baseline yet, so the baseline
    is read directly: no baseline is `unknown`, not a failure of volume.
    """
    frame = ctx.tfs.frame(c.role)
    if VOLUME_EXPANDED not in frame.columns:
        return GateResult(3, UNKNOWN, f"{c.role} frame carries no volume "
                                      "expansion (S12 is intraday-only)")
    base = frame[VOLUME_BASELINE].iloc[c.idx]
    if pd.isna(base) or base <= 0:
        return GateResult(3, UNKNOWN, "no volume baseline yet")
    if bool(frame[VOLUME_EXPANDED].iloc[c.idx]):
        return GateResult(3, PASS, f"volume expanded on {c.role} bar {c.idx}")
    return GateResult(3, FAIL, "no volume expansion on the trigger bar")


def entry_price(c: Candidate, ctx: GateContext) -> float:
    """Exit spec Part 2's limit price.

    Breakout/retest and S17: at the level, `entry.level_offset_ticks` in the
    trader's favour. Momentum: at its minor level. Everything else: the
    trigger bar's close. The Part 2 table names rejection, three-tail and
    engulfing for that row. It lists failed breakout and range reclaim
    nowhere, so they take the same row: like the others, price is already at
    the level on the bar that confirmed it.
    """
    if c.kind in MODE_OF:
        off = float(ctx.params.get("entry.level_offset_ticks")) * tick_size(ctx.params)
        return c.level - off if c.direction == LONG else c.level + off
    if c.kind == "momentum":
        return c.level
    return float(ctx.tfs.frame(c.role)["close"].iloc[c.idx])


def plan_trade(c: Candidate, ctx: GateContext) -> TradePlan | None:
    """Entry, Part 0 stop, S15 target and RR. None if ATR is not seeded."""
    p = ctx.params
    atr = ctx.entry[ATR].iloc[c.decision_idx]
    if pd.isna(atr):
        return None
    long_ = c.direction == LONG
    frame = ctx.tfs.frame(c.role)
    rows = frame.iloc[list(c.pattern_bars)]
    invalid = float(rows["low"].min() if long_ else rows["high"].max())
    buffer = float(p.get("stops.buffer_atr_multiple")) * float(atr)
    stop = invalid - buffer if long_ else invalid + buffer
    entry = entry_price(c, ctx)
    risk = entry - stop if long_ else stop - entry
    min_rr = float(p.get("targets.min_reward_risk"))

    if risk <= 0:
        return TradePlan(entry, stop, math.nan, risk, math.nan, invalid,
                         "none")

    # Only LIVE major swings (spec S1). A level price has already closed
    # through is not a target. It also means a live high is always above the
    # last close and a live low below it, so no swing low is ever "overhead".
    sw = structure.live_major_swings(ctx.pivots, c.decision_idx)
    beyond = sw[sw["price"] > entry] if long_ else sw[sw["price"] < entry]
    if beyond.empty:
        target = entry + min_rr * risk if long_ else entry - min_rr * risk
        return TradePlan(entry, stop, target, risk, min_rr, invalid,
                         "2R_fallback")

    nearest = beyond.loc[(beyond["price"] - entry).abs().idxmin()]
    target = float(nearest["price"])
    reward = abs(target - entry)
    rr = reward / risk
    floor = min_rr * risk
    ratio = max(reward, floor) / min(reward, floor)
    return TradePlan(entry, stop, target, risk, rr, invalid, "major_level",
                     f"major swing {nearest['kind']}", ratio,
                     ratio > float(p.get("targets.disagreement_flag_ratio")))


def gate_reward_risk(c: Candidate, ctx: GateContext
                     ) -> tuple[GateResult, TradePlan | None]:
    """4. RR >= min_reward_risk (S16), on the Part 0 stop and S15 target.

    The target is the next LIVE major level beyond entry (spec S1). 2R is
    the floor and the fallback when none exists, never a cap. A stop that does
    not sit beyond entry means there is no risk to measure. That is a
    failure, not a zero-risk trade.
    """
    plan = plan_trade(c, ctx)
    if plan is None:
        return GateResult(4, UNKNOWN, "ATR not seeded; no stop buffer"), None
    if plan.risk <= 0:
        return GateResult(4, FAIL, f"stop {plan.stop:g} is not beyond entry "
                                   f"{plan.entry:g}"), plan
    min_rr = float(ctx.params.get("targets.min_reward_risk"))
    detail = f"RR {plan.rr:.2f} to {plan.target_source}"
    if plan.disagreement_flag:
        detail += f" (flag: target is {plan.disagreement_ratio:.2f}x the 2R floor)"
    return GateResult(4, PASS if plan.rr >= min_rr else FAIL, detail), plan


def gate_htf(c: Candidate, ctx: GateContext) -> GateResult:
    """5. Continuation triggers need HTF bias in the trade direction.

    Neutral or opposite is no trade, without exception (spec S14). Reversal
    triggers are never gated by HTF state; Directional Context scores them.
    """
    if c.kind in REVERSAL:
        return GateResult(5, PASS, "reversal-type: never HTF-gated")
    bias = ctx.bias.iloc[c.decision_idx]
    if pd.isna(bias) or bias == "unknown":
        return GateResult(5, UNKNOWN, "HTF bias not yet known")
    want = "bullish" if c.direction == LONG else "bearish"
    if bias == want:
        return GateResult(5, PASS, f"HTF {bias}")
    return GateResult(5, FAIL, f"continuation {c.direction} against HTF {bias}")


def gate_veto(c: Candidate, ctx: GateContext) -> GateResult:
    """6. Risk-control veto, injected. Absent means NOT evaluated, not passed."""
    if ctx.veto is None:
        return GateResult(6, NOT_EVALUATED,
                          "no risk engine attached (Phase 3 log-only)")
    r = ctx.veto(c)
    if r.gate != 6:
        raise ValueError(f"veto returned a result for gate {r.gate}, not 6")
    return r


def evaluate(c: Candidate, ctx: GateContext) -> GateReport:
    """Run all six gates on one candidate."""
    g1 = gate_structure(c, ctx)
    if g1.status != PASS and (c.kind not in TRIGGER_KINDS
                              or c.direction not in (LONG, SHORT)):
        skip = [GateResult(g, NOT_EVALUATED, "gate 1: no directional trigger")
                for g in range(2, 6)]
        return GateReport(c, (g1, *skip, gate_veto(c, ctx)))
    g4, plan = gate_reward_risk(c, ctx)
    results = (g1, gate_level(c, ctx), gate_confirmation(c, ctx), g4,
               gate_htf(c, ctx), gate_veto(c, ctx))
    return GateReport(c, results, plan)
