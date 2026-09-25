"""Stage 2 confidence scoring -- "how good is this setup relative to others."

Six weighted components from docs/confidence_scoring_formula_spec.md, all
weights and scores in config/scoring_weights.yaml:

    score = 100 x [ 0.25 trigger_quality      + 0.20 confirmation_strength
                  + 0.20 level_confluence     + 0.15 directional_context
                  + 0.15 reward_risk_quality  + 0.05 volatility_fit ]

Only a setup that passed every Stage 1 gate is scored. The spec is explicit
that the score never launders a gate failure, so `score()` refuses anything
else rather than returning a low number.

## Where the spec text was amended or silent

Each is a real-data finding, recorded in docs/open_questions.md #16:

- CLV is SIGNED in the trade direction and floored at 0. `abs(CLV)` credited
  a long whose trigger bar closed at its low as fully as one at its high.
- A reversal WITH the HTF trend scores 1.0 context. The spec named fading
  (0.2), neutral (0.5) and exhaustion (1.0) only -- 31% of reversal setups
  fell outside all three.
- Exhaustion is read on the entry bar BEFORE the pattern starts, on any
  configured timeframe. At the decision bar the reversal bar itself has
  broken the run being faded, so the branch was all but empty (1 of 1,559).
- Magnitude: breakout/retest uses its retest bar's rejection magnitude (S9
  defines that bar as a rejection candle); three-tail averages each tail
  bar's capped ratio so one zero-body doji cannot saturate it; failed
  breakout and range reclaim have no spec magnitude and take the midpoint.
- Level confluence counts the traded level's own type, so an isolated real
  level scores 1/3 rather than the same as open space.
- Volatility fit's reversal row is gate 5's closed reversal list.

## Unknown stays unknown

A component whose input does not exist yet (no volume baseline, HTF bias
still warming up, volatility regime unseeded) is NaN, and so is the total.
Ranking must treat NaN as "not scored", never as zero.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features import structure, time_count
from features.schema import (
    BODY,
    BODY_RATIO,
    CLV,
    CONFIG_DIR,
    LOWER_WICK,
    UPPER_WICK,
    VOL_REGIME,
    VOLUME_RATIO,
    Params,
)
from signal_engine import gates
from signal_engine.gates import (
    CONTINUATION,
    LONG,
    PASS,
    REVERSAL,
    Candidate,
    GateContext,
    GateReport,
)
from signal_engine.timeframes import ENTRY

COMPONENTS = ("trigger_quality", "confirmation_strength", "level_confluence",
              "directional_context", "reward_risk_quality", "volatility_fit")

# marked-level name -> confluence TYPE. Highs and lows of one source are one
# type: a prior-day high and low stacked together are not two kinds of evidence.
LEVEL_TYPES = {
    "prior day high": "prior_day", "prior day low": "prior_day",
    "prior week high": "prior_week", "prior week low": "prior_week",
    "major swing high": "major_swing", "major swing low": "major_swing",
    "gap edge": "gap", "range high": "range", "range low": "range",
}


def load_scoring(config_dir: Path | None = None) -> Params:
    with open((config_dir or CONFIG_DIR) / "scoring_weights.yaml",
              encoding="utf-8") as fh:
        values = yaml.safe_load(fh)
    total = sum(values["weights"].values())
    if not math.isclose(total, 1.0):
        raise ValueError(f"scoring weights sum to {total}, not 1.0")
    if set(values["weights"]) != set(COMPONENTS):
        raise ValueError(f"scoring weights must name exactly {COMPONENTS}")
    return Params(values=values, symbol="scoring")


# ==========================================================================
# context
# ==========================================================================

@dataclass
class ScoreContext:
    gates: GateContext
    weights: Params
    exhaustion: dict[str, pd.DataFrame]   # per timeframe, on the entry index

    @classmethod
    def build(cls, ctx: GateContext, *, weights: Params | None = None,
              exhaustion: dict[str, pd.DataFrame] | None = None
              ) -> "ScoreContext":
        """S18 exhaustion on each configured timeframe, aligned causally onto
        the entry frame (a higher-timeframe count is visible once its bar
        closes). Computed per timeframe and never blended (spec S18)."""
        if exhaustion is None:
            tfs, p = ctx.tfs, ctx.params
            exhaustion = {}
            for tf in p.get("time_count.timeframes"):
                ex = time_count.exhaustion(tfs.by_freq(tf)["close"], p)
                if tf == tfs.freq(ENTRY):
                    exhaustion[tf] = ex[["exhausted", "count_direction"]]
                    continue
                src = tfs.by_freq(tf)
                exhaustion[tf] = pd.DataFrame({
                    "exhausted": structure.align_htf(
                        tfs.entry, src, ex["exhausted"].astype(float), tf),
                    "count_direction": structure.align_htf(
                        tfs.entry, src, ex["count_direction"].astype(float), tf),
                }, index=tfs.entry.index)
        return cls(gates=ctx, weights=weights or load_scoring(),
                   exhaustion=exhaustion)


@dataclass(frozen=True)
class ScoreBreakdown:
    """Every sub-score and the inputs behind it, for the log (spec: log every
    component alongside the final number)."""
    score: float
    trigger_quality: float
    tq_base: float
    tq_magnitude: float
    tq_magnitude_basis: str
    confirmation_strength: float
    cs_volume: float
    cs_clv: float
    level_confluence: float
    lc_types: str
    directional_context: float
    dc_case: str
    reward_risk_quality: float
    volatility_fit: float
    vf_regime: str

    def row(self) -> dict:
        return {f"s_{k}" if k != "score" else "score": v
                for k, v in asdict(self).items()}


# ==========================================================================
# components
# ==========================================================================

def _dir(c: Candidate) -> int:
    return 1 if c.direction == LONG else -1


def _wick_ratio(frame: pd.DataFrame, i: int, direction: int) -> float:
    """The rejecting wick over the body; infinite on a zero body."""
    wick = frame[LOWER_WICK if direction > 0 else UPPER_WICK].iloc[i]
    body = frame[BODY].iloc[i]
    return math.inf if body == 0 else float(wick) / float(body)


def _capped(ratio: float, threshold: float) -> float:
    return min(ratio / (2.0 * threshold), 1.0)


def magnitude(c: Candidate, sc: ScoreContext) -> tuple[float, str]:
    """(magnitude_factor in [0, 1], what it was computed from)."""
    p, w = sc.gates.params, sc.weights
    frame = sc.gates.tfs.frame(c.role)
    d = _dir(c)
    if c.kind == "rejection":
        thr = float(p.get("rejection.wick_body_ratio"))
        return _capped(_wick_ratio(frame, c.idx, d), thr), "rejection wick/body"
    if c.kind == "breakout_retest":
        thr = float(p.get("rejection.wick_body_ratio"))
        return (_capped(_wick_ratio(frame, c.idx, d), thr),
                "retest bar rejection wick/body")
    if c.kind == "three_tail":
        thr = float(p.get("three_tail.wick_body_ratio"))
        per = [_capped(_wick_ratio(frame, b, d), thr) for b in c.pattern_bars]
        return float(np.mean(per)), f"mean capped wick/body over {len(per)} tail bars"
    if c.kind == "engulfing":
        mult = float(p.get("engulfing.strength_multiplier"))
        body, prior = frame[BODY].iloc[c.idx], frame[BODY].iloc[c.idx - 1]
        return _capped(float(body) / float(prior), mult), "body / prior body"
    if c.kind == "momentum":
        full = float(w.get("magnitude.momentum_full_body_ratio"))
        return min(float(frame[BODY_RATIO].iloc[c.idx]) / full, 1.0), "body ratio"
    if c.kind == "confirmation_signal":
        k = float(p.get("confirmation_signal.k_confirm_bars"))
        bars = float(c.meta.get("bars_to_resolve", k))
        return min(k / bars, 1.0), "k_confirm / bars to confirm"
    return (float(w.get("magnitude.undefined_midpoint")),
            "no magnitude defined in spec: midpoint")


def trigger_quality(c: Candidate, sc: ScoreContext):
    base = float(sc.weights.get(f"base_scores.{c.kind}"))
    floor = float(sc.weights.get("magnitude.floor_multiplier"))
    m, basis = magnitude(c, sc)
    return base * (floor + (1.0 - floor) * m), base, m, basis


def confirmation_strength(c: Candidate, sc: ScoreContext):
    """average(volume_score, clv_score), both on the trigger's own bar."""
    frame = sc.gates.tfs.frame(c.role)
    mult = float(sc.gates.params.get("volume_expansion.multiplier"))
    ratio = frame[VOLUME_RATIO].iloc[c.idx] if VOLUME_RATIO in frame else math.nan
    vol = math.nan if pd.isna(ratio) else min(float(ratio) / (2.0 * mult), 1.0)
    clv = max(0.0, float(frame[CLV].iloc[c.idx]) * _dir(c))
    return (vol + clv) / 2.0, vol, clv


def level_confluence(c: Candidate, sc: ScoreContext):
    """Distinct marked-level TYPES at the setup, its own included."""
    lh = gates.level_hits(c, sc.gates)
    if lh.status not in (PASS, gates.FAIL):
        return math.nan, ""
    types = sorted({LEVEL_TYPES[n] for n, _ in lh.hits})
    full = int(sc.weights.get("level_confluence.full_score_types"))
    return min(len(types) / full, 1.0), ",".join(types)


def _entry_anchor(c: Candidate, sc: ScoreContext) -> int:
    """The entry bar before the pattern's first bar. For a higher-frame
    pattern (S19), the last entry bar opening before that bar opened."""
    if c.role == ENTRY:
        return c.pattern_bars[0] - 1
    t0 = sc.gates.tfs.frame(c.role)["ts"].iloc[c.pattern_bars[0]]
    return int((sc.gates.entry["ts"] < t0).sum()) - 1


def directional_context(c: Candidate, sc: ScoreContext):
    v = sc.weights
    if c.kind in CONTINUATION:
        return float(v.get("directional_context.continuation")), "continuation"
    at = _entry_anchor(c, sc)
    faded = -_dir(c)                     # a long fades a DOWN run
    if at >= 0:
        for tf, ex in sc.exhaustion.items():
            if ex["exhausted"].iloc[at] == 1 and ex["count_direction"].iloc[at] == faded:
                return float(v.get("directional_context.exhausted")), f"exhausted:{tf}"
    bias = sc.gates.bias.iloc[c.decision_idx]
    if pd.isna(bias) or bias == "unknown":
        return math.nan, "unknown"
    if bias == "neutral":
        return float(v.get("directional_context.neutral")), "neutral"
    with_trend = (bias == "bullish") == (c.direction == LONG)
    case = "with_trend" if with_trend else "against_trend"
    return float(v.get(f"directional_context.{case}")), case


def reward_risk_quality(r: GateReport, sc: ScoreContext) -> float:
    p = sc.gates.params
    lo, cap = float(p.get("targets.min_reward_risk")), float(p.get("targets.rr_cap"))
    if r.plan is None or pd.isna(r.plan.rr):
        return math.nan
    return max(0.0, min((r.plan.rr - lo) / (cap - lo), 1.0))


def volatility_fit(c: Candidate, sc: ScoreContext):
    regime = sc.gates.entry[VOL_REGIME].iloc[c.decision_idx]
    if pd.isna(regime) or regime == "unknown":
        return math.nan, str(regime)
    group = "breakout" if c.kind in CONTINUATION else "reversal"
    assert c.kind in CONTINUATION | REVERSAL
    return float(sc.weights.get(f"volatility_fit.{group}.{regime}")), str(regime)


# ==========================================================================
# the score
# ==========================================================================

def score(r: GateReport, sc: ScoreContext) -> ScoreBreakdown:
    if not r.is_candidate:
        raise ValueError("only a Stage 1 candidate is scored; the score never "
                         "stands in for a failed gate")
    c = r.candidate
    tq, base, m, basis = trigger_quality(c, sc)
    cs, vol, clv = confirmation_strength(c, sc)
    lc, types = level_confluence(c, sc)
    dc, case = directional_context(c, sc)
    rrq = reward_risk_quality(r, sc)
    vf, regime = volatility_fit(c, sc)
    parts = dict(trigger_quality=tq, confirmation_strength=cs,
                 level_confluence=lc, directional_context=dc,
                 reward_risk_quality=rrq, volatility_fit=vf)
    w = sc.weights.get("weights")
    total = 100.0 * sum(w[k] * parts[k] for k in COMPONENTS)   # NaN propagates
    return ScoreBreakdown(total, tq, base, m, basis, cs, vol, clv, lc, types,
                          dc, case, rrq, vf, regime)
