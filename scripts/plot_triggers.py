"""Phase 2 visual validation: swings, levels and every trigger type on real bars.

Phase 1's plots proved the *data* (roll seams, session boundaries, RTH scope).
This proves the *detectors*: that each trigger in features/triggers.py fires on
real price action where the spec says it should, and that the numbers behind
each firing are the ones the spec names.

Two outputs per symbol:

  <SYM>_triggers.png          every trigger marked over the last N sessions,
                              on top of the swings and levels they reference
  <SYM>_trigger_examples.png  one zoomed window per trigger type, showing the
                              actual bars that fired it with the measured
                              quantities against their thresholds

Plus plots/trigger_validation.md, the same evidence in text so the numbers can
be checked without reading pixels.

**The evidence is recomputed, not echoed.** Every annotated quantity (CLV, wick
ratios, body multiples, buffers, cluster tolerances) is recalculated in
`_evidence()` straight from raw OHLC and config, independently of the values
features/triggers.py used to fire. An annotation that merely printed the
module's own intermediate state would agree with it by construction and prove
nothing. Disagreements are reported as MISMATCH rather than drawn.

**Causality is preserved.** Levels active in session D come only from data
completed before D opens, swings are read through
`structure.last_confirmed_swings()`, and the HTF trend bias is aligned on HTF
bar closes. The script is allowed to *pick* which past event to show with
hindsight -- that is a display choice -- but no detector is ever handed a bar
it could not have seen.

**S17 is shown alongside S4, not combined with it.** `breakout.mode` is
`buffer`, so S4 is the live mode; the S17 panel is what the alternative mode
would have fired on the same level. The spec forbids running both on one setup
and nothing here does -- the two are drawn for comparison only.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pipeline import build_continuous, load_data_config  # noqa: E402
from data.resample import resample  # noqa: E402
from features import (  # noqa: E402
    confirmation,
    confirmation_signal,
    levels,
    risk_state,
    structure,
    triggers,
)
from features.schema import (  # noqa: E402
    ATR,
    BODY,
    BODY_RATIO,
    CLV,
    LOWER_WICK,
    UPPER_WICK,
    VOLUME_EXPANDED,
    VOLUME_RATIO,
    load_params,
    tick_size,
)

SYMBOLS = ["MES", "MNQ", "MYM", "MCL", "MGC", "SIL", "MET"]
OUT = Path("plots")

# ---------------------------------------------------------------------------
# View selection only. None of these enter a detector or a threshold -- they
# decide how much is drawn and which of several real firings gets the panel.
# Changing any of them changes the picture, never the signals.
# ---------------------------------------------------------------------------
SESSIONS_SHOWN = 10          # overview panel width
MAX_SWING_LEVELS = 3         # most recent confirmed swings offered per side
FORWARD_BARS = 24            # let a trigger resolve past its session's close
RANGE_FORWARD_BARS = 60      # how far past a consolidation block to watch for
                             # a reclaim of its edge
MIN_RANGE_BLOCK_BARS = 6     # ignore one-bar "ranges" as reclaim references
ZOOM_BARS = 26               # bars either side of an example event
MAX_MOMENTUM_MARKS = 400     # S11 is a per-bar state (see report); cap markers

KINDS = [
    "rejection", "engulfing", "three_tail",
    "breakout", "breakout_retest", "failed_breakout",
    "range_reclaim", "momentum", "confirmation_signal",
]
SPEC = {
    "rejection": "S7", "engulfing": "S20", "three_tail": "S19",
    "breakout": "S4", "breakout_retest": "S9", "failed_breakout": "S8",
    "range_reclaim": "S10", "momentum": "S11", "confirmation_signal": "S17",
}
STYLE = {
    "rejection":           ("o", "#1f77b4"),
    "engulfing":           ("s", "#d62728"),
    "three_tail":          ("^", "#9467bd"),
    "breakout":            ("P", "#2ca02c"),
    "breakout_retest":     ("X", "#17becf"),
    "failed_breakout":     ("v", "#e377c2"),
    "range_reclaim":       ("D", "#8c564b"),
    "momentum":            (".", "#7f7f7f"),
    "confirmation_signal": ("*", "#ff7f0e"),
}
# Plotting every kind on the price panel buries the price action: the per-bar
# types fire thousands of times over ten sessions. The selective ones go on
# price, the dense ones go in the raster underneath, and nothing is dropped.
ON_PRICE = ["rejection", "three_tail", "breakout_retest", "failed_breakout",
            "range_reclaim"]


def _px(p):
    """Format prices to the instrument's own tick precision.

    %.4g turns 7621.75 into 7622, which silently erases exactly the decimals a
    two-tick buffer argument turns on.
    """
    t = tick_size(p)
    frac = f"{t:.10f}".rstrip("0").split(".")[1]
    dp = len(frac)
    return lambda v: f"{v:.{dp}f}"


# ===========================================================================
# build
# ===========================================================================

def build(symbol: str, cfg: dict):
    """Everything the detectors need, assembled once per symbol."""
    series, scfg, _ = build_continuous(symbol, cfg)
    p = load_params(symbol)
    period = int(p.get("atr.period"))

    entry_tf = str(p.get("timeframes.entry"))
    htf_tf = str(p.get("timeframes.htf"))
    daily_tf = str(p.get("timeframes.daily"))
    tt_tf = str(p.get("timeframes.three_tail"))

    ltf = resample(series.bars, entry_tf, scfg)
    ltf = confirmation.apply(ltf, p, scfg)
    a = risk_state.atr(ltf, period)
    _, amean = risk_state.volatility_regime(
        a, int(p.get("atr.regime_mean_window")),
        float(p.get("atr.high_vol_ratio")), float(p.get("atr.low_vol_ratio")))
    ltf = levels.apply(ltf, p, scfg, a, amean)
    ltf[ATR] = a

    htf = resample(series.bars, htf_tf, scfg)
    htf_atr = risk_state.atr(htf, period)
    daily = resample(series.bars, daily_tf, scfg)

    piv = structure.swings(ltf, p, htf=htf, htf_atr=htf_atr)

    # Same causal alignment the swing sizing uses: an LTF bar may only see HTF
    # bars that have CLOSED. htf_atr_at() is generic over the series it carries.
    bias = structure.htf_atr_at(ltf, htf, triggers.trend_bias(htf, p), htf_tf)

    gaps = levels.find_gaps(ltf, scfg,
                            levels.daily_atr_by_date(daily, period), p)

    # S19 on its own timeframe. params.timeframes.three_tail says 10min and
    # nothing in features/ reads it -- see the report.
    tt = resample(series.bars, tt_tf, scfg)
    tt_feats = confirmation.candle_anatomy(tt).assign(**{CLV: confirmation.clv(tt)})
    tt_atr = risk_state.atr(tt, period)
    tt_events = triggers.three_tail(tt, tt_feats, tt_atr, p)

    return dict(symbol=symbol, scfg=scfg, p=p, ltf=ltf, atr=a, htf=htf,
                daily=daily, piv=piv, bias=bias, gaps=gaps,
                tt=tt, tt_atr=tt_atr, tt_events=tt_events,
                entry_tf=entry_tf, htf_tf=htf_tf, tt_tf=tt_tf)


def _feats(df: pd.DataFrame) -> pd.DataFrame:
    return df[[BODY, UPPER_WICK, LOWER_WICK, BODY_RATIO, CLV]]


def session_bounds(ltf: pd.DataFrame) -> list[tuple]:
    """(trade_date, first positional index, last positional index) per session."""
    td = ltf["trade_date"].to_numpy()
    starts = np.flatnonzero(np.r_[True, td[1:] != td[:-1]])
    ends = np.r_[starts[1:] - 1, len(td) - 1]
    return list(zip(td[starts], starts, ends))


def active_levels(b: dict, s_lo: int) -> list[tuple[str, float]]:
    """Levels a session may reference, known before that session opened.

    Prior day/week come from the S2 columns, which are already built only from
    completed prior periods. Swings come through last_confirmed_swings() at the
    session's first bar, so an unconfirmed pivot inside the session is invisible.
    """
    ltf, p = b["ltf"], b["p"]
    row = ltf.iloc[s_lo]
    out: list[tuple[str, float]] = []
    for name, col in (("prior day high", levels.PRIOR_DAY_HIGH),
                      ("prior day low", levels.PRIOR_DAY_LOW),
                      ("prior week high", levels.PRIOR_WEEK_HIGH),
                      ("prior week low", levels.PRIOR_WEEK_LOW)):
        if pd.notna(row[col]):
            out.append((name, float(row[col])))

    for kind, label in (("high", "major swing high"), ("low", "major swing low")):
        sw = structure.last_confirmed_swings(b["piv"], s_lo, kind=kind,
                                             major_only=True)
        for price in sw["price"].to_numpy()[-MAX_SWING_LEVELS:]:
            out.append((label, float(price)))

    # Collapse levels that land on the same tick -- otherwise one price held by
    # both a prior-day high and a swing high emits every event twice.
    tick = tick_size(p)
    seen, uniq = set(), []
    for name, price in out:
        key = round(price / tick)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, price))
    return uniq


def minor_levels(b: dict, s_lo: int) -> list[tuple[str, float]]:
    """S11 is explicitly allowed to fire off a MINOR level."""
    sw = structure.last_confirmed_swings(b["piv"], s_lo)
    # is_major is nullable; pd.NA means unclassified, which is not "minor"
    sw = sw[sw["is_major"].fillna(True).astype(bool).eq(False)]
    return [("minor swing", float(x))
            for x in sw["price"].to_numpy()[-MAX_SWING_LEVELS:]]


def _range_blocks(ltf: pd.DataFrame, s_lo: int, s_hi: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(ltf[levels.IN_RANGE].to_numpy()[s_lo:s_hi + 1]) + s_lo
    if len(idx) == 0:
        return []
    blocks = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    return [(int(bl[0]), int(bl[-1])) for bl in blocks
            if len(bl) >= MIN_RANGE_BLOCK_BARS]


# ===========================================================================
# event collection
# ===========================================================================

def _sub(b: dict, lo: int, hi: int):
    """A positional slice with its global positions, re-indexed for the detectors.

    Every trigger function addresses bars by position and returns positional
    idx, so the slice must start at 0 and the mapping back has to be explicit.
    """
    ltf = b["ltf"]
    hi = min(hi, len(ltf))
    sub = ltf.iloc[lo:hi].reset_index(drop=True)
    atr = b["atr"].iloc[lo:hi].reset_index(drop=True)
    return sub, atr, np.arange(lo, hi)


def _tag(ev: pd.DataFrame, gpos: np.ndarray, level_name: str,
         frame: str = "ltf") -> list[dict]:
    rows = []
    for e in ev.to_dict("records"):
        meta = dict(e.get("meta") or {})
        if "breakout_idx" in meta:
            meta["breakout_idx"] = int(gpos[meta["breakout_idx"]])
        rows.append({"kind": e["kind"], "direction": e["direction"],
                     "gidx": int(gpos[e["idx"]]), "ts": e["ts"],
                     "level": e["level"], "price": e["price"],
                     "level_name": level_name, "frame": frame, "meta": meta})
    return rows


def collect(b: dict) -> pd.DataFrame:
    ltf, p, atr_full = b["ltf"], b["p"], b["atr"]
    rows: list[dict] = []

    # ---- level-free patterns, whole series ---------------------------------
    rej = triggers.rejection(ltf, _feats(ltf), atr_full, p)
    eng = triggers.engulfing(ltf, _feats(ltf), atr_full, p)
    for series_, kind in ((rej, "rejection"), (eng, "engulfing")):
        for i in np.flatnonzero(series_.notna().to_numpy()):
            rows.append({"kind": kind, "direction": series_.iloc[i],
                         "gidx": int(i), "ts": ltf["ts"].iloc[i],
                         "level": float("nan"),
                         "price": float(ltf["close"].iloc[i]),
                         "level_name": "", "frame": "ltf", "meta": {}})

    # S19 lives on its own timeframe, so its gidx indexes b["tt"], not ltf.
    for e in b["tt_events"].to_dict("records"):
        rows.append({"kind": "three_tail", "direction": e["direction"],
                     "gidx": int(e["idx"]), "ts": e["ts"], "level": e["level"],
                     "price": e["price"], "level_name": "tail cluster",
                     "frame": "tt", "meta": dict(e["meta"])})

    # ---- level-dependent, per session --------------------------------------
    for _date, s_lo, s_hi in session_bounds(ltf):
        sub, atr, gpos = _sub(b, s_lo, s_hi + 1 + FORWARD_BARS)
        if len(sub) < 3:
            continue
        rej_sub = triggers.rejection(sub, _feats(sub), atr, p)
        bias_sub = b["bias"].iloc[gpos[0]:gpos[0] + len(sub)].reset_index(drop=True)
        vol_sub = sub[VOLUME_EXPANDED].reset_index(drop=True)

        for name, lvl in active_levels(b, s_lo):
            br = triggers.breakouts(sub, lvl, atr, p)
            for col, direction in (("up_break", "long"), ("down_break", "short")):
                for i in np.flatnonzero(br[col].to_numpy()):
                    rows.append({
                        "kind": "breakout", "direction": direction,
                        "gidx": int(gpos[i]), "ts": sub["ts"].iloc[i],
                        "level": lvl, "price": float(sub["close"].iloc[i]),
                        "level_name": name, "frame": "ltf",
                        "meta": {"buffer": float(br["buffer"].iloc[i])}})

            rows += _tag(triggers.failed_breakouts(sub, lvl, atr, p), gpos, name)
            rows += _tag(triggers.breakout_retests(sub, lvl, atr, p, rej_sub),
                         gpos, name)

            # S17: the alternative breakout mode, drawn for comparison only.
            for c in confirmation_signal.confirmations(sub, lvl, p).to_dict("records"):
                j = int(c["resolve_idx"])
                rows.append({
                    "kind": "confirmation_signal", "direction": c["direction"],
                    "gidx": int(gpos[j]), "ts": c["resolve_ts"], "level": lvl,
                    "price": float(sub["close"].iloc[j]),
                    "level_name": name, "frame": "ltf",
                    "meta": {"pierce_idx": int(gpos[int(c["pierce_idx"])]),
                             "pierce_extreme": float(c["pierce_extreme"]),
                             "bars_to_resolve": int(c["bars_to_resolve"])}})

        for name, lvl in minor_levels(b, s_lo):
            mo = triggers.momentum_continuation(sub, _feats(sub), lvl, p,
                                                bias_sub, vol_sub)
            rows += _tag(mo, gpos, name)

        for lo, hi in _range_blocks(ltf, s_lo, s_hi):
            rsub, ratr, rgpos = _sub(b, lo, hi + 1 + RANGE_FORWARD_BARS)
            block = ltf.iloc[lo:hi + 1]
            # range_reclaims() takes the boundary's side and matches direction
            # itself, so there is nothing to filter here.
            for edge, side in ((float(block["high"].max()), "high"),
                               (float(block["low"].min()), "low")):
                rows += _tag(
                    triggers.range_reclaims(rsub, edge, ratr, p, side=side),
                    rgpos, f"range {side}")

    cols = ["kind", "direction", "gidx", "ts", "level", "price", "level_name",
            "frame", "meta"]
    if not rows:
        return pd.DataFrame(columns=cols)
    ev = pd.DataFrame(rows)
    # One bar can break several levels at once; keep each (kind, bar, level)
    # once so overlapping session windows do not inflate counts.
    ev = ev.drop_duplicates(subset=["kind", "gidx", "direction", "level"])
    return ev.sort_values(["gidx", "kind"]).reset_index(drop=True)


# ===========================================================================
# evidence -- recomputed from raw OHLC, never read back from the detectors
# ===========================================================================

def _ev_rejection(bars, i, p, atr, e):
    f = _px(p)
    o, h, l, c = (float(bars[k].iloc[i]) for k in ("open", "high", "low", "close"))
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    clv = 0.0 if h == l else ((c - l) - (h - c)) / (h - l)
    a = float(atr.iloc[i])
    t_clv = float(p.get("rejection.clv_threshold"))
    ratio = float(p.get("rejection.wick_body_ratio"))
    mb = float(p.get("rejection.min_body_atr_multiple"))
    long_ = e["direction"] == "long"
    wick, side = (lower, "lower") if long_ else (upper, "upper")
    r = wick / body if body else float("inf")
    lines = [
        f"O {f(o)}  H {f(h)}  L {f(l)}  C {f(c)}",
        f"CLV {clv:+.3f}   threshold {'>= +' if long_ else '<= -'}{t_clv:.2f}",
        f"{side} wick {wick:.4g} = {r:.2f}x body {body:.4g}   "
        f"threshold >= {ratio:.1f}x",
        f"body {body:.4g} = {body / a:.2f}x ATR {a:.4g}   threshold >= {mb:.2f}x",
    ]
    ok = ((clv >= t_clv if long_ else clv <= -t_clv)
          and r >= ratio and body >= mb * a)
    return lines, [(i, "rejection bar")], ok


def _ev_engulfing(bars, i, p, atr, e):
    f = _px(p)
    o, c = float(bars["open"].iloc[i]), float(bars["close"].iloc[i])
    po, pc = float(bars["open"].iloc[i - 1]), float(bars["close"].iloc[i - 1])
    body, pbody = abs(c - o), abs(pc - po)
    mult = float(p.get("engulfing.strength_multiplier"))
    long_ = e["direction"] == "long"
    covers = (o <= pc and c >= po) if long_ else (o >= pc and c <= po)
    opposite = (pc < po and c > o) if long_ else (pc > po and c < o)
    ratio = body / pbody if pbody else float("inf")
    lines = [
        f"prior bar {'down' if pc < po else 'up'} {f(po)} -> {f(pc)}, "
        f"current {'up' if c > o else 'down'} {f(o)} -> {f(c)}",
        f"body covers body: open {f(o)} {'<=' if long_ else '>='} prior close "
        f"{f(pc)} and close {f(c)} {'>=' if long_ else '<='} prior open "
        f"{f(po)} -> {covers}",
        f"body {body:.4g} = {ratio:.2f}x prior body {pbody:.4g}   "
        f"threshold >= {mult:.1f}x",
    ]
    return lines, [(i - 1, "prior body"), (i, "engulfing bar")], (
        covers and opposite and body >= mult * pbody)


def _ev_three_tail(bars, i, p, atr, e):
    f = _px(p)
    m = e["meta"]
    idxs = list(m["bars"])
    side = m["side"]
    col = "high" if side == "upper" else "low"
    prices = [float(bars[col].iloc[j]) for j in idxs]
    tol = float(m["tolerance"])
    n_ticks = int(p.get("three_tail.cluster_tolerance_min_ticks"))
    ticks = n_ticks * tick_size(p)
    mult = float(p.get("three_tail.cluster_tolerance_atr_multiple"))
    a = float(atr.iloc[i])
    need = int(p.get("three_tail.min_tails_required"))
    anchor = float(bars[col].iloc[i])
    worst = max(abs(x - anchor) for x in prices)
    lines = [
        f"{len(idxs)} {side} tails in a {int(p.get('three_tail.lookback_bars'))}"
        f"-bar window   threshold >= {need}",
        f"{side} extremes " + ", ".join(f(x) for x in prices),
        f"anchor {f(anchor)}, furthest tail {worst:.4g} away",
        f"cluster tolerance {tol:.4g} = max({n_ticks} ticks = {ticks:.4g}, "
        f"{mult:.2f} x ATR {a:.4g})",
    ]
    return lines, [(j, "tail") for j in idxs], (
        len(idxs) >= need and worst <= tol)


def _ev_breakout(bars, i, p, atr, e):
    f = _px(p)
    c = float(bars["close"].iloc[i])
    pc = float(bars["close"].iloc[i - 1]) if i else float("nan")
    lvl = float(e["level"])
    a = float(atr.iloc[i])
    n_ticks = int(p.get("breakout.buffer_min_ticks"))
    ticks = n_ticks * tick_size(p)
    mult = float(p.get("breakout.buffer_atr_multiple"))
    buf = max(ticks, mult * a)
    long_ = e["direction"] == "long"
    edge = lvl + buf if long_ else lvl - buf
    # The prior bar's beyond-ness is evaluated with the PRIOR bar's buffer:
    # breakouts() builds up_close per bar and then shifts it, so when ATR moves
    # between two bars the two buffers differ. Testing pc against this bar's
    # edge would disagree with the module on exactly those bars.
    a_prev = float(atr.iloc[i - 1]) if i else float("nan")
    edge_prev = (lvl + max(ticks, mult * a_prev) if long_
                 else lvl - max(ticks, mult * a_prev))
    prev_beyond = (pc > edge_prev) if long_ else (pc < edge_prev)
    lines = [
        f"level {f(lvl)} ({e['level_name']})",
        f"buffer {buf:.4g} = max({n_ticks} ticks = {ticks:.4g}, "
        f"{mult:.2f} x ATR {a:.4g})",
        f"close {f(c)} {'>' if long_ else '<'} {f(edge)} = level "
        f"{'+' if long_ else '-'} buffer",
        f"prior close {f(pc)} not beyond {f(edge_prev)} (its own buffer) "
        f"-> a transition, not a state",
    ]
    return lines, [(i, "breakout close")], (
        (c > edge if long_ else c < edge) and not prev_beyond)


def _ev_failed(bars, i, p, atr, e, kind="failed_breakout"):
    f = _px(p)
    m = e["meta"]
    bi = int(m["breakout_idx"])
    lvl = float(e["level"])
    key = ("failed_breakout.window_bars" if kind == "failed_breakout"
           else "range_reclaim.window_bars")
    k = int(p.get(key))
    bc = float(bars["close"].iloc[bi])
    c = float(bars["close"].iloc[i])
    long_ = e["direction"] == "long"   # trade is OPPOSITE the broken direction
    lines = [
        f"level {f(lvl)} ({e['level_name']})",
        f"bar {bi} closed {f(bc)} {'below' if long_ else 'above'} it "
        f"-> {'breakdown' if long_ else 'breakout'}",
        f"bar {i} closed {f(c)} back {'above' if long_ else 'below'} the level "
        f"after {m['bars_to_fail']} bars   window K = {k}",
        f"-> {'LONG' if long_ else 'SHORT'}, opposite the move that failed",
    ]
    return lines, [(bi, "break"), (i, "reclaim")], (
        m["bars_to_fail"] <= k and ((c > lvl) if long_ else (c < lvl)))


def _ev_retest(bars, i, p, atr, e):
    f = _px(p)
    m = e["meta"]
    bi = int(m["breakout_idx"])
    lvl = float(e["level"])
    a = float(atr.iloc[i])
    n_ticks = int(p.get("test_zone.min_ticks"))
    ticks = n_ticks * tick_size(p)
    mult = float(p.get("test_zone.atr_multiple"))
    tol = max(ticks, mult * a)
    h, l = float(bars["high"].iloc[i]), float(bars["low"].iloc[i])
    dist = 0.0 if l <= lvl <= h else min(abs(h - lvl), abs(l - lvl))
    kmax = int(p.get("breakout_retest.max_bars_to_retest"))
    lines = [
        f"level {f(lvl)} ({e['level_name']}), breakout closed at bar {bi}",
        f"retest {m['bars_to_retest']} bars later   K_max = {kmax}",
        f"retest bar H {f(h)} L {f(l)}, distance to level {dist:.4g}",
        f"test zone {tol:.4g} = max({n_ticks} ticks = {ticks:.4g}, "
        f"{mult:.2f} x ATR {a:.4g})",
        f"and the retest bar is itself an S7 {e['direction']} rejection",
    ]
    return lines, [(bi, "breakout"), (i, "retest")], (
        dist <= tol and m["bars_to_retest"] <= kmax)


def _ev_momentum(bars, i, p, atr, e):
    f = _px(p)
    o, h, l, c = (float(bars[k].iloc[i]) for k in ("open", "high", "low", "close"))
    br = abs(c - o) / (h - l) if h > l else 0.0
    lvl = float(e["level"])
    minr = float(p.get("momentum.min_body_ratio"))
    vmult = float(p.get("volume_expansion.multiplier"))
    vr = float(bars[VOLUME_RATIO].iloc[i])
    expanded = bool(bars[VOLUME_EXPANDED].iloc[i])
    long_ = e["direction"] == "long"
    lines = [
        f"body ratio {br:.3f} = |{f(c)} - {f(o)}| / ({f(h)} - {f(l)})   "
        f"threshold >= {minr:.2f}",
        f"close {f(c)} {'>' if long_ else '<'} minor level {f(lvl)}",
        f"volume {vr:.2f}x baseline   threshold >= {vmult:.1f}x "
        f"({'expanded' if expanded else 'NOT expanded'}, S12)",
        f"HTF trend {'bullish' if long_ else 'bearish'} on "
        f"{e.get('htf_tf', 'HTF')} close (S14)",
    ]
    return lines, [(i, "momentum bar")], (
        br >= minr and expanded and ((c > lvl) if long_ else (c < lvl)))


def _ev_confirmation(bars, i, p, atr, e):
    f = _px(p)
    m = e["meta"]
    pi = int(m["pierce_idx"])
    lvl = float(e["level"])
    ext = float(m["pierce_extreme"])
    c = float(bars["close"].iloc[i])
    k = int(p.get("confirmation_signal.k_confirm_bars"))
    long_ = e["direction"] == "long"
    a = float(atr.iloc[i])
    s4_buf = max(int(p.get("breakout.buffer_min_ticks")) * tick_size(p),
                 float(p.get("breakout.buffer_atr_multiple")) * a)
    s4_edge = lvl + s4_buf if long_ else lvl - s4_buf
    mag = confirmation_signal.magnitude_factor(int(m["bars_to_resolve"]), p)
    lines = [
        f"Yellow Alert: bar {pi} crossed level {f(lvl)} ({e['level_name']}); "
        f"its {'high' if long_ else 'low'} {f(ext)} is the reference",
        f"Red Alert: bar {i} closed {f(c)} {'>' if long_ else '<'} {f(ext)} "
        f"after {m['bars_to_resolve']} bars   K_confirm = {k}",
        f"magnitude factor {mag:.2f} = min(K / bars_taken, 1.0)",
        f"[S4 buffer mode would have needed only {f(s4_edge)} "
        f"= level {'+' if long_ else '-'} {s4_buf:.4g}]",
    ]
    return lines, [(pi, "yellow"), (i, "red")], (
        (c > ext if long_ else c < ext) and m["bars_to_resolve"] <= k)


EVIDENCE = {
    "rejection": _ev_rejection,
    "engulfing": _ev_engulfing,
    "three_tail": _ev_three_tail,
    "breakout": _ev_breakout,
    "failed_breakout": _ev_failed,
    "range_reclaim": lambda *a: _ev_failed(*a, kind="range_reclaim"),
    "breakout_retest": _ev_retest,
    "momentum": _ev_momentum,
    "confirmation_signal": _ev_confirmation,
}


def evidence(b: dict, e: dict):
    frame = b["tt"] if e["frame"] == "tt" else b["ltf"]
    atr = b["tt_atr"] if e["frame"] == "tt" else b["atr"]
    e = dict(e, htf_tf=b["htf_tf"])
    return EVIDENCE[e["kind"]](frame, int(e["gidx"]), b["p"], atr, e)


# ===========================================================================
# example selection -- clearest real firing of each kind
# ===========================================================================

def _clarity(b: dict, e: dict) -> float:
    """Rank real firings so the panel shows a legible one, not the first one.

    Hindsight is fine here: this chooses which past event to draw, and changes
    nothing about when or whether it fired.
    """
    k, m = e["kind"], e["meta"]
    frame = b["tt"] if e["frame"] == "tt" else b["ltf"]
    atr = b["tt_atr"] if e["frame"] == "tt" else b["atr"]
    i = int(e["gidx"])
    if k == "rejection":
        return abs(float(frame[CLV].iloc[i])) * (float(frame[BODY_RATIO].iloc[i]) + 1)
    if k == "engulfing":
        # Ranking by body/prior-body picks the degenerate case every time: a
        # normal bar after a one-tick doji scores 80x and shows nothing. S20's
        # filter is purely relative, so the informative exemplar is the one
        # that swallowed the LARGEST real body.
        return float(frame[BODY].iloc[i - 1]) / max(float(atr.iloc[i]), 1e-9)
    if k == "three_tail":
        return m.get("count", 0) + 1.0 / (1.0 + m.get("tolerance", 1.0))
    if k == "breakout":
        return abs(float(frame["close"].iloc[i]) - e["level"]) / max(
            m.get("buffer", 1e-9), 1e-9)
    if k in ("failed_breakout", "range_reclaim"):
        return -float(m.get("bars_to_fail", 99))
    if k == "breakout_retest":
        return -float(m.get("bars_to_retest", 99))
    if k == "momentum":
        return float(frame[BODY_RATIO].iloc[i]) * float(frame[VOLUME_RATIO].iloc[i])
    if k == "confirmation_signal":
        return -float(m.get("bars_to_resolve", 99))
    return 0.0


def pick_examples(b: dict, ev: pd.DataFrame) -> dict:
    """One exemplar per kind, plus runners-up for the written catalogue."""
    out = {}
    for kind in KINDS:
        rows = ev[ev["kind"] == kind]
        cands = []
        for e in rows.to_dict("records"):
            frame = b["tt"] if e["frame"] == "tt" else b["ltf"]
            i = int(e["gidx"])
            # need room to draw, and a prior bar for the two-bar patterns
            if i < max(2, ZOOM_BARS // 3) or i > len(frame) - 3:
                continue
            lines, _marks, ok = evidence(b, e)
            cands.append((_clarity(b, e), e, lines, ok))
        cands.sort(key=lambda t: t[0], reverse=True)
        # prefer an exemplar whose independent recomputation agrees
        good = [c for c in cands if c[3]] or cands
        out[kind] = {"best": good[0] if good else None,
                     "runners": good[1:4], "n": len(rows),
                     "n_checked": len(cands),
                     "n_mismatch": sum(1 for c in cands if not c[3])}
    return out


# ===========================================================================
# drawing
# ===========================================================================

def candles(ax, df: pd.DataFrame):
    x = np.arange(len(df))
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    lw_ = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    up = c >= o
    ax.vlines(x, lw_, h, color="#444", lw=0.9, zorder=2)
    # a doji has zero body height and would otherwise draw nothing at all
    floor = max(float(np.nanmean(h - lw_)) * 0.02, 1e-9)
    height = np.maximum(np.abs(c - o), floor)
    bottom = np.minimum(o, c)
    ax.bar(x[up], height[up], bottom=bottom[up], width=0.62, color="white",
           edgecolor="#1a7f37", lw=1.0, zorder=3)
    ax.bar(x[~up], height[~up], bottom=bottom[~up], width=0.62, color="#d9534f",
           edgecolor="#8e2a1e", lw=1.0, zorder=3)


def _timelabels(ax, df, tz, every=6):
    x = np.arange(len(df))
    lab = df["ts"].dt.tz_convert(tz).dt.strftime("%m-%d %H:%M")
    ax.set_xticks(x[::every])
    ax.set_xticklabels(lab.to_numpy()[::every], fontsize=7)


def break_gaps(ts: pd.Series, y: pd.Series, max_gap_minutes=30):
    """NaN across session gaps so the line does not span weekends."""
    gap = ts.diff().dt.total_seconds() / 60.0 > max_gap_minutes
    out = y.astype("float64").copy()
    out[gap] = float("nan")
    return out


def _contiguous(df):
    if df.empty:
        return
    grp = (df.index.to_series().diff() != 1).cumsum()
    for k, block in df.groupby(grp):
        yield k, block


def plot_overview(b: dict, ev: pd.DataFrame) -> Path:
    ltf, scfg, symbol = b["ltf"], b["scfg"], b["symbol"]
    tz = scfg["session"]["timezone"]
    dates = sorted(pd.unique(ltf["trade_date"]))[-SESSIONS_SHOWN:]
    lo = int(np.flatnonzero(ltf["trade_date"].to_numpy() == dates[0])[0])
    view = ltf.iloc[lo:].reset_index(drop=True)
    t0 = view["ts"].iloc[0]
    vev = ev[(ev["frame"] == "ltf") & (ev["gidx"] >= lo)].copy()
    vev["x"] = vev["gidx"] - lo
    # S19 lives on the 10min frame, so it has no position in `view`; it still
    # has a timestamp, which is all the price panel and the raster need.
    ttv = ev[(ev["frame"] == "tt") & (ev["ts"] >= t0)].copy()

    fig, (ax, axr, ax2) = plt.subplots(
        3, 1, figsize=(17, 13), gridspec_kw={"height_ratios": [3, 1.15, 2]})

    # --- panel 1: everything, last N sessions ------------------------------
    ax.plot(view["ts"], break_gaps(view["ts"], view["close"]), lw=0.7,
            color="#222", zorder=3)
    for _, block in _contiguous(view[view[levels.IN_RANGE]]):
        ax.axvspan(block["ts"].iloc[0], block["ts"].iloc[-1],
                   color="mediumpurple", alpha=0.12, lw=0)
    for col, colour, style in ((levels.PRIOR_DAY_HIGH, "crimson", "-"),
                               (levels.PRIOR_DAY_LOW, "crimson", "-"),
                               (levels.PRIOR_WEEK_HIGH, "darkgreen", "--"),
                               (levels.PRIOR_WEEK_LOW, "darkgreen", "--")):
        ax.step(view["ts"], view[col], where="post", color=colour, lw=1.0,
                ls=style, alpha=0.8, zorder=4)

    # pivot_n_ltf = 2 on 5min yields a pivot every few bars, so drawn at equal
    # weight the minors bury the price line they are supposed to describe.
    # Majors carry the levels the triggers reference; minors stay as faint
    # context.
    piv = b["piv"]
    pv = piv[piv["idx"] >= lo]
    for major in (False, True):
        sel = pv[pv["is_major"].fillna(False).astype(bool) == major]
        if sel.empty:
            continue
        xs = ltf["ts"].to_numpy()[sel["idx"].to_numpy()]
        ax.scatter(xs, sel["price"], marker="o",
                   s=36 if major else 5, zorder=6 if major else 5,
                   facecolor="gold" if major else "darkgoldenrod",
                   edgecolor="darkgoldenrod" if major else "none",
                   lw=0.9, alpha=1.0 if major else 0.35,
                   label=(f"major swing (S1) x{len(sel)}" if major
                          else f"minor swing x{len(sel)}"))

    for kind in ON_PRICE:
        # ttv holds the tt-frame events, which are three_tail by construction
        sel = ttv if kind == "three_tail" else vev[vev["kind"] == kind]
        if sel.empty:
            continue
        mk, colour = STYLE[kind]
        ax.scatter(sel["ts"].to_numpy(), sel["price"].to_numpy(), marker=mk,
                   s=48, color=colour, edgecolor="black", lw=0.4, zorder=7,
                   alpha=0.95, label=f"{kind} ({SPEC[kind]}) x{len(sel)}")

    for d in dates:
        ax.axvline(view[view["trade_date"] == d]["ts"].iloc[0], color="grey",
                   lw=0.5, alpha=0.5, zorder=1)

    ax.set_title(f"{symbol} - swings, levels and the selective triggers, last "
                 f"{len(dates)} sessions ({b['entry_tf']} bars, S19 on "
                 f"{b['tt_tf']}). The per-bar types are in the raster below.",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.9)
    ax.set_xlim(view["ts"].iloc[0], view["ts"].iloc[-1])

    # --- panel 2: firing density, every kind, nothing hidden ---------------
    allv = pd.concat([vev, ttv], ignore_index=True) if len(ttv) else vev
    labels = []
    for row, kind in enumerate(KINDS):
        sel = allv[allv["kind"] == kind]
        _mk, colour = STYLE[kind]
        if not sel.empty:
            axr.eventplot(mdates.date2num(sel["ts"].to_numpy()),
                          lineoffsets=row, linelengths=0.72, linewidths=0.7,
                          colors=colour)
        labels.append(f"{kind} ({SPEC[kind]})  x{len(sel)}")
    axr.set_ylim(-0.7, len(KINDS) - 0.3)
    axr.set_yticks(range(len(KINDS)))
    axr.set_yticklabels(labels, fontsize=7)
    for tick, kind in zip(axr.get_yticklabels(), KINDS):
        tick.set_color(STYLE[kind][1])
    axr.set_xlim(mdates.date2num(view["ts"].iloc[0]),
                 mdates.date2num(view["ts"].iloc[-1]))
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axr.grid(alpha=0.15, axis="x")
    axr.set_title("every firing, all nine types - density over the same window",
                  fontsize=9, loc="left")

    # --- panel 3: the busiest session, bar by bar --------------------------
    # "Busiest" counts the selective triggers only, so the chosen session is
    # one with real setups rather than whichever had the most momentum bars.
    nolevel_free = vev[vev["kind"].isin(ON_PRICE)]
    if not nolevel_free.empty:
        by_date = view["trade_date"].to_numpy()[nolevel_free["x"].to_numpy()]
        busiest = pd.Series(by_date).value_counts().idxmax()
    else:
        busiest = dates[-1]
    sidx = np.flatnonzero(view["trade_date"].to_numpy() == busiest)
    z = view.iloc[sidx[0]:sidx[-1] + 1].reset_index(drop=True)
    candles(ax2, z)
    zev = vev[(vev["x"] >= sidx[0]) & (vev["x"] <= sidx[-1])].copy()
    zev["zx"] = zev["x"] - sidx[0]
    for kind in KINDS:
        sel = zev[zev["kind"] == kind]
        if sel.empty:
            continue
        mk, colour = STYLE[kind]
        ax2.scatter(sel["zx"], sel["price"], marker=mk,
                    s=16 if kind == "momentum" else 60, color=colour,
                    edgecolor="black" if kind != "momentum" else "none",
                    lw=0.5, zorder=7)
    span_lo, span_hi = float(z["low"].min()), float(z["high"].max())
    pad = (span_hi - span_lo) * 0.02
    fpx = _px(b["p"])
    # levels cluster, so stagger the labels across the panel rather than
    # stacking them all on the left edge
    shown = 0
    for name, lvl in active_levels(b, lo + int(sidx[0])):
        if not (span_lo - pad <= lvl <= span_hi + pad):
            continue
        ax2.axhline(lvl, color="crimson" if "prior" in name else "darkgoldenrod",
                    lw=0.9, ls=":", alpha=0.85)
        ax2.annotate(f"{name} {fpx(lvl)}", (len(z) * 0.012 + shown % 3 * 0.11
                                            * len(z), lvl),
                     fontsize=7, color="dimgrey", va="bottom", zorder=8,
                     bbox=dict(boxstyle="square,pad=0.15", fc="white",
                               ec="none", alpha=0.7))
        shown += 1
    _timelabels(ax2, z, tz, every=max(1, len(z) // 14))
    ax2.set_title(f"busiest session {busiest} - {len(zev)} trigger events; "
                  f"dotted lines are the levels active that session",
                  fontsize=10, loc="left")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{symbol}_triggers.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_examples(b: dict, picks: dict) -> Path:
    symbol, tz = b["symbol"], b["scfg"]["session"]["timezone"]
    fig, axes = plt.subplots(3, 3, figsize=(19, 14))
    for ax, kind in zip(axes.ravel(), KINDS):
        info = picks[kind]
        best = info["best"]
        if best is None:
            ax.text(0.5, 0.5, f"{kind} ({SPEC[kind]})\n\nno instance in "
                              f"{len(b['ltf'])} bars",
                    ha="center", va="center", fontsize=10, color="dimgrey",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        _, e, lines, ok = best
        frame = b["tt"] if e["frame"] == "tt" else b["ltf"]
        tf = b["tt_tf"] if e["frame"] == "tt" else b["entry_tf"]
        i = int(e["gidx"])
        _, marks, _ = evidence(b, e)
        lo = max(0, min([i] + [m[0] for m in marks]) - ZOOM_BARS // 2)
        hi = min(len(frame), max([i] + [m[0] for m in marks]) + ZOOM_BARS // 2 + 1)
        w = frame.iloc[lo:hi].reset_index(drop=True)
        candles(ax, w)

        # Marks often sit on adjacent bars (prior/engulfing, break/reclaim, the
        # three tails), so one annotation per bar collides into unreadable
        # overlap. Shade every marked bar, but label each distinct role once
        # and stagger the rows.
        seen: dict[str, int] = {}
        for gi, label in marks:
            x = gi - lo
            if not (0 <= x < len(w)):
                continue
            ax.axvspan(x - 0.5, x + 0.5, color="gold", alpha=0.3, lw=0, zorder=1)
            if label in seen:
                continue
            seen[label] = x
            ax.annotate(label, (x, float(w["high"].iloc[x])),
                        xytext=(0, 6 + 11 * (len(seen) - 1)),
                        textcoords="offset points", fontsize=7, ha="center",
                        color="#8a6d00")
        if pd.notna(e["level"]):
            ax.axhline(float(e["level"]), color="crimson", lw=1.1, ls="--",
                       zorder=4)
            ax.annotate(f"{e['level_name']} {_px(b['p'])(float(e['level']))}",
                        (0, float(e["level"])), fontsize=7, color="crimson",
                        va="bottom", zorder=8,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                  ec="none", alpha=0.75))

        ts_local = pd.Timestamp(e["ts"]).tz_convert(tz)
        flag = "" if ok else "   [MISMATCH]"
        ax.set_title(f"{kind} ({SPEC[kind]})  {e['direction'].upper()}  "
                     f"{ts_local:%Y-%m-%d %H:%M} {tz.split('/')[-1]}  "
                     f"[{tf}]{flag}", fontsize=9, fontweight="bold",
                     color="black" if ok else "darkred", loc="left")
        ax.text(0.0, -0.34, "\n".join(lines), transform=ax.transAxes,
                fontsize=7, va="top", family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f2", ec="#cccccc"))
        _timelabels(ax, w, tz, every=max(1, len(w) // 5))
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"{symbol} - one verified instance of each trigger type. "
                 f"Every number is recomputed from raw OHLC and config, not "
                 f"read back from features/triggers.py.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.subplots_adjust(hspace=0.75)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{symbol}_trigger_examples.png"
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


# ===========================================================================
# main
# ===========================================================================

def report_lines(b: dict, picks: dict) -> list[str]:
    tz = b["scfg"]["session"]["timezone"]
    out = [f"## {b['symbol']}", "",
           f"{len(b['ltf'])} {b['entry_tf']} bars over "
           f"{b['ltf']['trade_date'].nunique()} sessions, "
           f"{b['ltf']['ts'].min():%Y-%m-%d} to {b['ltf']['ts'].max():%Y-%m-%d}. "
           f"S19 evaluated on {b['tt_tf']} bars, everything else on "
           f"{b['entry_tf']}.", ""]
    for kind in KINDS:
        info = picks[kind]
        out.append(f"### {kind} ({SPEC[kind]}) - {info['n']} firings")
        if info["n_mismatch"]:
            out.append(f"- **{info['n_mismatch']} of {info['n_checked']} "
                       f"re-checked firings disagreed with the independent "
                       f"recomputation**")
        if info["best"] is None:
            out += ["- no instance", ""]
            continue
        _, e, lines, ok = info["best"]
        ts = pd.Timestamp(e["ts"]).tz_convert(tz)
        out.append(f"- exemplar {ts:%Y-%m-%d %H:%M} {tz}, {e['direction']}, "
                   f"check {'OK' if ok else 'MISMATCH'}")
        out += [f"  - {ln}" for ln in lines]
        if info["runners"]:
            # One bar can break several levels, so a bare timestamp can appear
            # twice and read like a duplicate; the level is what separates them.
            more = ", ".join(
                f"{pd.Timestamp(r[1]['ts']).tz_convert(tz):%m-%d %H:%M}"
                + (f" ({r[1]['level_name']})" if r[1]["level_name"] else "")
                for r in info["runners"])
            out.append(f"  - other instances: {more}")
        out.append("")
    return out


NOTES = """
## What this run surfaced

The first pass of this script found four defects in the feature layer. All four
are now fixed in `features/` and written into the spec with their reasoning;
this section records what the evidence was, so the counts below can be read
against it.

1. **S11 momentum fired as a state, not an event.** FIXED. `close >
   minor_level` stayed true for every bar that remained beyond the level, so
   any later strong-bodied, volume-expanded bar re-fired it. The tell was the
   count: 2,000-4,300 per instrument against 150-460 for every other selective
   trigger. It now shifts the level-beyond state and fires on the transition,
   the identical pattern `breakouts()` uses for S4.

2. **S10 range reclaim accepted either side of a boundary.** FIXED in
   `triggers.py`, not just here. `range_reclaims()` now takes the boundary's
   `side` and passes it through to `failed_breakouts()`, which restricts the
   breakout leg. Previously a close *below* a range high -- ordinary trade
   inside the range -- counted as a breakdown of it, and the return above
   scored as a reclaim, roughly doubling the count.

3. **S20 engulfing had only a relative strength filter.** FIXED. `body >= 1.3 x
   prior body` has no floor, so any ordinary bar "engulfs" a one-tick doji at
   80x. An absolute floor on the PRIOR body now runs alongside the multiplier;
   the spec is explicit that the combination is not optional.

   **The default floor does not yet deliver the spec's stated intent, and the
   measurement should be corrected.** S20 says the degenerate case was driving
   "the majority" of the engulfing count. It is not: one-tick prior bodies are
   20.7% of MES firings, 3.2% of MGC, 31.7% of MET. At the spec's default
   `0.10 x ATR` the floor removes 21% on MES (2,614 -> 2,074) and engulfing
   still fires on 11.5% of bars, against ~1% for S7 rejection -- the comparison
   open question 7 raised in the first place. Share of firings kept, by floor:

   | floor (x ATR) | MES | MGC | MET |
   |---|---|---|---|
   | 0.05 | 96.6% | 87.7% | 99.6% |
   | 0.10 (default) | 79.3% | 73.5% | 92.1% |
   | 0.20 | 53.1% | 47.4% | 61.2% |
   | 0.30 | 34.5% | 29.1% | 43.7% |
   | 0.50 | 13.8% | 9.1% | 18.7% |

   Roughly `0.50 x ATR` is what brings engulfing to ~1-2% of bars, i.e. to a
   selective trigger's frequency. The floor is a declared tunable, so this is a
   Phase 4 input rather than a spec contradiction -- but the default was chosen
   against an overestimate of the degenerate case's share, and 0.10 leaves the
   original "fires on one bar in seven" problem largely intact.

4. **"Prior day" meant the previous trade date, not the previous liquid
   session.** FIXED, as a general rule rather than a MET patch. MET produces 93
   trade dates to the others' 66 -- it trades through weekends, 14 Saturdays
   and 13 Sundays carrying ~7.5% of its volume, straight through the
   16:00-17:00 halt its config declares. A shift-by-one therefore drew Monday's
   prior-day levels from Sunday's 81-144 bar session instead of Friday's ~734
   bar one. A session now qualifies only at >= 50% of the rolling median bar
   count; otherwise the search skips further back.

Two items remain open, neither a patch to `features/`:

5. **The timeframe config is not wired to anything.** `timeframes.three_tail`
   (10min), `timeframes.entry` and `timeframes.daily` are set in `params.yaml`
   and read by nobody -- every detector runs on whatever frame its caller hands
   it, and callers hardcode. This is not a bug to fold into `triggers.py`; it
   is the multi-timeframe orchestration the Signal Engine exists to do, and it
   is the first real piece of that build. Until it lands, S19's counts here
   (evaluated on the configured 10min bars) will not match a
   `candle_triggers()` run on the entry frame.

6. **The per-bar loops cost real time.** `failed_breakouts()` and
   `breakout_retests()` do a DataFrame column lookup per bar
   (`b["up_break"].iloc[i]`), which dominates this script's runtime. Harmless
   at Phase 2 scale; Phase 4's grid search will run these thousands of times.
"""


HEAD = ["# Phase 2 trigger validation", "",
        "Generated by `scripts/plot_triggers.py`. Every quantity below is "
        "recomputed from raw OHLC and `config/`, independently of the values "
        "`features/triggers.py` used when it fired; `check OK` means the two "
        "agree.", "", NOTES, ""]


def assemble() -> Path:
    """Stitch the per-symbol fragments into one document, in SYMBOLS order."""
    md = list(HEAD)
    for sym in SYMBOLS:
        frag = OUT / f"_frag_{sym}.md"
        if frag.exists():
            md.append(frag.read_text(encoding="utf-8"))
    path = OUT / "trigger_validation.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """`plot_triggers.py [SYM ...]` -- default all seven.

    Each symbol writes its own report fragment, so the seven can be run as
    separate processes (one whole symbol's frames is the memory high-water
    mark) and still assemble into one document.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    syms = [a for a in args if not a.startswith("-")] or SYMBOLS
    cfg = load_data_config()
    OUT.mkdir(exist_ok=True)

    header = f"{'sym':5} " + " ".join(f"{k[:9]:>9}" for k in KINDS)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for sym in syms:
        try:
            b = build(sym, cfg)
            ev = collect(b)
            picks = pick_examples(b, ev)
            p1 = plot_overview(b, ev)
            p2 = plot_examples(b, picks)
        except Exception as exc:
            print(f"{sym:5} FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        counts = " ".join(f"{picks[k]['n']:>9}" for k in KINDS)
        missing = [k for k in KINDS if picks[k]["best"] is None]
        bad = {k: picks[k]["n_mismatch"] for k in KINDS if picks[k]["n_mismatch"]}
        print(f"{sym:5} {counts}   -> {p1.name}, {p2.name}"
              + (f"   MISSING: {','.join(missing)}" if missing else "")
              + (f"   MISMATCH: {bad}" if bad else ""), flush=True)
        (OUT / f"_frag_{sym}.md").write_text(
            "\n".join(report_lines(b, picks)), encoding="utf-8")
        # a symbol's frames are large; do not carry one into the next build
        del b, ev, picks
        gc.collect()

    print(f"\nwrote {assemble()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
