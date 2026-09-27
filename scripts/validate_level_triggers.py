"""Independent real-data validation of the engine's level-defined triggers:
S8 failed breakout, S9 breakout/retest, S10 range reclaim, S17 confirmation.

scripts/plot_triggers.py checks these on its own older enumeration (the 3
most recent major swings per session; range edges taken over the WHOLE block,
which looks ahead), not on the events the Signal Engine produces. This
re-derives the engine's events in plain loops over raw bars, without calling
the engine's enumeration, liveness or detector code.

Taken as given (already reviewed in Phase 2): pivots (S1), prior day/week
columns (S2), gap zones and when each becomes knowable (S3), the rolling range
edges (S5), ATR and the HTF bias. Re-derived here: the S4 breakout buffer and
its transition rule, failure windows, the S6 test zone, the S7 rejection
candle, the S17 alert state machine, side rules, major-swing death (buffered
close) and the "marked on the bar before the pattern" anchor.

FORWARD: every engine event is checked against its type's rules.
REVERSE: every event the rules say should exist must be in the engine output.

S17 only exists in breakout.mode = confirmation_signal, so it is validated on
a second context built with that mode. Its alert logic is stateful, so each
level is scanned from the bar it becomes marked -- the same semantics as the
engine, stated deliberately: an alert armed before a level existed is not
about that level.

Usage: python scripts/validate_level_triggers.py SYM [SYM ...]
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.pipeline import build_continuous, load_data_config  # noqa: E402
from features import levels  # noqa: E402
from features.schema import ATR, Params, load_params, tick_size  # noqa: E402
from signal_engine import engine, gates, timeframes  # noqa: E402

KINDS = ("failed_breakout", "breakout_retest", "range_reclaim", "confirmation_signal")


class Rules:
    """The rules, restated from the spec and params -- no engine calls."""

    def __init__(self, ctx: gates.GateContext):
        e, p = ctx.entry, ctx.params
        self.ctx, self.p, self.e = ctx, p, e
        self.o, self.h, self.l, self.c = (e[k].to_numpy(dtype=float)
                                          for k in ("open", "high", "low", "close"))
        self.atr = e[ATR].to_numpy(dtype=float)
        self.n = len(e)
        self.tick = tick_size(p)
        self.buf = np.maximum(int(p.get("breakout.buffer_min_ticks")) * self.tick,
                              float(p.get("breakout.buffer_atr_multiple")) * self.atr)
        self.zone = np.maximum(int(p.get("test_zone.min_ticks")) * self.tick,
                               float(p.get("test_zone.atr_multiple")) * self.atr)
        self.k_fail = int(p.get("failed_breakout.window_bars"))
        self.k_retest = int(p.get("breakout_retest.max_bars_to_retest"))
        self.k_range = int(p.get("range_reclaim.window_bars"))
        self.k_conf = int(p.get("confirmation_signal.k_confirm_bars"))
        self.span = max(self.k_fail, self.k_retest, self.k_range, self.k_conf)
        self.rej = self._rejection()

    def key(self, level):
        return round(level / self.tick)

    # -- S7, restated ------------------------------------------------------
    def _rejection(self):
        o, h, l, c, p = self.o, self.h, self.l, self.c, self.p
        body = np.abs(c - o)
        rng = h - l
        clv = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng > 0, rng, 1), 0.0)
        lower, upper = np.minimum(o, c) - l, h - np.maximum(o, c)
        ratio = float(p.get("rejection.wick_body_ratio"))
        t = float(p.get("rejection.clv_threshold"))
        big = body >= float(p.get("rejection.min_body_atr_multiple")) * self.atr
        out = np.full(self.n, None, dtype=object)
        out[(clv >= t) & (lower > 0) & (lower >= ratio * body) & big] = "long"
        out[(clv <= -t) & (upper > 0) & (upper >= ratio * body) & big] = "short"
        return out

    # -- S4 breakout transitions, restated ---------------------------------
    def breakouts(self, level):
        c, b = self.c, self.buf
        up_state, dn_state = c > level + b, c < level - b
        up = np.zeros(self.n, bool); dn = np.zeros(self.n, bool)
        up[1:] = up_state[1:] & ~up_state[:-1]
        dn[1:] = dn_state[1:] & ~dn_state[:-1]
        return up, dn

    # -- when each S1-S4 level is marked, restated ---------------------------
    def marked(self) -> dict[int, np.ndarray]:
        e, ctx, n = self.e, self.ctx, self.n
        out: dict[int, np.ndarray] = {}

        # Keep each level's ACTUAL source price. Rebuilding it as key * tick
        # puts float noise on non-binary ticks (0.1 * 1201 = 120.10000000000001),
        # which flips strict comparisons when a close lands exactly on a level.
        self.price: dict[int, float] = {}
        self.price_conflicts = 0

        def mark(level, mask):
            k = self.key(level)
            out[k] = out.get(k, np.zeros(n, bool)) | mask
            if k not in self.price:
                self.price[k] = float(level)
            elif abs(self.price[k] - float(level)) > 1e-9:
                self.price_conflicts += 1

        for col in (levels.PRIOR_DAY_HIGH, levels.PRIOR_DAY_LOW,
                    levels.PRIOR_WEEK_HIGH, levels.PRIOR_WEEK_LOW):
            v = e[col].to_numpy(dtype=float)
            for lv in np.unique(v[np.isfinite(v)]):
                mark(lv, np.isclose(v, lv))
        g = ctx.gaps
        if not g.empty:
            ts = e["ts"]
            days = pd.to_datetime(pd.Series(list(e["trade_date"])))
            for z in g.itertuples():
                if pd.isna(z.active_from):
                    continue
                live = (ts >= z.active_from).to_numpy().copy()
                if pd.notna(z.filled_date):
                    live &= (days <= pd.Timestamp(z.filled_date)).to_numpy()
                mark(z.zone_low, live); mark(z.zone_high, live)
        piv = ctx.pivots
        major = piv[piv["is_major"].fillna(False).astype(bool)]
        c = self.c
        for r in major.itertuples():
            lv, pv, conf = float(r.price), int(r.idx), int(r.confirmed_idx)
            after = np.arange(pv + 1, n)
            brk = (c[after] > lv + self.buf[after]) if r.kind == "high" \
                else (c[after] < lv - self.buf[after])
            dead = int(after[np.argmax(brk)]) if brk.any() else n
            mask = np.zeros(n, bool)
            mask[conf:dead] = True                     # live: confirmed .. dead-1
            mark(lv, mask)
        return out


def _levels_price(marked_keys, tick):
    return {k: k * tick for k in marked_keys}


def expected_s8_s9(R: Rules, marked, with_retest: bool):
    s8, s9 = set(), set()
    for k, live in marked.items():
        lv = R.price[k]
        up, dn = R.breakouts(lv)
        for b in np.flatnonzero(up | dn):
            if b < 1 or not live[b - 1]:
                continue
            is_up = bool(up[b])
            back = (lambda x: x < lv) if is_up else (lambda x: x > lv)
            for j in range(b + 1, min(b + 1 + R.k_fail, R.n)):
                if back(R.c[j]):
                    s8.add((int(j), "short" if is_up else "long", k)); break
            if with_retest:
                want = "long" if is_up else "short"
                for j in range(b + 1, min(b + 1 + R.k_retest, R.n)):
                    if back(R.c[j]):
                        break
                    touch = R.l[j] <= lv <= R.h[j]
                    near = min(abs(R.h[j] - lv), abs(R.l[j] - lv)) <= R.zone[j]
                    if (touch or near) and R.rej[j] == want:
                        s9.add((int(j), want, k)); break
    return s8, s9


def expected_s10(R: Rules):
    out = set()
    for side, col in (("high", levels.RANGE_HIGH), ("low", levels.RANGE_LOW)):
        edge = R.e[col].to_numpy(dtype=float)
        for b in range(2, R.n):
            lv = edge[b - 1]
            if not np.isfinite(lv):
                continue
            prev, cur = R.c[b - 1], R.c[b]
            if side == "high":
                if not (cur > lv + R.buf[b] and not prev > lv + R.buf[b - 1]):
                    continue
            elif not (cur < lv - R.buf[b] and not prev < lv - R.buf[b - 1]):
                continue
            for j in range(b + 1, min(b + 1 + R.k_range, R.n)):
                if (R.c[j] < lv) if side == "high" else (R.c[j] > lv):
                    out.add((j, "short" if side == "high" else "long", R.key(lv))); break
    return out


def expected_s17(R: Rules, marked):
    out = set()
    for k, live in marked.items():
        lv = R.price[k]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], live.astype(int), [0]])))
        for a, b_end in zip(edges[::2], edges[1::2] - 1):
            w = min(b_end + 1 + R.span, R.n - 1)
            pending: dict[str, int] = {}
            for i in range(a, w + 1):
                for d in ("long", "short"):
                    if d not in pending or i == pending[d]:
                        continue
                    pv = pending[d]
                    ext = R.h[pv] if d == "long" else R.l[pv]
                    if (R.c[i] > ext) if d == "long" else (R.c[i] < ext):
                        if a <= pv - 1 <= b_end:
                            out.add((i, d, k))
                        del pending[d]
                    elif (R.c[i] < lv) if d == "long" else (R.c[i] > lv):
                        del pending[d]
                    elif i - pv >= R.k_conf:
                        del pending[d]
                if i == a:
                    continue                      # no prior bar inside the scan
                if "long" not in pending and R.h[i] > lv and R.c[i - 1] <= lv:
                    pending["long"] = i
                if "short" not in pending and R.l[i] < lv and R.c[i - 1] >= lv:
                    pending["short"] = i
    return out


def forward(R: Rules, marked, cands):
    """Per-event rule checks; returns {kind: {rule: failures}}."""
    fails = {k: {} for k in KINDS}

    def f(kind, rule):
        fails[kind][rule] = fails[kind].get(rule, 0) + 1

    for cd in cands:
        k, lv, i = R.key(cd.level), cd.level, cd.idx
        start = cd.pattern_bars[0]
        long_ = cd.direction == "long"
        if cd.kind in ("failed_breakout", "breakout_retest"):
            b = start
            up, dn = R.breakouts(lv)
            if not (b >= 1 and k in marked and marked[k][b - 1]):
                f(cd.kind, "level marked on bar before the breakout")
            brk_up = cd.direction == ("short" if cd.kind == "failed_breakout" else "long")
            if not (up[b] if brk_up else dn[b]):
                f(cd.kind, "breakout is a buffered transition")
            if cd.kind == "failed_breakout":
                if not (i - b <= R.k_fail and ((R.c[i] < lv) if brk_up else (R.c[i] > lv))):
                    f(cd.kind, "closes back through within K")
                if any((R.c[j] < lv) if brk_up else (R.c[j] > lv) for j in range(b + 1, i)):
                    f(cd.kind, "first close back")
            else:
                if i - b > R.k_retest or any((R.c[j] < lv) if brk_up else (R.c[j] > lv)
                                             for j in range(b + 1, i + 1)):
                    f(cd.kind, "no failure before the retest, within K_max")
                touch = R.l[i] <= lv <= R.h[i]
                if not (touch or min(abs(R.h[i] - lv), abs(R.l[i] - lv)) <= R.zone[i]):
                    f(cd.kind, "retest bar in the test zone")
                if R.rej[i] != cd.direction:
                    f(cd.kind, "retest bar is an S7 rejection in the breakout direction")
        elif cd.kind == "range_reclaim":
            b = start
            side = "high" if not long_ else "low"
            col = levels.RANGE_HIGH if side == "high" else levels.RANGE_LOW
            edge = R.e[col].iloc[b - 1]
            if not (pd.notna(edge) and R.key(edge) == k):
                f(cd.kind, "edge is the range edge on the bar before the escape")
            up, dn = R.breakouts(lv)
            if not (up[b] if side == "high" else dn[b]):
                f(cd.kind, "escape on the boundary's own side, buffered transition")
            if not (i - b <= R.k_range and ((R.c[i] < lv) if side == "high" else (R.c[i] > lv))):
                f(cd.kind, "closes back inside within K")
        elif cd.kind == "confirmation_signal":
            pv = start
            if not (pv >= 1 and k in marked and marked[k][pv - 1]):
                f(cd.kind, "level marked on bar before the pierce")
            cross = (R.h[pv] > lv and R.c[pv - 1] <= lv) if long_ \
                else (R.l[pv] < lv and R.c[pv - 1] >= lv)
            if not cross:
                f(cd.kind, "pierce is a crossing")
            ext = R.h[pv] if long_ else R.l[pv]
            if not ((R.c[i] > ext) if long_ else (R.c[i] < ext)) or i - pv > R.k_conf:
                f(cd.kind, "closes beyond the pierce extreme within K")
            if any((R.c[j] < lv) if long_ else (R.c[j] > lv) for j in range(pv + 1, i)):
                f(cd.kind, "no close back through the level first")
    return {k: v for k, v in fails.items()}


def run_mode(symbol, series, scfg, mode):
    p = load_params(symbol)
    values = copy.deepcopy(p.values)
    values["breakout"]["mode"] = mode
    params = Params(values=values, symbol=symbol)
    tfs = timeframes.build(symbol, series.bars, scfg, params)
    ctx = gates.GateContext.build(tfs, scfg)
    saved = engine.minor_level_intervals
    engine.minor_level_intervals = lambda k: []          # momentum validated separately
    try:
        cands = [c for c in engine.level_dependent_candidates(ctx) if c.kind in KINDS]
    finally:
        engine.minor_level_intervals = saved
    return ctx, cands


def validate(symbol: str) -> dict:
    series, scfg, _ = build_continuous(symbol, load_data_config())
    report = {}
    for mode in ("buffer", "confirmation_signal"):
        ctx, cands = run_mode(symbol, series, scfg, mode)
        R = Rules(ctx)
        marked = R.marked()
        s8, s9 = expected_s8_s9(R, marked, with_retest=(mode == "buffer"))
        exp = {"failed_breakout": s8}
        if mode == "buffer":
            exp["breakout_retest"] = s9
            exp["range_reclaim"] = expected_s10(R)
        else:
            exp["confirmation_signal"] = expected_s17(R, marked)
        fw = forward(R, marked, cands)
        report.setdefault("_price_conflicts", 0)
        report["_price_conflicts"] += R.price_conflicts
        for kind, want in exp.items():
            if kind == "failed_breakout" and mode == "confirmation_signal":
                continue                                  # already covered in buffer mode
            got = {(c.idx, c.direction, R.key(c.level)) for c in cands if c.kind == kind}
            report[kind] = dict(engine=len(got), expected=len(want),
                                matched=len(got & want),
                                missing=sorted(want - got), extra=sorted(got - want),
                                forward=fw.get(kind, {}))
    return report


def main(argv):
    bad = 0
    for sym in argv:
        rep = validate(sym)
        if rep["_price_conflicts"]:
            bad += 1
            print(f"{sym}: {rep['_price_conflicts']} level(s) where two sources on one "
                  "tick disagree about the price -- investigate before trusting")
        for kind in KINDS:
            r = rep[kind]
            ok = not r["missing"] and not r["extra"] and not any(r["forward"].values())
            bad += not ok
            print(f"{sym} {kind:20}: engine {r['engine']:5}, expected {r['expected']:5}, "
                  f"matched {r['matched']:5}, missing {len(r['missing']):3}, "
                  f"extra {len(r['extra']):3} -> {'OK' if ok else 'MISMATCH'}", flush=True)
            if r["forward"]:
                print(f"     forward rule failures: {r['forward']}")
            for label in ("missing", "extra"):
                for row in r[label][:3]:
                    print(f"     {label}: bar {row[0]} {row[1]} level-tick {row[2]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
