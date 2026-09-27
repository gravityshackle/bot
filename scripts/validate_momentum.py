"""Independent re-validation of the engine's S11 momentum events on real data.

scripts/plot_triggers.py checks momentum's per-bar arithmetic, but on its own
older enumeration (the 3 most recent minor swings per session, no liveness),
and its check ignores the HTF trend, the event-vs-state transition and the
side match. So the momentum events the Signal Engine actually produces had
never been re-derived from raw OHLC since those fixes landed. This does it.

Written in plain loops over raw bars and pivots, without calling the engine's
enumeration, liveness or momentum code, so agreement means something.

FORWARD -- every engine momentum event must satisfy all six rules:
  1 level is a confirmed MINOR swing, side matching direction (high->long, low->short)
  2 that swing is live on the bar before: no close through it since it formed
  3 transition: previous close not beyond the level, this close beyond it
  4 body ratio >= momentum.min_body_ratio (raw OHLC)
  5 volume expanded on the bar (S12)
  6 HTF bias matches the direction (S14, causally aligned)

REVERSE -- for every minor swing, its FIRST close through the level after it
confirmed is a momentum event if that bar satisfies rules 4-6 and the swing
confirmed before that bar. The engine must have produced exactly those.

Usage: python scripts/validate_momentum.py SYM [SYM ...]   (one process each is kindest)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data.pipeline import build_continuous, load_data_config  # noqa: E402
from features.schema import VOLUME_EXPANDED, tick_size  # noqa: E402
from signal_engine import engine, gates, timeframes  # noqa: E402


def validate(symbol: str) -> dict:
    series, scfg, _ = build_continuous(symbol, load_data_config())
    tfs = timeframes.build(symbol, series.bars, scfg)
    ctx = gates.GateContext.build(tfs, scfg)
    p = ctx.params
    e = ctx.entry
    o, h, l, c = (e[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    vol_ok = e[VOLUME_EXPANDED].to_numpy(dtype=bool)
    bias = ctx.bias.to_numpy(dtype=object)
    min_ratio = float(p.get("momentum.min_body_ratio"))
    tick = tick_size(p)

    piv = ctx.pivots
    minor = piv[piv["is_major"].notna() & ~piv["is_major"].fillna(True).astype(bool)]

    def body_ok(i: int) -> bool:
        rng = h[i] - l[i]
        return rng > 0 and abs(c[i] - o[i]) / rng >= min_ratio

    def trend_ok(i: int, direction: str) -> bool:
        return bias[i] == ("bullish" if direction == "long" else "bearish")

    def beyond(x: float, level: float, kind: str) -> bool:
        return x > level if kind == "high" else x < level

    # ---- engine output (momentum only) -------------------------------------
    engine.marked_level_intervals = lambda k: []
    engine.range_edge_intervals = lambda k: []
    got = [cd for cd in engine.level_dependent_candidates(ctx) if cd.kind == "momentum"]
    got_keys = {(cd.idx, cd.direction, round(cd.level / tick)) for cd in got}

    # ---- reverse: what the rules say should exist --------------------------
    expected = set()
    for r in minor.itertuples():
        kind, level, pv, conf = r.kind, float(r.price), int(r.idx), int(r.confirmed_idx)
        direction = "long" if kind == "high" else "short"
        first = next((j for j in range(pv + 1, len(e)) if beyond(c[j], level, kind)), None)
        if first is None or first - 1 < conf:
            continue                          # never crossed, or crossed before it was known
        i = first
        if not beyond(c[i - 1], level, kind) and body_ok(i) and vol_ok[i] and trend_ok(i, direction):
            expected.add((i, direction, round(level / tick)))

    # ---- forward: each engine event against the six rules -------------------
    fails = {k: 0 for k in ("1 minor swing, side", "2 live before", "3 transition",
                            "4 body ratio", "5 volume", "6 HTF trend")}
    for cd in got:
        i, level, direction = cd.idx, cd.level, cd.direction
        kind = "high" if direction == "long" else "low"
        same = minor[((minor["price"] - level).abs() < tick / 2) & (minor["kind"] == kind)
                     & (minor["confirmed_idx"] <= i - 1)]
        if same.empty:
            fails["1 minor swing, side"] += 1
        elif not any(not any(beyond(c[j], level, kind) for j in range(int(pv) + 1, i))
                     for pv in same["idx"]):
            fails["2 live before"] += 1
        if not (not beyond(c[i - 1], level, kind) and beyond(c[i], level, kind)):
            fails["3 transition"] += 1
        if not body_ok(i):
            fails["4 body ratio"] += 1
        if not vol_ok[i]:
            fails["5 volume"] += 1
        if not trend_ok(i, direction):
            fails["6 HTF trend"] += 1

    missing = sorted(expected - got_keys)
    extra = sorted(got_keys - expected)
    return dict(symbol=symbol, engine=len(got), expected=len(expected),
                matched=len(expected & got_keys), missing=missing, extra=extra,
                forward_fails=fails)


def main(argv: list[str]) -> int:
    bad = 0
    for sym in argv:
        r = validate(sym)
        ok = not r["missing"] and not r["extra"] and not any(r["forward_fails"].values())
        bad += not ok
        print(f"{sym}: engine {r['engine']}, independently expected {r['expected']}, "
              f"matched {r['matched']}, missing {len(r['missing'])}, extra {len(r['extra'])} "
              f"-> {'OK' if ok else 'MISMATCH'}", flush=True)
        failed = {k: v for k, v in r["forward_fails"].items() if v}
        if failed:
            print(f"   forward rule failures: {failed}")
        for label, rows in (("missing", r["missing"]), ("extra", r["extra"])):
            for idx, d, lv in rows[:5]:
                print(f"   {label}: bar {idx} {d} level {lv}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
