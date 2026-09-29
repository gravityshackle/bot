"""Per-trigger expectancy, pooled across instruments: momentum vs the reversal types.

Usage (from the repo root):  python scripts/edge_by_trigger.py

Reads cache/setup_study/ only (each result re-verified as in a4_report.py) and
writes cache/analysis/edge_by_trigger.json. Population: filled trades from
sized setups. Layers: signal R (exact, before slippage), r_gross, r_net.

Groups. The engine has two families. Continuation triggers are HTF-gated:
momentum, plus breakout_retest, which is tiny. Reversal-type triggers are
never HTF-gated: three_tail, rejection, engulfing, failed_breakout,
range_reclaim. breakout_retest is reported on its own, and the momentum
comparison is made both with and without it on the reversal side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import a4, edge  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import a4_report as rep  # noqa: E402

REVERSAL = ("three_tail", "rejection", "engulfing", "failed_breakout", "range_reclaim")
KEEP = ["period", "kind", "r_signal", "r_gross", "r_net", "filled"]


def main() -> int:
    syms = rep.rs.default_symbols()
    prepared = {}
    for s in syms:
        df, _ = rep._load(s)
        cfg = a4.A4Config.load(s)
        p = a4.prepare(df, cfg, slippage=rep.slippage_by_kind(s, df["exit_kind"].dropna().unique()))
        prepared[s] = p[KEEP].copy()
        del df, p
    cfg = a4.A4Config.load(syms[0])
    f = edge.pooled(prepared)
    f["family"] = np.where(f["kind"] == "momentum", "momentum",
                           np.where(f["kind"] == "breakout_retest", "breakout_retest", "reversal (5)"))
    rng = np.random.default_rng(cfg.seed)
    by_kind = edge.group_table(f, "kind", cfg, rng)
    by_family = edge.group_table(f, "family", cfg, rng)
    mom, rev5 = f["kind"] == "momentum", f["kind"].isin(REVERSAL)
    rev6 = rev5 | (f["kind"] == "breakout_retest")
    gaps = {f"{name} / {layer}": edge.gap(f, mom, other, layer, cfg, rng)
            for name, other in (("momentum - reversal(5)", rev5), ("momentum - reversal(5)+breakout_retest", rev6))
            for layer in a4.LAYERS}
    no_sil = f["symbol"] != "SIL"
    gaps["momentum - reversal(5) / r_signal, SIL excluded"] = edge.gap(
        f[no_sil], mom[no_sil], rev5[no_sil], "r_signal", cfg, rng)
    per_sym = edge.per_instrument_gap(f, mom, rev5, "r_signal")

    pd.set_option("display.width", 200)
    fmt = lambda t, by: t.assign(**{c: t[c].map(lambda v: f"{v:+.3f}") for c in  # noqa: E731
                                     ("signal", "signal_lo", "signal_hi", "gross", "net", "balanced_signal")},
                                 share=t["share"].map(lambda v: f"{100 * v:.1f}%"),
                                 win_signal=t["win_signal"].map(lambda v: f"{100 * v:.0f}%"))
    print(f"filled trades pooled: {len(f):,} across {len(syms)} instruments\n")
    print("BY FAMILY\n" + fmt(by_family, "family").to_string(index=False))
    print("\nBY TRIGGER\n" + fmt(by_kind, "kind").to_string(index=False))
    print("\nGAPS (mean a - mean b, 95% bootstrap interval)")
    for k, g in gaps.items():
        print(f"  {k:58} " + "   ".join(f"{p}: {g[p]['slope']:+.3f} [{g[p]['lo']:+.3f}, {g[p]['hi']:+.3f}] {g[p]['direction']}"
                                         for p in (a4.IS, a4.OOS)))
    print("\nPER INSTRUMENT: momentum - reversal(5), signal R")
    print(per_sym.pivot(index="symbol", columns="period", values="gap").round(3).to_string())
    print("\nPER INSTRUMENT: mean signal R")
    pi = f.assign(fam=np.where(mom, "momentum", np.where(rev5, "reversal(5)", "other"))).query("fam != 'other'")
    print(pi.pivot_table(index="symbol", columns=["period", "fam"], values="r_signal", aggfunc="mean").round(3).to_string())

    out = {"population": "filled trades from sized setups", "n": int(len(f)),
           "by_family": by_family.to_dict("records"), "by_kind": by_kind.to_dict("records"),
           "gaps": gaps, "per_instrument_gap": per_sym.to_dict("records")}
    path = Path("cache") / "analysis" / "edge_by_trigger.json"
    path.write_text(json.dumps(rep._json(out), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
