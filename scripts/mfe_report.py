"""Do trades reach their targets, and does breakeven/trailing help or hurt? Per trigger.

Usage (from the repo root):  python scripts/mfe_report.py

Needs cache/setup_study/ (verified as in a4_report.py) and cache/analysis/mfe/
(scripts/mfe_per_trade.py, one instrument at a time). Pooled across
instruments by trigger and trigger family, in-sample and out-of-sample, with
per-instrument signs of the state machine's effect as a robustness check.
Writes cache/analysis/mfe_by_trigger.json.

effect = actual trade R minus the plain bracket's R (the original stop and
target held to the day boundary: no breakeven, no trailing), per trade.
Negative means the exit state machine costs R.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import a4_report as rep  # noqa: E402
from analysis import a4, mfe  # noqa: E402

MFE = Path("cache") / "analysis" / "mfe"
REVERSAL = ("three_tail", "rejection", "engulfing", "failed_breakout", "range_reclaim")
STUDY_COLS = ["period", "kind", "exit_kind", "path", "r_signal", "r_gross", "r_net", "target_source", "stop_atr"]
PRIOR = Path("cache") / "analysis" / "prior_atr"


def main() -> int:
    parts = []
    for s in rep.rs.default_symbols():
        df, man = rep._load(s)
        info = json.loads((MFE / f"{s}.json").read_text(encoding="utf-8"))
        if info["study_data_sha256"] != man["data_sha256"] or info["bracket_mismatches"] != 0:
            raise SystemExit(f"{s}: MFE results do not belong to the verified study result, or failed equivalence")
        p = a4.prepare(df, a4.A4Config.load(s), slippage=rep.slippage_by_kind(s, df["exit_kind"].dropna().unique()))
        p["stop_atr"] = a4.stop_width_atr(p, pickle.load(open(PRIOR / f"{s}.pkl", "rb")),
                                          rep.load_symbol_config(s)).to_numpy()
        m = pickle.load(open(MFE / f"{s}.pkl", "rb"))
        f = p.loc[p["filled"], STUDY_COLS]
        if not f.index.equals(m.index.sort_values()) and set(f.index) != set(m.index):
            raise SystemExit(f"{s}: MFE rows do not match the study's filled trades")
        j = f.join(m, how="inner")
        j["symbol"] = s
        parts.append(j)
        del df, p
    f = pd.concat(parts, ignore_index=True)
    f["family"] = np.where(f["kind"] == "momentum", "momentum",
                           np.where(f["kind"] == "breakout_retest", "breakout_retest", "reversal (5)"))
    f["all"] = "all triggers"
    cfg = a4.A4Config.load("MES")
    rng = np.random.default_rng(cfg.seed)
    tables = {by: mfe.summary(f, by, cfg, rng) for by in ("all", "family", "kind")}

    pct = lambda v: f"{100 * v:4.0f}%"                                   # noqa: E731
    print(f"filled trades: {len(f):,}; brackets unresolved at data end: {int((f['bracket_exit'] == 'unresolved').sum())}\n")
    for by, t in tables.items():
        print(f"=== by {by}: reach ===")
        print(f"  {'':3} {by if by != 'all' else '':16} {'n':>6} {'tgtR':>5} {'MFE med':>7} {'p75':>5} {'p90':>5} "
              f"{'MFE/tgt':>7} {'>=1R':>5} {'>=2R':>5} {'>=tgt':>5} | {'tgt filled':>10} {'bracket tgt':>11}")
        for r in t.itertuples(index=False):
            d = r._asdict()
            print(f"  {d['period']:3} {str(d[by]) if by != 'all' else '':16} {d['n']:6,} {d['target_r_med']:5.2f} "
                  f"{d['mfe_med']:7.2f} {d['mfe_p75']:5.2f} {d['mfe_p90']:5.2f} {d['mfe_over_target_med']:7.2f} "
                  f"{pct(d['reached_1r'])} {pct(d['reached_2r'])} {pct(d['reached_target'])} | "
                  f"{pct(d['target_filled']):>10} {pct(d['bracket_target']):>11}")
        print(f"=== by {by}: R, actual vs plain bracket (effect = actual - bracket, 95% CI) ===")
        for r in t.itertuples(index=False):
            d = r._asdict()
            eff = "  ".join(f"{lay[2:]:6} {d[f'actual_{lay}']:+.3f} vs {d[f'bracket_{lay}']:+.3f} = "
                            f"{d[f'effect_{lay}']:+.3f} [{d[f'effect_{lay}_lo']:+.3f},{d[f'effect_{lay}_hi']:+.3f}]"
                            for lay in ("r_signal", "r_net"))
            print(f"  {d['period']:3} {str(d[by]) if by != 'all' else '':16} {eff}")
        print()

    # the target in R is (level distance) / (stop width): both in prior-session daily ATR
    f["target_atr"] = f["stop_atr"] * f["target_r"]
    f["mfe_atr"] = f["stop_atr"] * f["mfe_r"]
    sc_ = f.dropna(subset=["stop_atr"])
    scale = sc_.groupby("kind").agg(n=("target_r", "size"),
                                    fallback_2r=("target_source", lambda x: float((x == "2R_fallback").mean())),
                                    target_r=("target_r", "median"), stop_atr=("stop_atr", "median"),
                                    target_atr=("target_atr", "median"), mfe_atr=("mfe_atr", "median"),
                                    mfe_r=("mfe_r", "median"))
    scale.loc["ALL"] = [len(sc_), float((sc_["target_source"] == "2R_fallback").mean()), sc_["target_r"].median(),
                        sc_["stop_atr"].median(), sc_["target_atr"].median(), sc_["mfe_atr"].median(),
                        sc_["mfe_r"].median()]
    print("scale: medians in R and in prior-session daily ATR (target R = target_atr / stop_atr)\n"
          + scale.round(3).to_string() + "\n")

    ok = f[f["bracket_exit"] != "unresolved"]
    e = (ok.assign(effect=mfe.paired(ok, "r_signal", "bracket_r_signal"))
         .pivot_table(index="symbol", columns="period", values="effect", aggfunc="mean"))
    print("per instrument: state-machine effect on signal R (actual - bracket)\n" + e.round(3).to_string())
    b = ok.pivot_table(index="symbol", columns="period", values="bracket_r_signal", aggfunc="mean")
    print("\nper instrument: plain-bracket signal R\n" + b.round(3).to_string())
    out = {"n": int(len(f)), "tables": {k: v.to_dict("records") for k, v in tables.items()},
           "scale": scale.reset_index().to_dict("records"),
           "per_instrument_effect_signal": e.round(6).reset_index().to_dict("records"),
           "per_instrument_bracket_signal": b.round(6).reset_index().to_dict("records")}
    (Path("cache") / "analysis" / "mfe_by_trigger.json").write_text(json.dumps(rep._json(out), indent=1),
                                                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
