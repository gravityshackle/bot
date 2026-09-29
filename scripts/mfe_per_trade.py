"""MFE and the plain-bracket counterfactual for one instrument's filled trades (cache only).

Usage (from the repo root; one process per instrument, nothing else heavy running):
    python scripts/mfe_per_trade.py SYM

Reads the verified study result (cache/setup_study/SYM.pkl) and the cached
1-minute series the study ran on, and writes cache/analysis/mfe/SYM.pkl: one
row per filled trade, keyed by the study row's index.

Built-in equivalence check, which must pass or nothing is written. For every
trade whose stop never moved (stop_moves == 0), the lifecycle's exit IS its
first scan, so the bracket must reproduce the study's actual exit kind, exit
time, exit fill and r_gross exactly.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import a4_report as rep  # noqa: E402
from analysis import mfe  # noqa: E402
from backtest.setup_study import StudyConfig  # noqa: E402
from data.pipeline import build_continuous, load_data_config, load_symbol_config  # noqa: E402
from features.schema import load_params  # noqa: E402

OUT = Path("cache") / "analysis" / "mfe"


def _y(p):
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(sym: str) -> int:
    t0 = time.time()
    df, man = rep._load(sym)                     # refuses a stale or damaged result
    f = df[(df["contracts"] > 0) & (df["entry_status"] == "filled")]
    del df
    sc = StudyConfig.from_configs(costs_cfg=_y("config/costs.yaml"), exec_cfg=_y("config/execution.yaml"),
                                  risk_cfg=_y("config/risk.yaml"), symbol_cfg=load_symbol_config(sym),
                                  timeframe=str(load_params(sym).get("timeframes.entry")))
    series, _, _raw = build_continuous(sym, load_data_config())
    del _raw
    bars = series.bars.reset_index(drop=True)
    t_load = time.time() - t0
    res = mfe.per_trade(bars, f, cost=sc.cost, cfg=sc.cfg)

    fixed = f["stop_moves"] == 0
    j = res.loc[fixed[fixed].index]
    a = f.loc[fixed]
    same = ((j["bracket_exit"] == a["exit_kind"]) & (j["bracket_exit_ts"] == a["exit_ts"])
            & np.isclose(j["bracket_exit_fill"], a["exit_fill"], rtol=0, atol=1e-9)
            & np.isclose(j["bracket_r_gross"], a["r_gross"], rtol=0, atol=1e-12))
    bad = int((~same).sum())
    info = {"symbol": sym, "study_commit": man["commit"], "study_data_sha256": man["data_sha256"],
            "filled": int(len(f)), "fixed_stop_trades": int(fixed.sum()), "bracket_mismatches": bad,
            "bracket_unresolved": int((res["bracket_exit"] == "unresolved").sum()),
            "load_seconds": round(t_load, 1), "seconds": round(time.time() - t0, 1)}
    print(f"{sym}: {len(f):,} filled; equivalence on {fixed.sum():,} fixed-stop trades: "
          f"{'EXACT' if bad == 0 else f'{bad} MISMATCHES'}; brackets unresolved at data end: "
          f"{info['bracket_unresolved']}; load {t_load:.0f}s, total {info['seconds']:.0f}s", flush=True)
    if bad:
        print(j.loc[~same].head().to_string(), "\n", a.loc[~same[~same].index,
              ["exit_kind", "exit_ts", "exit_fill", "r_gross"]].head().to_string())
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".{sym}.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(OUT / f"{sym}.pkl")
    (OUT / f"{sym}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/mfe_per_trade.py SYM")
    sys.exit(main(sys.argv[1]))
