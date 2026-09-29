"""Phase 4.4: the A4 answer, per instrument, in-sample and out-of-sample (reads results only).

Usage (from the repo root):
    python scripts/a4_report.py [SYM ...]

Needs cache/setup_study/ (scripts/run_setup_study.py) and
cache/analysis/prior_atr/ (scripts/a4_prior_atr.py, one instrument at a time).
Each study result is re-verified first: its stamp must match the current tree
and its data must match the sha256 in its manifest. A stale or damaged result
is refused, not analysed.

The verdicts use r_gross (after slippage, before fees). Every band also shows
signal R, r_net, dollars and fees as a share of risk, and each instrument gets
a separate cost-drag finding. Writes cache/analysis/a4/SYM.json (every table and
answer) and prints the tables plus a verdict summary. Instruments are NOT
pooled here: whether pooling is valid is decided from these results.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import a4  # noqa: E402
from backtest.setup_study import StudyConfig  # noqa: E402
from data.pipeline import load_symbol_config  # noqa: E402
from features.schema import load_params  # noqa: E402

STUDY = Path("cache") / "setup_study"
PRIOR = Path("cache") / "analysis" / "prior_atr"
OUT = Path("cache") / "analysis" / "a4"

_spec = importlib.util.spec_from_file_location("run_setup_study", ROOT / "scripts" / "run_setup_study.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def _load(sym: str) -> tuple[pd.DataFrame, dict]:
    why = rs.status(STUDY, sym, rs.stamp(sym, None))
    if why != "done":
        raise SystemExit(f"{sym}: study result is {why}; re-run scripts/run_setup_study.py first")
    man = json.loads((STUDY / f"{sym}.json").read_text(encoding="utf-8"))
    return pd.read_pickle(STUDY / f"{sym}.pkl"), man


def slippage_by_kind(sym: str, kinds) -> dict:
    """The study's own cost model's slippage per exit kind, in price points."""
    import yaml

    def y(p):
        with open(p, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    sc = StudyConfig.from_configs(costs_cfg=y("config/costs.yaml"), exec_cfg=y("config/execution.yaml"),
                                  risk_cfg=y("config/risk.yaml"), symbol_cfg=load_symbol_config(sym),
                                  timeframe=str(load_params(sym).get("timeframes.entry")))
    return {k: sc.cost.slippage(k) for k in kinds}


def _json(v):
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return v


def _pct(x):
    return "  -  " if x != x else f"{100 * x:4.0f}%"


def _num(x, w=6):
    return " " * (w - 1) + "-" if x != x else f"{x:{w}.2f}"


def _fmt_table(t: pd.DataFrame, by: str) -> str:
    lines = [f"  {'':3} {by:9} {'sized':>6} {'filled':>6} {'fill':>5} {'win':>5} {'scr':>5} "
             f"{'R gross':>7} {'95% CI':>15} {'med':>6} {'p10':>6} {'p90':>6} {'top5%':>5} |"
             f" {'signal':>6} {'net':>6} {'$ net':>6} {'fee/rsk':>7} {'fee>=R':>6}"]
    for r in t.itertuples(index=False):
        d = r._asdict()
        ci = "" if d["exp_lo"] != d["exp_lo"] else f"[{d['exp_lo']:+.2f},{d['exp_hi']:+.2f}]"
        lines.append(f"  {d['period']:3} {str(d[by]):9} {d['sized']:6,} {d['filled']:6,} {_pct(d['fill_rate'])} "
                     f"{_pct(d['win_rate'])} {_pct(d['scratch_rate'])} {_num(d['exp_r'], 7)} {ci:>15} "
                     f"{_num(d['median_r'])} {_num(d['p10_r'])} {_num(d['p90_r'])} {_pct(d['tail_share'])} |"
                     f" {_num(d['exp_signal'])} {_num(d['exp_net'])} {_num(d['usd_net'])} "
                     f"{_pct(d['fee_over_risk_median']):>7} {_pct(d['fees_ge_risk']):>6}"
                     f"{'  THIN' if d['thin'] else ''}")
    return "\n".join(lines)


def _fmt_trend(name: str, t: dict) -> str:
    if t["direction"] == "insufficient":
        return f"    {name:14} insufficient (n={t['n']:,})"
    extra = f", covers {100 * t['oos_coverage']:.0f}% of OOS" if "oos_coverage" in t else ""
    return (f"    {name:14} {t['slope']:+.3f} R [{t['lo']:+.3f}, {t['hi']:+.3f}] -> {t['direction']:5} "
            f"(n={t['n']:,}{extra})")


def _attribution(a: dict) -> str:
    at = a["attribution"]
    parts = []
    for layer in ("slippage", "fees"):
        where = [p for p in (a4.IS, a4.OOS) if at[layer][p]]
        parts.append(f"{layer}: {'changes the direction in ' + '+'.join(where) if where else 'no effect on direction'}")
    parts.append("sizing: " + {None: "not checked (reported directly)",
                               True: "CHANGES the in-sample direction",
                               False: "no effect on direction"}[at["sizing"]])
    return "    attribution    " + "; ".join(parts)


def _cost_finding(p: pd.DataFrame, cfg: a4.A4Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    out = {}
    for period in (a4.IS, a4.OOS):
        s = a4.cell_stats(p[p["period"] == period], cfg, rng)
        out[period] = {"filled": s["filled"], "slippage_r": s["exp_signal"] - s["exp_r"],
                       "fees_r": s["exp_r"] - s["exp_net"], "fee_over_risk_median": s["fee_over_risk_median"],
                       "fees_ge_risk": s["fees_ge_risk"], "exp_signal": s["exp_signal"], "exp_gross": s["exp_r"],
                       "exp_net": s["exp_net"], "usd_net": s["usd_net"]}
    return out


def report(sym: str) -> dict:
    df, man = _load(sym)
    cfg = a4.A4Config.load(sym)
    prior = pickle.load(open(PRIOR / f"{sym}.pkl", "rb"))
    p = a4.prepare(df, cfg, slippage=slippage_by_kind(sym, df["exit_kind"].dropna().unique()))
    p["stop_atr"] = a4.stop_width_atr(p, prior, load_symbol_config(sym)).to_numpy()
    rr_t, sc_t = a4.band_table(p, "rr_band", cfg), a4.band_table(p, "score_q", cfg)
    ans = {"component": a4.rr_component_answer(p, cfg, sym), "cap": a4.cap_answer(p, cfg, sym),
           "score": a4.score_answer(p, cfg, sym)}
    shift, cost = a4.survivor_shift(p), _cost_finding(p, cfg)
    flag = " [LOW-SAMPLE / DIRECTIONAL ONLY: never answers A4 on its own]" if sym in cfg.directional_only_symbols else ""
    stricter = " [sizing-shift: stricter check applies]" if sym in cfg.sizing_shift_symbols else ""
    print(f"\n{'=' * 130}\n{sym}{flag}{stricter}\n"
          f"  study {man['commit'][:7]}, {len(df):,} Stage 1 setups; split {cfg.split.date()}; verdicts on {cfg.outcome}; "
          f"stop_atr missing on {int(p['stop_atr'].isna().sum()):,} (sessions before ATR exists)")
    print(f"\n  RR band (live below the cap {cfg.rr_cap:g}, saturated from it)\n" + _fmt_table(rr_t, "rr_band"))
    print("\n  Score quintile (edges from in-sample sized setups)\n" + _fmt_table(sc_t, "score_q"))
    for k, title in (("component", "reward/risk component: R per live RR band"),
                     ("cap", "cap: saturated bands minus [3, 4)"),
                     ("score", "score: R per quintile")):
        a = ans[k]
        print(f"\n  {title}  =>  {a['verdict'].upper()}")
        for part in ("IS", "OOS", "IS_restricted", "IS_stratified"):
            if part in a:
                print(_fmt_trend(part, a[part]))
        for period in (a4.IS, a4.OOS):
            L = a["layers"][period]
            print(f"    {period} layers      signal {L['r_signal']['direction']:12} gross {L['r_gross']['direction']:12} "
                  f"net {L['r_net']['direction']}")
        print(_attribution(a))
    print("\n  COST DRAG (its own finding; not part of the verdict), mean per filled trade")
    for period, c in cost.items():
        print(f"    {period:3} signal {c['exp_signal']:+.2f} R -> gross {c['exp_gross']:+.2f} R (slippage {c['slippage_r']:.2f} R)"
              f" -> net {c['exp_net']:+.2f} R (fees {c['fees_r']:.2f} R) = ${c['usd_net']:+.2f}; "
              f"fees are {100 * c['fee_over_risk_median']:.0f}% of risk at the median, >= the whole risk on "
              f"{100 * c['fees_ge_risk']:.0f}% of trades")
    print("\n  filled trades, in-sample vs out-of-sample (p10 / p50 / p90, KS D, p)")
    for col, s in shift.items():
        print(f"    {col:9} IS {s['IS_p10_50_90'][0]:8.2f} {s['IS_p10_50_90'][1]:8.2f} {s['IS_p10_50_90'][2]:8.2f}"
              f" | OOS {s['OOS_p10_50_90'][0]:8.2f} {s['OOS_p10_50_90'][1]:8.2f} {s['OOS_p10_50_90'][2]:8.2f}"
              f" | D {s['D']:.3f}  p {s['p']:.2g}")
    cfg_sha = hashlib.sha256((ROOT / a4.CONFIG_PATH).read_bytes()).hexdigest()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"symbol": sym, "study_commit": man["commit"], "study_data_sha256": man["data_sha256"],
           "a4_config_sha256": cfg_sha, "analysis_commit": head, "outcome": cfg.outcome,
           "directional_only": sym in cfg.directional_only_symbols,
           "sizing_shift": sym in cfg.sizing_shift_symbols,
           "rr_bands": rr_t.astype({"rr_band": str}).to_dict("records"), "score_quintiles": sc_t.to_dict("records"),
           "answers": ans, "cost_drag": cost, "survivor_shift": shift}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{sym}.json").write_text(json.dumps(_json(out), indent=1), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    syms = argv or rs.default_symbols()
    missing = [s for s in syms if not (PRIOR / f"{s}.pkl").exists()]
    if missing:
        raise SystemExit(f"no prior ATR for {missing}; run scripts/a4_prior_atr.py SYM for each")
    res = {}
    for s in syms:
        res[s] = report(s)
    print(f"\n{'=' * 130}\nVERDICTS on r_gross (per instrument; not pooled)")
    print(f"  {'':4} {'component':22} {'cap':22} {'score':22}")
    for s, r in res.items():
        tag = " (directional only)" if r["directional_only"] else ""
        print(f"  {s:4} " + " ".join(f"{r['answers'][k]['verdict']:22}" for k in ("component", "cap", "score")) + tag)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
