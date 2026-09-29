"""Phase 4.4 follow-up (F1): do trades reach their targets, and does the exit
state machine help or hurt?

Per filled trade, in R (the initial risk: entry fill to the resting stop order):

  excursion  the in-trade maximum favourable excursion (MFE), entry to exit,
             against the target: reached 1R (the breakeven trigger), 2R, and
             the target itself.
  bracket    the counterfactual of holding the ORIGINAL stop and target to the
             day boundary, with no breakeven and no trailing. This is the
             lifecycle's first exit scan (execution.simulated_execution.
             scan_exit), priced by the engine's own round_trip, so it shares
             every fill rule with the study. Its signal, gross and net R set
             against the actual trade's give the state machine's effect in R.

The simulator's rules, applied to the excursion as well:
  - the fill bar's extreme never counts: its high/low may come before the
    fill, and on the fill bar only a stop can exit (fill_bar_exits);
  - reaching the target needs target_through_ticks beyond it, as the target
    limit does; a touch is not a reach;
  - a stop exit's own bar never counts (adverse first); a target or flatten
    bar does.

The entry is the study's recorded fill (a limit at its order price, no
slippage, the configured fees). Nothing about entries is re-simulated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from execution.simulated_execution import ENTRY_LIMIT, Fill, round_trip, scan_exit

STOP_EXIT = "stop_market_exit"


def _sign(direction: str) -> float:
    return 1.0 if direction == "long" else -1.0


def _risk_pts(row) -> float:
    return abs(row["entry_fill"] - row["stop_order"])


def _pos(b: pd.DataFrame, ts: pd.Timestamp, what: str) -> int:
    i = int(b["ts"].searchsorted(ts, side="left"))
    if i >= len(b) or b["ts"].iloc[i] != ts:
        raise ValueError(f"the {what} at {ts} is not on a bar")
    return i


def _mfe_r(b: pd.DataFrame, row, lo: int, hi: int) -> float:
    """Best favourable R over bars [lo, hi), floored at 0 (the entry price itself)."""
    if hi <= lo:
        return 0.0
    s = _sign(row["direction"])
    ext = b["high"].iloc[lo:hi].max() if s > 0 else b["low"].iloc[lo:hi].min()
    return max(0.0, s * (ext - row["entry_fill"]) / _risk_pts(row))


def _levels(row, mfe_r: float, cost, cfg) -> dict:
    s = _sign(row["direction"])
    risk = _risk_pts(row)
    target_r = s * (row["target_order"] - row["entry_fill"]) / risk
    need = target_r + cfg.target_through_ticks * cost.tick_size / risk
    return {"mfe_r": mfe_r, "target_r": target_r, "reach_needed_r": need,
            "reached_1r": mfe_r >= 1.0, "reached_2r": mfe_r >= 2.0, "reached_target": mfe_r >= need - 1e-12}


def excursion(b: pd.DataFrame, row, *, entry_pos: int, exit_pos: int, exit_kind: str, cost, cfg) -> dict:
    """The in-trade MFE and reach flags. Bars [entry_pos, exit_pos] are the trade's."""
    hi = exit_pos if exit_kind == STOP_EXIT else exit_pos + 1
    return _levels(row, _mfe_r(b, row, entry_pos + 1, hi), cost, cfg)


def bracket(b: pd.DataFrame, row, *, entry_ts: pd.Timestamp, cost, cfg) -> dict:
    """Hold the original stop and target to the day boundary (no breakeven, no trailing)."""
    s = _sign(row["direction"])
    pos = _pos(b, entry_ts, "entry")
    td = b["trade_date"].to_numpy()
    day = np.flatnonzero(td[pos:] != td[pos])
    end = pos + int(day[0]) + 1 if len(day) else len(b)
    w = b.iloc[:end]
    contracts = int(row["contracts"])
    entry = Fill(ENTRY_LIMIT, entry_ts, int(s), contracts, float(row["entry_order"]),
                 float(row["entry_order"]), float(row["entry_fill"]), cost.fees(contracts))
    ex = scan_exit(w, pos, direction=row["direction"], stop_ticks=cost.to_ticks(row["stop_order"]),
                   target_ticks=cost.to_ticks(row["target_order"]), contracts=contracts, cost=cost,
                   cfg=cfg, fill_bar=True)
    if ex.fill is None:
        # still holding when the data ends (the last, partial session): no
        # outcome exists, so none is invented; callers count and exclude these
        nan = float("nan")
        return {"bracket_exit": "unresolved", "bracket_exit_ts": pd.NaT, "bracket_exit_fill": nan,
                "bracket_r_gross": nan, "bracket_r_net": nan, "bracket_r_signal": nan,
                "bracket_mfe_r": nan, "bracket_reached_target": False}
    rt = round_trip(entry, ex.fill, direction=row["direction"], stop=float(row["stop_order"]), cost=cost)
    exit_pos = pos + int(w["ts"].iloc[pos:].searchsorted(ex.fill.ts, side="left"))
    e = excursion(w, row, entry_pos=pos, exit_pos=exit_pos, exit_kind=ex.fill.kind, cost=cost, cfg=cfg)
    return {"bracket_exit": ex.fill.kind, "bracket_exit_ts": ex.fill.ts, "bracket_exit_fill": ex.fill.fill_price,
            "bracket_r_gross": rt["r_gross"], "bracket_r_net": rt["r_net"],
            "bracket_r_signal": rt["r_gross"] + cost.slippage(ex.fill.kind) / _risk_pts(row),
            "bracket_mfe_r": e["mfe_r"], "bracket_reached_target": ex.fill.kind == "target_limit"}


def per_trade(b: pd.DataFrame, filled: pd.DataFrame, *, cost, cfg) -> pd.DataFrame:
    """Excursion and bracket for every filled study row (index kept)."""
    out = {}
    for idx, row in filled.iterrows():
        pos = _pos(b, row["entry_ts"], "entry")
        xpos = _pos(b, row["exit_ts"], "exit")
        rec = excursion(b, row, entry_pos=pos, exit_pos=xpos, exit_kind=row["exit_kind"], cost=cost, cfg=cfg)
        rec.update(bracket(b, row, entry_ts=row["entry_ts"], cost=cost, cfg=cfg))
        out[idx] = rec
    return pd.DataFrame.from_dict(out, orient="index")


def paired(f: pd.DataFrame, actual: str, counterfactual: str) -> pd.Series:
    """Per trade: actual minus counterfactual (the state machine's effect, in R)."""
    return f[actual] - f[counterfactual]


def summary(f: pd.DataFrame, by: str, cfg, rng) -> pd.DataFrame:
    """Per period and group: reach rates, the MFE distribution, and actual vs bracket R.

    `f`: filled trades carrying the study's columns, a4.prepare's `period` and
    `r_signal`, and per_trade's columns. Brackets unresolved at the data's end
    are counted and left out of every bracket figure (and of the paired
    differences), never filled in."""
    from analysis.a4 import IS, OOS, _boot_mean

    rows = []
    for period in (IS, OOS):
        sub = f[f["period"] == period]
        for g, x in sub.groupby(by, sort=True):
            ok = x[x["bracket_exit"] != "unresolved"]
            row = {"period": period, by: g, "n": len(x), "unresolved": len(x) - len(ok),
                   "target_r_med": float(x["target_r"].median()),
                   "mfe_med": float(x["mfe_r"].median()), "mfe_p75": float(x["mfe_r"].quantile(0.75)),
                   "mfe_p90": float(x["mfe_r"].quantile(0.9)),
                   "mfe_over_target_med": float((x["mfe_r"] / x["target_r"]).median()),
                   "reached_1r": float(x["reached_1r"].mean()), "reached_2r": float(x["reached_2r"].mean()),
                   "reached_target": float(x["reached_target"].mean()),
                   "target_filled": float((x["exit_kind"] == "target_limit").mean()),
                   "bracket_target": float(ok["bracket_reached_target"].mean()),
                   "bracket_mfe_med": float(ok["bracket_mfe_r"].median())}
            for layer, bl in (("r_signal", "bracket_r_signal"), ("r_gross", "bracket_r_gross"),
                              ("r_net", "bracket_r_net")):
                d = paired(ok, layer, bl).to_numpy(dtype="float64")
                lo, hi = _boot_mean(d, cfg, rng)
                row.update({f"actual_{layer}": float(ok[layer].mean()), f"bracket_{layer}": float(ok[bl].mean()),
                            f"effect_{layer}": float(d.mean()), f"effect_{layer}_lo": lo,
                            f"effect_{layer}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)
