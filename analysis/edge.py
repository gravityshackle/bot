"""Phase 4.4 follow-up: is the negative expectancy the trigger logic, or one trigger?

Pooled across instruments, which is valid for this question: it asks about
the trigger logic itself, and R puts every instrument on one scale. Per
instrument figures are reported beside the pool as a robustness check, along
with an instrument-balanced mean (the mean of per-instrument means), so no
single instrument's trade count carries the pooled answer.

Population: filled trades from sized setups (discarded and unfilled setups
have no outcome). Layers as in analysis/a4.py: signal R (r_gross with the
modelled slippage added back, gap losses kept), r_gross, r_net.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import a4

LAYERS = a4.LAYERS


def pooled(prepared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Filled trades of every instrument, one frame, with a `symbol` column."""
    parts = []
    for sym, p in prepared.items():
        f = p[p["filled"]].copy()
        f["symbol"] = sym
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def group_table(f: pd.DataFrame, by: str, cfg: a4.A4Config, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for period in (a4.IS, a4.OOS):
        sub = f[f["period"] == period]
        for g, x in sub.groupby(by, sort=True):
            r = x["r_signal"].to_numpy(dtype="float64")
            lo, hi = a4._boot_mean(r, cfg, rng)
            per_sym = x.groupby("symbol")["r_signal"].mean()
            rows.append({"period": period, by: g, "n": len(x), "share": len(x) / len(sub),
                         "signal": float(r.mean()), "signal_lo": lo, "signal_hi": hi,
                         "gross": float(x["r_gross"].mean()), "net": float(x["r_net"].mean()),
                         "win_signal": float((r > 0).mean()),
                         "balanced_signal": float(per_sym.mean()), "instruments": int(per_sym.size),
                         "instruments_positive": int((per_sym > 0).sum()),
                         "thin": len(x) < cfg.min_cell_trades})
    return pd.DataFrame(rows)


def gap(f: pd.DataFrame, a: pd.Series, b: pd.Series, layer: str, cfg: a4.A4Config,
        rng: np.random.Generator) -> dict:
    """Mean `layer` of group a minus group b, per period, with a bootstrap interval
    (a 0/1 trend, so the same machinery and 'flat' rule as A4)."""
    out = {}
    for period in (a4.IS, a4.OOS):
        m = (f["period"] == period).to_numpy()
        x = np.where(a.to_numpy()[m], 1.0, np.where(b.to_numpy()[m], 0.0, np.nan))
        out[period] = a4.trend(x, f.loc[m, layer].to_numpy(dtype="float64"), cfg, rng)
    return out


def per_instrument_gap(f: pd.DataFrame, a: pd.Series, b: pd.Series, layer: str) -> pd.DataFrame:
    """The a-minus-b gap in mean `layer`, per instrument and period (no intervals:
    this is the sign-consistency check behind the pooled figure)."""
    rows = []
    for (sym, period), x in f.groupby(["symbol", "period"], sort=True):
        ia, ib = a.loc[x.index], b.loc[x.index]
        ma, mb = x.loc[ia, layer].mean(), x.loc[ib, layer].mean()
        rows.append({"symbol": sym, "period": period, "n_a": int(ia.sum()), "n_b": int(ib.sum()),
                     "mean_a": float(ma), "mean_b": float(mb), "gap": float(ma - mb)})
    return pd.DataFrame(rows)
