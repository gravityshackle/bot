"""analysis/edge.py: per-trigger expectancy, pooled across instruments.

Pinned: the population is filled trades only. Group means and shares are per
period. The instrument-balanced mean weights instruments equally, whatever
their trade counts. The a-minus-b gap is a difference in means with an
interval, "flat" when it spans zero. The per-instrument gaps reproduce the
pooled gap's sign-consistency check exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import a4, edge
from tests.test_a4_analysis import SLIP, cfg, frame


def _prepared(sym, n, kinds, r, start="2022-01-01"):
    ts = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    df = frame(n, ts=ts, rr=np.full(n, 2.2), r=r)
    df["kind"] = kinds
    return a4.prepare(df, cfg(), slippage=SLIP)


def test_pooled_keeps_filled_trades_only_and_tags_the_instrument():
    p = _prepared("MES", 10, ["momentum"] * 10, np.zeros(10))
    p.loc[p.index[:3], "filled"] = False
    f = edge.pooled({"MES": p, "MNQ": _prepared("MNQ", 5, ["momentum"] * 5, np.zeros(5))})
    assert len(f) == 12 and set(f["symbol"]) == {"MES", "MNQ"}


def test_group_table_means_shares_and_balanced_mean():
    # MES: 90 momentum at -0.5 and 10 reversal at +1. MNQ: 10 momentum at +0.5, 90 reversal at 0.
    mes = _prepared("MES", 100, ["momentum"] * 90 + ["reversal"] * 10, np.r_[np.full(90, -0.5), np.full(10, 1.0)])
    mnq = _prepared("MNQ", 100, ["momentum"] * 10 + ["reversal"] * 90, np.r_[np.full(10, 0.5), np.full(90, 0.0)])
    f = edge.pooled({"MES": mes, "MNQ": mnq})
    t = edge.group_table(f, "kind", cfg(draws=200, min_cell_trades=5), np.random.default_rng(0))
    mom = t[(t["period"] == "IS") & (t["kind"] == "momentum")].iloc[0]
    assert mom["n"] == 100 and mom["share"] == pytest.approx(0.5)
    assert mom["signal"] == pytest.approx((90 * -0.5 + 10 * 0.5) / 100)          # pooled: trade-weighted
    assert mom["balanced_signal"] == pytest.approx((-0.5 + 0.5) / 2)              # balanced: instrument-weighted
    assert mom["instruments_positive"] == 1
    assert mom["signal_lo"] == pytest.approx(mom["signal_hi"], abs=0.2)


def test_gap_is_a_difference_in_means_with_an_interval():
    rng = np.random.default_rng(3)
    n = 4000
    kinds = np.where(rng.uniform(size=n) < 0.6, "momentum", "reversal")
    r = np.where(kinds == "momentum", -0.3, 0.1) + rng.normal(0, 1, n)
    p = _prepared("MES", n, kinds, r, start="2020-01-01")
    f = edge.pooled({"MES": p})
    g = edge.gap(f, f["kind"] == "momentum", f["kind"] == "reversal", "r_signal", cfg(draws=300),
                 np.random.default_rng(1))
    both = g["IS"]
    m = f["period"] == "IS"
    ref = f.loc[m & (f["kind"] == "momentum"), "r_signal"].mean() - f.loc[m & (f["kind"] == "reversal"), "r_signal"].mean()
    assert both["slope"] == pytest.approx(ref)
    assert both["direction"] == "down" and both["hi"] < 0


def test_per_instrument_gap_matches_group_means():
    mes = _prepared("MES", 100, ["momentum"] * 90 + ["reversal"] * 10, np.r_[np.full(90, -0.5), np.full(10, 1.0)])
    f = edge.pooled({"MES": mes})
    g = edge.per_instrument_gap(f, f["kind"] == "momentum", f["kind"] == "reversal", "r_signal")
    row = g[g["period"] == "IS"].iloc[0]
    assert (row["n_a"], row["n_b"]) == (90, 10)
    assert row["gap"] == pytest.approx(-1.5)
