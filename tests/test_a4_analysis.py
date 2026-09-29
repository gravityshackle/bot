"""Phase 4.4: the A4 analysis (analysis/a4.py), on hand-built frames.

What is pinned:
  - Bands: RR bands are left-closed with an open top, and the cap must be an
    edge. Score quintile edges come from in-sample sized setups only, so
    out-of-sample values can't move them.
  - Populations: fill rate is over sized setups. Win rate, expectancy and the
    R distribution are over filled trades only, on the verdict metric
    (r_gross: after slippage, before fees; decided 2026-09-29). Signal R
    (exit at the intended price) and r_net are the layers either side of it. Discarded setups (0
    contracts) are counted but never enter an outcome figure.
  - Stop width in ATR: the previous completed session's daily ATR, looked up
    by each setup's own trade date, so a session's own range never scales its
    own stops.
  - Cost attribution names which layer makes a trend: slippage when signal R
    and r_gross disagree, fees when r_gross and r_net disagree. It is kept
    separate from the sizing check.
  - Trend and cap figures carry bootstrap intervals and a direction. The
    direction is "flat" whenever the interval spans zero.
  - The stricter check does what it claims. Suppose outcomes depend only on
    stop width, and in-sample high-RR setups tend to have wider stops. Then
    the raw in-sample RR trend looks real, and restricting to the OOS
    stop-width range does NOT remove it, because the link survives inside the
    range. The stratified trend (RR bands compared within stop-width slices,
    weighted to the OOS mix) loses it, and the verdict is "unresolved:
    sizing". When the trend is real (outcomes depend on RR band whatever the
    stop width), every version keeps it and the verdict stands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import a4

SPLIT = pd.Timestamp("2024-09-12", tz="UTC")


def cfg(**over) -> a4.A4Config:
    base = dict(split=SPLIT, score_quantiles=5, rr_edges=(2.0, 2.5, 3.0, 4.0, 6.0), rr_cap=4.0,
                draws=400, seed=7, confidence=0.95, min_cell_trades=30, tail_top_fraction=0.05,
                sizing_shift_symbols=("MNQ", "MGC", "SIL"), directional_only_symbols=("SIL",),
                support_quantiles=(0.05, 0.95), reweight_bins=10, atr_period=14, outcome="r_gross")
    base.update(over)
    return a4.A4Config(**base)


def frame(n, *, ts, rr, r, score=None, contracts=1, filled=True, path=None, stop_atr=None,
          slip=0.0, fee=0.0):
    """A minimal study-shaped frame. Every array argument has length n.

    `r` is r_gross. A long entry at 100 with its stop at 99 makes 1 point = 1 R,
    so exit_reference = 100 + r + slip gives signal R = r + slip, and fees of
    `fee` R give r_net = r - fee."""
    r = np.asarray(r, dtype="float64") * np.ones(n)
    slip = np.asarray(slip, dtype="float64") * np.ones(n)
    fee = np.asarray(fee, dtype="float64") * np.ones(n)
    df = pd.DataFrame({
        "ts": ts, "rr": rr, "direction": "long", "entry_fill": 100.0, "stop_order": 99.0,
        "exit_fill": 100.0 + r, "exit_reference": 100.0 + r + slip,
        "r_gross": r, "r_net": r - fee, "risk_usd": 1.0, "fees_usd": fee, "net_usd": r - fee,
        "score": score if score is not None else np.linspace(40, 80, n),
        "contracts": contracts,
        "entry_status": np.where(np.asarray(filled) if not np.isscalar(filled) else np.full(n, filled),
                                 "filled", "expired"),
        "path": path if path is not None else "stopped_out",
    })
    out = ["r_net", "r_gross", "exit_fill", "exit_reference", "entry_fill", "fees_usd", "net_usd", "risk_usd"]
    df.loc[df["entry_status"] != "filled", out] = np.nan
    df.loc[df["contracts"] == 0, "entry_status"] = pd.NA
    df.loc[df["contracts"] == 0, out] = np.nan
    if stop_atr is not None:
        df["stop_atr"] = stop_atr
    return df


# --- bands ---------------------------------------------------------------------------

def test_rr_bands_are_left_closed_with_an_open_top():
    b = a4.rr_band(pd.Series([2.0, 2.49, 2.5, 3.99, 4.0, 5.99, 6.0, 3330.0]), (2.0, 2.5, 3.0, 4.0, 6.0))
    assert list(b.astype(str)) == ["[2, 2.5)", "[2, 2.5)", "[2.5, 3)", "[3, 4)", "[4, 6)", "[4, 6)",
                                   "[6, inf)", "[6, inf)"]
    assert list(b.cat.categories) == ["[2, 2.5)", "[2.5, 3)", "[3, 4)", "[4, 6)", "[6, inf)"]


def test_rr_below_the_first_edge_is_an_error_not_a_band():
    with pytest.raises(ValueError, match="below"):
        a4.rr_band(pd.Series([1.99, 2.5]), (2.0, 2.5, 3.0, 4.0, 6.0))


def test_the_cap_must_be_a_band_edge():
    with pytest.raises(ValueError, match="rr_cap"):
        cfg(rr_cap=3.5).validate()
    cfg().validate()


def test_live_and_saturated_bands_split_at_the_cap():
    c = cfg()
    assert a4.live_bands(c) == ["[2, 2.5)", "[2.5, 3)", "[3, 4)"]
    assert a4.saturated_bands(c) == ["[4, 6)", "[6, inf)"]


def test_score_edges_come_from_in_sample_only():
    is_scores = pd.Series(np.arange(100, dtype=float))
    edges = a4.score_edges(is_scores, 5)
    assert np.allclose(edges, [19.8, 39.6, 59.4, 79.2])
    # out-of-sample values are banded against those edges, clamped at the ends
    q = a4.score_band(pd.Series([-50.0, 0.0, 19.8, 50.0, 99.0, 500.0]), edges)
    assert list(q) == [1, 1, 2, 3, 5, 5]


def test_prepare_bands_oos_by_in_sample_score_edges():
    n = 200
    ts = pd.date_range("2023-01-01", periods=n, freq="7D", tz="UTC")
    score = np.where(ts < SPLIT, np.linspace(0, 100, n), 1000.0)     # OOS scores far above IS
    p = a4.prepare(frame(n, ts=ts, rr=np.full(n, 2.2), r=np.zeros(n), score=score), cfg())
    assert set(p.loc[p["period"] == "OOS", "score_q"]) == {5}
    assert set(p.loc[p["period"] == "IS", "score_q"]) == {1, 2, 3, 4, 5}
    assert (p["period"] == "IS").sum() == (ts < SPLIT).sum()


# --- populations ---------------------------------------------------------------------

def test_cell_stats_populations():
    # 10 sized: 6 filled (4 wins, 1 scratch), 4 expired; plus 5 discarded
    r = [1.5, 2.0, -1.0, 3.0, 0.5, 0.0] + [np.nan] * 4 + [np.nan] * 5
    df = pd.DataFrame({
        "contracts": [1] * 10 + [0] * 5,
        "entry_status": ["filled"] * 6 + ["expired"] * 4 + [pd.NA] * 5,
        "r_gross": r, "r_signal": np.array(r) + 0.1, "r_net": np.array(r) - 0.2,
        "fees_usd": [1.0] * 6 + [np.nan] * 9, "risk_usd": [2.0, 2.0, 0.5, 4.0, 1.0, 1.0] + [np.nan] * 9,
        "net_usd": [1.0] * 6 + [np.nan] * 9,
        "path": ["planned_exit", "trailing", "stopped_out", "planned_exit", "trailing", "scratch"]
                + ["not_filled"] * 4 + [pd.NA] * 5,
    })
    df = a4.add_flags(df, "r_gross")
    s = a4.cell_stats(df, cfg(min_cell_trades=5), np.random.default_rng(0))
    assert (s["setups"], s["discarded"], s["sized"], s["filled"]) == (15, 5, 10, 6)
    assert s["fill_rate"] == pytest.approx(0.6)
    assert s["win_rate"] == pytest.approx(4 / 6)
    assert s["scratch_rate"] == pytest.approx(1 / 6)
    assert s["exp_r"] == pytest.approx(np.mean([1.5, 2.0, -1.0, 3.0, 0.5, 0.0]))
    assert s["exp_lo"] <= s["exp_r"] <= s["exp_hi"]
    assert s["median_r"] == pytest.approx(1.0)                # of [-1, 0, 0.5, 1.5, 2, 3]
    assert s["exp_signal"] == pytest.approx(s["exp_r"] + 0.1)
    assert s["exp_net"] == pytest.approx(s["exp_r"] - 0.2)
    assert s["fee_over_risk_median"] == pytest.approx(np.median([0.5, 0.5, 2.0, 0.25, 1.0, 1.0]))
    assert s["fees_ge_risk"] == pytest.approx(3 / 6)
    assert not s["thin"]
    assert a4.cell_stats(df, cfg(min_cell_trades=7), np.random.default_rng(0))["thin"]


def test_a_discarded_setup_marked_filled_is_refused():
    n = 50
    df = frame(n, ts=pd.date_range("2023-01-01", periods=n, freq="7D", tz="UTC"),
               rr=np.full(n, 2.2), r=np.zeros(n))
    df.loc[3, "contracts"] = 0                      # discarded, yet...
    df.loc[3, "entry_status"] = "filled"            # ...marked filled
    with pytest.raises(ValueError, match="sized to 0 contracts are marked filled"):
        a4.prepare(df, cfg())


def test_tail_share_is_the_top_fraction_of_winners_share_of_positive_r():
    r = pd.Series([10.0] + [1.0] * 19 + [-1.0] * 30)     # 20 winners, top 5% = 1 winner
    assert a4.tail_share(r, 0.05) == pytest.approx(10 / 29)
    assert a4.tail_share(pd.Series([10.0, 1.0, -1.0]), 0.05) == pytest.approx(10 / 11)  # at least one
    assert np.isnan(a4.tail_share(pd.Series([-1.0, -2.0]), 0.05))


# --- stop width in ATR -------------------------------------------------------------------

def test_prior_session_atr_uses_only_completed_sessions():
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    daily = pd.DataFrame({"trade_date": days.date, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    base = a4.prior_session_atr(daily, 14)
    shocked = daily.copy()
    shocked.loc[20, ["high", "low"]] = [150.0, 50.0]            # session 20 is enormous
    after = a4.prior_session_atr(shocked, 14)
    d = daily["trade_date"]
    assert after[d[20]] == base[d[20]]                          # its own stops don't see it
    assert after[d[21]] > base[d[21]]                           # the next session's do
    assert np.isnan(base[d[0]])


def test_stop_width_atr_looks_up_each_setups_own_trade_date():
    scfg = {"session": {"timezone": "America/Chicago"}, "day_boundary": "17:00"}
    prior = pd.Series([2.0, 4.0], index=pd.Index([pd.Timestamp("2024-03-05").date(),
                                                  pd.Timestamp("2024-03-06").date()], name="trade_date"))
    df = pd.DataFrame({
        # 16:59 CT on 03-05 belongs to 03-05; 17:00 CT belongs to 03-06
        "ts": pd.to_datetime(["2024-03-05 22:59", "2024-03-05 23:00"], utc=True),
        "entry": [100.0, 100.0], "stop_order": [98.0, 102.0],
    })
    assert list(a4.stop_width_atr(df, prior, scfg)) == [1.0, 0.5]


# --- trend, cap, verdict ---------------------------------------------------------------------

def _trend_data(rng, n, slope):
    band = rng.integers(0, 3, n)
    return band, slope * band + rng.normal(0, 1, n)


def test_trend_direction_and_interval():
    rng = np.random.default_rng(1)
    x, y = _trend_data(rng, 3000, 0.3)
    t = a4.trend(x, y, cfg(), np.random.default_rng(2))
    assert t["direction"] == "up" and t["lo"] > 0 and t["slope"] == pytest.approx(0.3, abs=0.06)
    x, y = _trend_data(rng, 3000, -0.3)
    assert a4.trend(x, y, cfg(), np.random.default_rng(2))["direction"] == "down"
    x, y = _trend_data(rng, 3000, 0.0)
    t = a4.trend(x, y, cfg(), np.random.default_rng(2))
    assert t["direction"] == "flat" and t["lo"] < 0 < t["hi"]


def test_trend_bootstrap_is_reproducible_from_the_seed():
    rng = np.random.default_rng(1)
    x, y = _trend_data(rng, 500, 0.1)
    a = a4.trend(x, y, cfg(), np.random.default_rng(5))
    b = a4.trend(x, y, cfg(), np.random.default_rng(5))
    assert a == b


def test_weighted_trend_matches_weighted_least_squares():
    x = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    y = np.array([0, 1, 1, 2, 5, 3], dtype=float)
    w = np.array([1, 1, 2, 2, 1, 3], dtype=float)
    X = np.column_stack([np.ones_like(x), x]) * np.sqrt(w)[:, None]
    ref = np.linalg.lstsq(X, y * np.sqrt(w), rcond=None)[0][1]
    assert a4.slope(x, y, w) == pytest.approx(ref)


def test_cap_gap_compares_saturated_bands_with_the_top_live_band():
    band = pd.Series(pd.Categorical(["[3, 4)"] * 400 + ["[4, 6)"] * 200 + ["[6, inf)"] * 200,
                                    categories=a4.rr_band(pd.Series([2.0]), cfg().rr_edges).cat.categories))
    r = np.r_[np.full(400, 0.2), np.full(400, -0.3)] + np.random.default_rng(3).normal(0, 0.5, 800)
    g = a4.cap_gap(band, r, cfg(), np.random.default_rng(4))
    assert g["direction"] == "down" and g["gap"] == pytest.approx(-0.5, abs=0.1)


@pytest.mark.parametrize("is_dir,oos_dir,expected", [
    ("up", "up", "consistent: up"), ("down", "down", "consistent: down"),
    ("flat", "flat", "consistent: flat"), ("up", "flat", "not replicated"),
    ("up", "down", "not replicated"), ("flat", "down", "not replicated"),
])
def test_verdict_for_directly_reported_instruments(is_dir, oos_dir, expected):
    assert a4.verdict(is_dir, oos_dir) == expected


def test_verdict_for_sizing_shift_instruments_needs_both_matched_trends():
    assert a4.verdict("up", "up", matched=("up", "up")) == "consistent: up"
    assert a4.verdict("up", "up", matched=("up", "flat")) == "unresolved: sizing"
    assert a4.verdict("up", "up", matched=("flat", "flat")) == "unresolved: sizing"
    assert a4.verdict("up", "flat", matched=("up", "up")) == "not replicated"
    # a failed matched check is the more specific finding, so it wins
    assert a4.verdict("up", "flat", matched=("flat", "flat")) == "unresolved: sizing"


def test_too_little_data_is_its_own_verdict_never_a_direction():
    assert a4.verdict("insufficient", "up") == "insufficient data"
    assert a4.verdict("up", "insufficient") == "insufficient data"
    assert a4.verdict("up", "up", matched=("insufficient", "up")) == "unresolved: sizing"


def test_trend_on_one_band_or_too_few_trades_is_insufficient():
    c = cfg(min_cell_trades=30)
    assert a4.trend(np.zeros(500), np.ones(500), c, np.random.default_rng(0))["direction"] == "insufficient"
    x = np.r_[np.zeros(500), np.ones(10)]
    assert a4.trend(x, np.random.default_rng(1).normal(size=510), c,
                    np.random.default_rng(0))["direction"] == "insufficient"


# --- the stricter check, end to end ------------------------------------------------------------

def test_support_mask_keeps_in_sample_inside_the_oos_range():
    oos = pd.Series(np.linspace(1.0, 2.0, 101))
    is_ = pd.Series([0.5, 1.0, 1.04, 1.5, 1.96, 2.0, 3.0])
    m = a4.support_mask(is_, oos, (0.05, 0.95))
    lo, hi = np.quantile(oos, [0.05, 0.95])
    assert list(m) == list((is_ >= lo) & (is_ <= hi))


def test_reweighting_matches_the_oos_stop_width_mix():
    rng = np.random.default_rng(8)
    is_ = pd.Series(rng.uniform(0, 1, 4000) ** 0.7)       # skewed wide
    oos = pd.Series(rng.uniform(0, 1, 3000) ** 1.5)       # skewed narrow, same range
    w = a4.reweight(is_, oos, 10)
    edges = np.quantile(oos, np.linspace(0, 1, 11))
    bins_is = np.clip(np.searchsorted(edges, is_, side="right") - 1, 0, 9)
    share = np.bincount(bins_is, weights=w, minlength=10) / w.sum()
    assert np.allclose(share, 0.1, atol=0.005)
    assert (w >= 0).all()
    assert a4.oos_coverage(is_, oos, 10) == 1.0


def test_oos_slices_without_in_sample_trades_are_reported_not_hidden():
    is_ = pd.Series(np.linspace(0.5, 1.0, 1000))          # in-sample never had stops below 0.5
    oos = pd.Series(np.linspace(0.0, 1.0, 1000))
    w = a4.reweight(is_, oos, 10)
    assert a4.oos_coverage(is_, oos, 10) == pytest.approx(0.5, abs=0.01)
    assert w.sum() > 0


def _confound_frames(real_rr_effect: bool):
    """Two periods. Outcomes depend on stop width (confound) or on RR band (real).

    In-sample, high-RR setups tend to have wide stops. Out-of-sample, sizing
    has removed the wide stops, the MNQ/MGC/SIL situation."""
    rng = np.random.default_rng(11)
    n_is, n_oos = 9000, 5000
    band_is = rng.integers(0, 3, n_is)
    stop_is = rng.uniform(0.2, 1.0, n_is) + 0.4 * band_is          # wider stops in higher bands
    band_oos = rng.integers(0, 3, n_oos)
    stop_oos = rng.uniform(0.2, 1.2, n_oos)                         # wide stops gone, every band
    rr_of = np.array([2.2, 2.7, 3.5])

    def outcome(band, stop):
        effect = 0.3 * band if real_rr_effect else 0.4 * (stop - 1.0)
        return effect + rng.normal(0, 1, len(band))

    ts_is = pd.date_range("2021-10-01", periods=n_is, freq="3h", tz="UTC")
    ts_oos = pd.date_range("2024-10-01", periods=n_oos, freq="3h", tz="UTC")
    df = pd.concat([
        frame(n_is, ts=ts_is, rr=rr_of[band_is], r=outcome(band_is, stop_is), stop_atr=stop_is),
        frame(n_oos, ts=ts_oos, rr=rr_of[band_oos], r=outcome(band_oos, stop_oos), stop_atr=stop_oos),
    ], ignore_index=True)
    return df


def test_stricter_check_exposes_a_trend_made_by_sizing_composition():
    c = cfg(draws=300)
    res = a4.rr_component_answer(a4.prepare(_confound_frames(real_rr_effect=False), c), c, "MNQ")
    assert res["IS"]["direction"] == "up"                     # the raw in-sample trend looks real
    assert res["IS_restricted"]["direction"] == "up"          # restriction alone doesn't remove it
    assert res["IS_stratified"]["direction"] == "flat"        # comparing within stop width does
    assert res["IS_stratified"]["oos_coverage"] == pytest.approx(1.0)
    # the restriction is to the OOS range, not in-sample's own
    p = a4.prepare(_confound_frames(real_rr_effect=False), c)
    f = p[p["filled"]]
    lo, hi = np.quantile(f.loc[f["period"] == "OOS", "stop_atr"], c.support_quantiles)
    s_is = f.loc[f["period"] == "IS", "stop_atr"]
    assert res["IS_restricted"]["n"] == int(((s_is >= lo) & (s_is <= hi)).sum())
    assert res["IS_restricted"]["n"] < res["IS"]["n"]
    assert res["verdict"] == "unresolved: sizing"


def test_stricter_check_keeps_a_real_trend():
    c = cfg(draws=300)
    res = a4.rr_component_answer(a4.prepare(_confound_frames(real_rr_effect=True), c), c, "MNQ")
    assert res["IS"]["direction"] == "up"
    assert res["IS_restricted"]["direction"] == "up"
    assert res["IS_stratified"]["direction"] == "up"
    assert res["OOS"]["direction"] == "up"
    assert res["verdict"] == "consistent: up"


def test_direct_instruments_skip_the_stricter_check_but_flag_directional_only():
    c = cfg(draws=200)
    p = a4.prepare(_confound_frames(real_rr_effect=True), c)
    mes = a4.rr_component_answer(p, c, "MES")
    assert "IS_restricted" not in mes and mes["verdict"] == "consistent: up" and not mes["directional_only"]
    sil = a4.rr_component_answer(p, c, "SIL")
    assert sil["directional_only"] and "IS_restricted" in sil


def test_ks_statistic():
    a = pd.Series(np.arange(100, dtype=float))
    assert a4.ks(a, a)["D"] == 0.0
    b = pd.Series(np.arange(100, dtype=float) + 1000)
    k = a4.ks(a, b)
    assert k["D"] == 1.0 and k["p"] < 1e-6


# --- cost layers ---------------------------------------------------------------------------------

def test_signal_r_is_the_exit_at_its_intended_price_long_and_short():
    df = pd.DataFrame({"direction": ["long", "short"], "entry_fill": [100.0, 100.0],
                       "stop_order": [98.0, 102.0], "exit_reference": [104.0, 97.0],
                       "entry_status": ["filled", "filled"]})
    assert list(a4.signal_r(df)) == [2.0, 1.5]


def test_win_rate_uses_the_verdict_metric_not_r_net():
    n = 60
    p = a4.prepare(frame(n, ts=pd.date_range("2023-01-01", periods=n, freq="7D", tz="UTC"),
                         rr=np.full(n, 2.2), r=np.full(n, 0.2), fee=0.5), cfg())
    assert p["win"].all() and (p["r_net"] < 0).all()


def _cost_frames(kind: str):
    """r_gross (or signal R) flat across live RR bands; a cost that grows with the band."""
    rng = np.random.default_rng(21)
    n = 6000
    band = rng.integers(0, 3, n)
    ts = np.r_[pd.date_range("2021-10-01", periods=n // 2, freq="3h", tz="UTC"),
               pd.date_range("2024-10-01", periods=n // 2, freq="3h", tz="UTC")]
    base = rng.normal(0, 1, n)
    cost = 0.4 * band
    rr = np.array([2.2, 2.7, 3.5])[band]
    if kind == "fees":
        return frame(n, ts=ts, rr=rr, r=base, fee=cost, stop_atr=rng.uniform(0.2, 1.0, n))
    return frame(n, ts=ts, rr=rr, r=base - cost, slip=cost, stop_atr=rng.uniform(0.2, 1.0, n))


def test_attribution_names_fees_when_only_fees_make_the_trend():
    c = cfg(draws=300)
    res = a4.rr_component_answer(a4.prepare(_cost_frames("fees"), c), c, "MNQ")
    assert res["verdict"] == "consistent: flat"                      # r_gross decides
    assert res["layers"]["IS"]["r_net"]["direction"] == "down"
    assert res["attribution"]["fees"] == {"IS": True, "OOS": True}
    assert res["attribution"]["slippage"] == {"IS": False, "OOS": False}
    assert res["attribution"]["sizing"] is False


def test_attribution_names_slippage_separately_from_fees():
    c = cfg(draws=300)
    res = a4.rr_component_answer(a4.prepare(_cost_frames("slippage"), c), c, "MNQ")
    assert res["layers"]["IS"]["r_signal"]["direction"] == "flat"
    assert res["IS"]["direction"] == "down"                          # r_gross carries the slippage
    assert res["attribution"]["slippage"] == {"IS": True, "OOS": True}
    assert res["attribution"]["fees"] == {"IS": False, "OOS": False}


def test_sizing_attribution_is_separate_from_cost_attribution():
    c = cfg(draws=300)
    res = a4.rr_component_answer(a4.prepare(_confound_frames(real_rr_effect=False), c), c, "MNQ")
    assert res["attribution"]["sizing"] is True
    assert res["attribution"]["fees"] == {"IS": False, "OOS": False}
    assert res["attribution"]["slippage"] == {"IS": False, "OOS": False}
    mes = a4.rr_component_answer(a4.prepare(_confound_frames(real_rr_effect=False), c), c, "MES")
    assert mes["attribution"]["sizing"] is None                       # not checked there
