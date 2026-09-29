"""analysis/mfe.py: favourable excursion against the target, and the plain bracket.

Pinned, on hand-built 1-minute MES bars with the real cost model and
execution config:
  - The fill bar's extreme never counts: its high may come before the fill,
    and on the fill bar only a stop can exit (execution.yaml fill_bar_exits).
  - Reaching the target means trading through it by trade_through_ticks, as
    the simulator's target limit needs. A touch is not a reach.
  - A stop exit's own bar never counts (the simulator's adverse-first rule).
  - Short trades mirror long ones.
  - The bracket is the lifecycle's first scan: the original stop and target,
    no breakeven, no trailing, flattened at the day boundary. It holds
    through a pullback to entry that breakeven would have scratched.
  - The bracket's R comes from the engine's own round_trip; signal R adds
    back the modelled slippage of the bracket's exit kind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis import mfe
from backtest.setup_study import StudyConfig
from data.pipeline import load_symbol_config
from features.schema import load_params


def _y(p):
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def sc():
    return StudyConfig.from_configs(costs_cfg=_y("config/costs.yaml"), exec_cfg=_y("config/execution.yaml"),
                                    risk_cfg=_y("config/risk.yaml"), symbol_cfg=load_symbol_config("MES"),
                                    timeframe=str(load_params("MES").get("timeframes.entry")))


def bars(path, *, day_break=None, start="2024-03-05 15:00"):
    """1-minute bars from (open, high, low, close) tuples; a new trade date from `day_break` on."""
    ts = pd.date_range(start, periods=len(path), freq="1min", tz="UTC")
    o, h, l, c = (np.array(x, dtype="float64") for x in zip(*path))
    td = np.full(len(path), pd.Timestamp("2024-03-05").date(), dtype=object)
    if day_break is not None:
        td[day_break:] = pd.Timestamp("2024-03-06").date()
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c,
                         "raw_symbol": "MESH4", "trade_date": td})


def trade(direction="long", entry=100.0, stop=99.0, target=102.0, **kw):
    """A study-shaped filled row: 1 contract, 4 ticks = 1 point = 1 R."""
    return dict(direction=direction, entry_order=entry, entry_fill=entry, stop_order=stop,
                target_order=target, contracts=1, **kw)


FLAT = (100.0, 100.0, 100.0, 100.0)


# --- the in-trade excursion --------------------------------------------------------------

def test_the_fill_bar_extreme_never_counts(sc):
    b = bars([(100.0, 110.0, 99.5, 100.0), FLAT, (100.0, 101.0, 100.0, 100.5), FLAT])
    e = mfe.excursion(b, trade(), entry_pos=0, exit_pos=3, exit_kind="day_boundary_flatten", cost=sc.cost, cfg=sc.cfg)
    assert e["mfe_r"] == pytest.approx(1.0)                 # bar 2's 101, not the fill bar's 110
    assert not e["reached_target"]


def test_a_touch_of_the_target_is_not_a_reach(sc):
    touch = bars([FLAT, (100.0, 102.0, 100.0, 101.0), FLAT])
    through = bars([FLAT, (100.0, 102.25, 100.0, 101.0), FLAT])
    k = dict(entry_pos=0, exit_pos=2, exit_kind="day_boundary_flatten", cost=sc.cost, cfg=sc.cfg)
    t = mfe.excursion(touch, trade(), **k)
    assert t["mfe_r"] == pytest.approx(2.0) and t["target_r"] == pytest.approx(2.0)
    assert not t["reached_target"]
    assert mfe.excursion(through, trade(), **k)["reached_target"]
    assert t["reach_needed_r"] == pytest.approx(2.25)       # target + 1 tick, in R


def test_a_stop_exits_own_bar_never_counts(sc):
    b = bars([FLAT, (100.0, 100.5, 100.0, 100.25), (100.25, 103.0, 98.5, 99.0)])
    e = mfe.excursion(b, trade(), entry_pos=0, exit_pos=2, exit_kind="stop_market_exit", cost=sc.cost, cfg=sc.cfg)
    assert e["mfe_r"] == pytest.approx(0.5)


def test_a_target_exits_own_bar_counts(sc):
    b = bars([FLAT, (100.0, 102.25, 100.0, 102.0)])
    e = mfe.excursion(b, trade(), entry_pos=0, exit_pos=1, exit_kind="target_limit", cost=sc.cost, cfg=sc.cfg)
    assert e["reached_target"] and e["mfe_r"] == pytest.approx(2.25)


def test_no_bar_after_the_fill_means_zero_excursion(sc):
    b = bars([(100.0, 100.0, 98.5, 98.75)])
    e = mfe.excursion(b, trade(), entry_pos=0, exit_pos=0, exit_kind="stop_market_exit", cost=sc.cost, cfg=sc.cfg)
    assert e["mfe_r"] == 0.0 and not e["reached_1r"]


def test_a_trade_that_only_moves_against_it_has_zero_excursion_not_negative(sc):
    b = bars([FLAT, (99.75, 99.75, 99.25, 99.5), (99.5, 99.75, 99.25, 99.5)])
    e = mfe.excursion(b, trade(), entry_pos=0, exit_pos=2, exit_kind="day_boundary_flatten", cost=sc.cost, cfg=sc.cfg)
    assert e["mfe_r"] == 0.0


def test_short_mirrors_long(sc):
    b = bars([FLAT, (100.0, 100.0, 97.75, 98.0), FLAT])
    e = mfe.excursion(b, trade("short", entry=100.0, stop=101.0, target=98.0), entry_pos=0, exit_pos=2,
                      exit_kind="day_boundary_flatten", cost=sc.cost, cfg=sc.cfg)
    assert e["mfe_r"] == pytest.approx(2.25) and e["reached_target"] and e["reached_2r"]


def test_reach_levels(sc):
    b = bars([FLAT, (100.0, 101.5, 100.0, 101.0), FLAT])
    e = mfe.excursion(b, trade(target=103.0), entry_pos=0, exit_pos=2, exit_kind="day_boundary_flatten",
                      cost=sc.cost, cfg=sc.cfg)
    assert (e["reached_1r"], e["reached_2r"], e["reached_target"]) == (True, False, False)


# --- the bracket ---------------------------------------------------------------------------

def test_the_bracket_holds_through_a_pullback_breakeven_would_scratch(sc):
    # up 1.5 R, back to entry (breakeven would scratch here), then through the target
    b = bars([FLAT, (100.0, 101.5, 100.0, 101.0), (101.0, 101.0, 100.0, 100.0),
              (100.0, 102.25, 100.0, 102.0), FLAT])
    r = mfe.bracket(b, trade(), entry_ts=b["ts"].iloc[0], cost=sc.cost, cfg=sc.cfg)
    assert r["bracket_exit"] == "target_limit"
    assert r["bracket_r_gross"] == pytest.approx(2.0)                 # a limit fills at its price
    fees_r = 2 * sc.cost.fees(1) / (1.0 * sc.cost.point_value)
    assert r["bracket_r_net"] == pytest.approx(2.0 - fees_r)
    assert r["bracket_r_signal"] == pytest.approx(2.0)
    assert r["bracket_reached_target"] and r["bracket_mfe_r"] == pytest.approx(2.25)


def test_the_bracket_target_cannot_fill_on_the_fill_bar(sc):
    # the fill bar trades through the target, but a target can't fill there
    b = bars([(100.0, 102.5, 99.5, 100.0), FLAT, FLAT], day_break=2)
    r = mfe.bracket(b, trade(), entry_ts=b["ts"].iloc[0], cost=sc.cost, cfg=sc.cfg)
    assert r["bracket_exit"] == "day_boundary_flatten" and not r["bracket_reached_target"]


def test_the_bracket_stop_loses_one_r_plus_slippage(sc):
    b = bars([FLAT, (100.0, 100.5, 98.75, 98.75), FLAT])
    r = mfe.bracket(b, trade(), entry_ts=b["ts"].iloc[0], cost=sc.cost, cfg=sc.cfg)
    slip_r = sc.cost.slippage("stop_market_exit") / 1.0
    assert r["bracket_exit"] == "stop_market_exit"
    assert r["bracket_r_gross"] == pytest.approx(-1.0 - slip_r)
    assert r["bracket_r_signal"] == pytest.approx(-1.0)
    assert r["bracket_mfe_r"] == pytest.approx(0.0)                 # its only move came on the stop bar


def test_the_bracket_flattens_at_the_day_boundary(sc):
    b = bars([FLAT, (100.0, 101.0, 100.0, 100.5), (100.5, 100.5, 100.5, 100.5), FLAT], day_break=3)
    r = mfe.bracket(b, trade(), entry_ts=b["ts"].iloc[0], cost=sc.cost, cfg=sc.cfg)
    assert r["bracket_exit"] == "day_boundary_flatten"
    assert r["bracket_r_signal"] == pytest.approx(0.5)              # the last bar's close, before slippage


def test_a_bracket_still_open_when_the_data_ends_is_unresolved_not_invented(sc):
    # Real data (2026-09-29): an MBT trade on the last, partial session. Its
    # actual trade closed, but the bracket would still be holding.
    b = bars([FLAT, (100.0, 101.0, 99.5, 100.5), FLAT])          # no day boundary before the data ends
    r = mfe.bracket(b, trade(), entry_ts=b["ts"].iloc[0], cost=sc.cost, cfg=sc.cfg)
    assert r["bracket_exit"] == "unresolved"
    assert all(np.isnan(r[k]) for k in ("bracket_r_gross", "bracket_r_net", "bracket_r_signal", "bracket_mfe_r"))
    assert not r["bracket_reached_target"]


def test_the_entry_must_be_on_a_bar(sc):
    b = bars([FLAT, FLAT])
    with pytest.raises(ValueError, match="entry"):
        mfe.bracket(b, trade(), entry_ts=b["ts"].iloc[0] + pd.Timedelta(seconds=30), cost=sc.cost, cfg=sc.cfg)


# --- summaries -------------------------------------------------------------------------------

def test_summary_rates_and_effects_leave_unresolved_brackets_out():
    from tests.test_a4_analysis import cfg as a4cfg
    f = pd.DataFrame({
        "period": ["IS"] * 5, "kind": ["momentum"] * 5,
        "target_r": [2.0] * 5, "mfe_r": [0.5, 1.2, 2.5, 3.0, 0.0],
        "reached_1r": [False, True, True, True, False], "reached_2r": [False, False, True, True, False],
        "reached_target": [False, False, True, True, False],
        "exit_kind": ["stop_market_exit", "stop_market_exit", "target_limit", "stop_market_exit", "stop_market_exit"],
        "bracket_exit": ["stop_market_exit", "target_limit", "target_limit", "target_limit", "unresolved"],
        "bracket_reached_target": [False, True, True, True, False],
        "bracket_mfe_r": [0.5, 2.2, 2.5, 2.3, np.nan],
        "r_signal": [-1.0, 0.0, 2.0, 0.0, 5.0], "bracket_r_signal": [-1.0, 2.0, 2.0, 2.0, np.nan],
        "r_gross": [-1.0, 0.0, 2.0, 0.0, 5.0], "bracket_r_gross": [-1.0, 2.0, 2.0, 2.0, np.nan],
        "r_net": [-1.1, -0.1, 1.9, -0.1, 4.9], "bracket_r_net": [-1.1, 1.9, 1.9, 1.9, np.nan],
    })
    s = mfe.summary(f, "kind", a4cfg(draws=100), np.random.default_rng(0)).iloc[0]
    assert (s["n"], s["unresolved"]) == (5, 1)
    assert s["reached_target"] == pytest.approx(2 / 5) and s["target_filled"] == pytest.approx(1 / 5)
    assert s["bracket_target"] == pytest.approx(3 / 4)                  # the unresolved one left out
    assert s["actual_r_signal"] == pytest.approx(0.25)                   # (-1 + 0 + 2 + 0) / 4, not the 5.0
    assert s["effect_r_signal"] == pytest.approx((0 - 2 + 0 - 2) / 4)
    assert s["mfe_over_target_med"] == pytest.approx(0.6)


def test_paired_difference_is_actual_minus_bracket_per_trade():
    f = pd.DataFrame({"period": ["IS"] * 4, "kind": ["momentum"] * 4,
                      "r_signal": [0.0, 0.0, -1.0, 2.0], "bracket_r_signal": [2.0, -1.0, -1.0, 2.0]})
    d = mfe.paired(f, "r_signal", "bracket_r_signal")
    assert list(d) == [-2.0, 1.0, 0.0, 0.0]
