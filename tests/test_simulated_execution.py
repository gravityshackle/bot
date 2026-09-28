"""Phase 4.1 fill simulator (exit spec Parts 2-3, docs/phase4_plan.md section 2).

Every test walks a hand-built 1-minute path whose outcome can be worked out
by hand. Geometry: an MES-like instrument (tick 0.25, $5/point), so a long
bids 100.00 with its stop at 99.00 and its target at 102.00. Fees are
$0.25 + $0.50 per contract per side; market orders slip 1 tick.

The load-bearing tests pin the conservative choices, each against the naive
version that would inflate results:
  - a touch is not a fill, for entries and for targets
  - the decision bar itself can never fill its own order (no lookahead)
  - the entry is live for exactly K entry-timeframe bars
  - a bar that reaches both the stop and the target is a stop
  - the bar that fills the entry can stop out but cannot reach the target
  - a stop gapped through fills at the open, not at the stop
  - only market orders slip; fees are charged on entry AND exit
  - running out of data is `unresolved`, never `expired` or a flat exit
"""
from __future__ import annotations

import copy
from datetime import date

import pandas as pd
import pytest
import yaml

from execution import simulated_execution as sx
from execution.simulated_execution import (
    CostModel,
    ExecConfig,
    entry_window,
    market_exit,
    round_trip,
    simulate_entry,
    simulate_exit,
)
from signal_engine.gates import LONG, SHORT

T0 = pd.Timestamp("2026-03-02 15:00", tz="UTC")          # 09:00 CT, mid-RTH
D0 = date(2026, 3, 2)
MIN = pd.Timedelta(minutes=1)

SYMBOL_CFG = {"symbol": "TST",
              "contract_spec": {"tick_size": 0.25, "tick_value": 1.25, "point_value": 5.0}}
COSTS_CFG = {
    "slippage": {"applies_to": ["stop_market_exit", "breaker_flatten", "day_boundary_flatten"],
                 "thin_session_multiplier": 1.0},
    "symbols": {"TST": {"ibkr_commission_side": 0.25,
                        "estimated_exchange_regulatory_side": 0.50,
                        "slippage_ticks": 1}},
}
EXEC_CFG = {
    "entry": {"k_entry_expire_bars": 3, "trade_through_ticks": 1},
    "targets": {"trade_through_ticks": 1},
    "fills": {"limit_price_improvement": False, "stop_trigger": "touch",
              "same_bar_stop_and_target": "stop", "fill_bar_exits": ["stop"]},
    "rounding": {"stop": "away_from_entry", "target": "toward_entry"},
    "day_boundary": {"policy": "flatten"},
}


def cost(symbol_cfg=SYMBOL_CFG, costs_cfg=COSTS_CFG) -> CostModel:
    return CostModel.from_config(costs_cfg, symbol_cfg)


def ecfg(**over) -> ExecConfig:
    cfg = copy.deepcopy(EXEC_CFG)
    for dotted, value in over.items():
        section, key = dotted.split("__")
        cfg[section][key] = value
    return ExecConfig.from_config(cfg)


def mbars(rows, start=T0, trade_date=D0, raw="MESH6") -> pd.DataFrame:
    """1-minute bars from (open, high, low, close) rows, one per minute.

    A row may carry a 5th element: its trade date (default `trade_date`)."""
    out = []
    for i, r in enumerate(rows):
        o, h, lo, c = r[:4]
        assert lo <= min(o, c) and h >= max(o, c), r
        out.append({"ts": start + i * MIN, "open": o, "high": h, "low": lo, "close": c,
                    "volume": 100, "raw_symbol": raw,
                    "trade_date": r[4] if len(r) > 4 else trade_date})
    return pd.DataFrame(out)


FLAT = (100.50, 100.75, 100.25, 100.50)        # nowhere near entry, stop or target


def enter(bars, *, direction=LONG, limit=100.00, stop=99.00, cfg=None, until=None,
          costm=None):
    return simulate_entry(bars, direction=direction, limit=limit, stop=stop,
                          live_from=T0, live_until=until if until is not None
                          else T0 + 15 * MIN, trade_date=D0, contracts=1,
                          cost=costm or cost(), cfg=cfg or ecfg())


def long_position(rows, *, stop=99.00, target=102.00, cfg=None, fill_row=(100.25, 100.50, 99.75, 100.25)):
    """Bars whose first row fills a long at 100.00; exits are walked from it."""
    bars = mbars([fill_row] + list(rows))
    e = enter(bars, cfg=cfg)
    assert e.status == "filled" and e.bar_pos == 0, e
    return bars, e, simulate_exit(bars, e, direction=LONG, stop=stop, target=target,
                                  cost=cost(), cfg=cfg or ecfg())


# ---------------------------------------------------------------------------
# entry: trade-through, never a touch
# ---------------------------------------------------------------------------

def test_a_touch_does_not_fill_a_long_entry():
    e = enter(mbars([(100.50, 100.50, 100.00, 100.25)] + [FLAT] * 14))
    assert e.status == "expired" and e.fill is None, e


def test_one_tick_through_fills_a_long_at_its_limit():
    e = enter(mbars([FLAT, (100.25, 100.50, 99.75, 100.25)]))
    assert e.status == "filled" and e.bar_pos == 1
    assert e.fill.fill_price == 100.00 and e.fill.order_price == 100.00
    assert e.fill.kind == "entry_limit" and e.fill.side == +1


def test_a_short_entry_needs_a_trade_above_its_limit():
    flat_below = (99.50, 99.75, 99.25, 99.50)      # FLAT trades above a 100.00 offer
    touch = enter(mbars([(99.50, 100.00, 99.50, 99.75)] + [flat_below] * 14),
                  direction=SHORT, limit=100.00, stop=101.00)
    through = enter(mbars([(99.75, 100.25, 99.50, 99.75)]),
                    direction=SHORT, limit=100.00, stop=101.00)
    assert touch.status == "expired"
    assert through.status == "filled" and through.fill.side == -1


def test_trade_through_is_judged_in_ticks_not_floats():
    """70.07 - 0.01 is 70.05999999999999 in floating point, so a naive
    `low <= limit - tick` says a bar trading at 70.06 never went through a
    70.07 bid. MCL/MGC/SIL have exactly these ticks."""
    mcl = {"symbol": "TST", "contract_spec": {"tick_size": 0.01, "tick_value": 1.0,
                                              "point_value": 100.0}}
    bars = mbars([(70.10, 70.10, 70.06, 70.08)])
    e = simulate_entry(bars, direction=LONG, limit=70.07, stop=69.50, live_from=T0,
                       live_until=T0 + 15 * MIN, trade_date=D0, contracts=1,
                       cost=cost(mcl), cfg=ecfg())
    assert e.status == "filled", e


def test_a_gap_through_the_limit_fills_at_the_limit_not_the_open():
    e = enter(mbars([(99.50, 99.75, 99.25, 99.50)]))
    assert e.fill.fill_price == 100.00


def test_an_entry_price_off_the_tick_grid_is_refused():
    with pytest.raises(ValueError, match="tick grid"):
        enter(mbars([FLAT]), limit=100.10)


def test_a_stop_not_beyond_the_entry_is_refused():
    with pytest.raises(ValueError, match="beyond"):
        enter(mbars([FLAT]), stop=100.00)


# ---------------------------------------------------------------------------
# entry: when the order is live
# ---------------------------------------------------------------------------

def test_the_decision_bar_cannot_fill_its_own_order():
    """The order goes in at the decision bar's CLOSE. Its minutes trade through
    the limit, but they happened before the order existed."""
    decision_minutes = [(100.25, 100.50, 99.50, 100.25)] * 5
    bars = mbars(decision_minutes + [FLAT] * 15, start=T0 - 5 * MIN)
    e = enter(bars)
    assert e.status == "expired", e


def entry_frame(n):
    ts = [T0 - 5 * pd.Timedelta(minutes=5) + i * pd.Timedelta(minutes=5) for i in range(n)]
    return pd.DataFrame({"ts": ts, "trade_date": [D0] * n})


def test_entry_window_spans_exactly_k_entry_bars_after_the_decision_bar():
    frame = entry_frame(12)
    d = 4                                            # frame ts[4] = T0 - 5min
    live_from, live_until = entry_window(frame, d, ecfg(), "5min")
    assert live_from == T0                           # the decision bar's close
    assert live_until == T0 + 15 * MIN               # close of the 3rd bar after it


def test_last_minute_of_the_window_fills_and_the_next_does_not():
    through = (100.25, 100.50, 99.75, 100.25)
    live_from, live_until = entry_window(entry_frame(12), 4, ecfg(), "5min")
    last = enter(mbars([FLAT] * 14 + [through]), until=live_until)
    late = enter(mbars([FLAT] * 15 + [through]), until=live_until)
    assert last.status == "filled" and last.bar_pos == 14
    assert late.status == "expired"


def test_entry_window_counts_bars_that_exist_not_clock_time():
    """A missing 5-minute bar (no trades) must not shorten the window."""
    frame = entry_frame(12).drop(index=6).reset_index(drop=True)
    _, live_until = entry_window(frame, 4, ecfg(), "5min")
    assert live_until == T0 + 20 * MIN


def test_data_ending_inside_the_window_is_unresolved_not_expired():
    """The window's end is known only if the entry frame holds all K bars."""
    live_from, live_until = entry_window(entry_frame(6), 4, ecfg(), "5min")
    assert live_until is None
    e = enter(mbars([FLAT] * 3), until=pd.NaT)
    assert e.status == "unresolved" and e.fill is None


def test_a_pending_entry_is_cancelled_at_the_day_boundary():
    d1 = date(2026, 3, 3)
    through_next_day = (100.25, 100.50, 99.75, 100.25, d1)
    e = enter(mbars([FLAT, FLAT, through_next_day]))
    assert e.status == "cancelled_day_boundary" and e.fill is None


def test_invalidation_before_fill_cancels_the_order():
    """Only reachable when the trade-through requirement is wider than the
    entry-to-stop distance: with 1 tick, a close beyond the stop means price
    already went through the limit. Checked here with a 6-tick requirement:
    the bar trades down to 99.50, only 2 ticks through 100.00, and closes
    below the 99.75 stop."""
    cfg = ecfg(entry__trade_through_ticks=6)
    e = enter(mbars([(100.25, 100.25, 99.50, 99.50)] + [FLAT] * 5), stop=99.75, cfg=cfg)
    assert e.status == "cancelled_invalidated", e


def test_a_bar_that_fills_then_closes_beyond_the_stop_is_a_filled_loser():
    """The naive order (check the close first, cancel) would erase a real
    loss: the path went through the limit before it reached the stop."""
    bars, e, x = long_position([], fill_row=(100.25, 100.25, 98.75, 98.75))
    assert e.status == "filled"
    assert x.status == "stop" and x.bar_pos == 0
    assert x.fill.fill_price == 98.75                 # the stop, 99.00, less 1 tick


# ---------------------------------------------------------------------------
# exits
# ---------------------------------------------------------------------------

def test_target_needs_a_trade_through_and_fills_at_its_price():
    _, _, touch = long_position([(101.50, 102.00, 101.50, 101.75)])
    _, _, through = long_position([(101.50, 102.25, 101.50, 101.75)])
    assert touch.status == "unresolved"
    assert through.status == "target" and through.fill.fill_price == 102.00
    assert through.fill.kind == "target_limit"


def test_a_stop_triggers_on_a_touch_and_slips_one_tick():
    _, _, x = long_position([(99.50, 99.75, 99.00, 99.25)])
    assert x.status == "stop" and x.fill.fill_price == 98.75
    assert x.fill.reference_price == 99.00 and x.fill.kind == "stop_market_exit"


def test_a_stop_gapped_through_fills_at_the_open_less_slippage():
    _, _, x = long_position([FLAT, (98.00, 98.50, 97.75, 98.25)])
    assert x.status == "stop" and x.fill.fill_price == 97.75


def test_a_bar_reaching_stop_and_target_is_a_stop():
    _, _, x = long_position([(100.50, 102.50, 98.75, 101.00)])
    assert x.status == "stop" and x.bar_pos == 1


def test_the_fill_bar_cannot_also_reach_the_target():
    """The bar traded 102.25 and 99.75, order unknown: the high may have come
    before the fill. The next bar is flat, so the trade stays open."""
    _, _, x = long_position([FLAT], fill_row=(101.00, 102.50, 99.75, 101.00))
    assert x.status == "unresolved", x


def test_a_short_stop_gapped_through_fills_above_the_open():
    bars = mbars([(100.25, 100.25, 99.75, 100.00), (101.50, 102.00, 101.25, 101.75)])
    e = enter(bars, direction=SHORT, limit=100.00, stop=101.00)
    x = simulate_exit(bars, e, direction=SHORT, stop=101.00, target=98.00,
                      cost=cost(), cfg=ecfg())
    assert x.status == "stop" and x.fill.fill_price == 101.75 and x.fill.side == +1


def test_an_open_position_is_flattened_on_the_last_bar_of_its_trade_date():
    d1 = date(2026, 3, 3)
    _, _, x = long_position([FLAT, (100.75, 101.00, 100.50, 101.00),
                             (101.00, 101.25, 100.75, 101.00, d1)])
    assert x.status == "day_boundary" and x.bar_pos == 2
    assert x.fill.kind == "day_boundary_flatten"
    assert x.fill.reference_price == 101.00 and x.fill.fill_price == 100.75


def test_data_ending_with_a_position_open_is_unresolved():
    _, _, x = long_position([FLAT, FLAT])
    assert x.status == "unresolved" and x.fill is None


def test_a_contract_change_inside_a_trade_raises():
    bars = pd.concat([mbars([(100.25, 100.50, 99.75, 100.25)]),
                      mbars([FLAT], start=T0 + MIN, raw="MESM6")], ignore_index=True)
    e = enter(bars)
    with pytest.raises(ValueError, match="contract"):
        simulate_exit(bars, e, direction=LONG, stop=99.00, target=102.00,
                      cost=cost(), cfg=ecfg())


def test_stop_and_target_round_to_the_grid_conservatively():
    """Stops round away from entry (never tighter than Part 0), targets toward
    it (never more reward than the level gives)."""
    _, _, x = long_position([(99.50, 99.75, 99.00, 99.25)], stop=99.10, target=102.10)
    assert x.status == "stop" and x.fill.reference_price == 99.00
    _, _, y = long_position([(101.50, 102.25, 101.50, 101.75)], stop=99.10, target=102.10)
    assert y.status == "target" and y.fill.fill_price == 102.00


def test_rounding_ignores_float_noise_that_is_already_on_the_grid():
    """On a 0.01 tick, 70.00 - 0.07 divides to 6993.000000000001 ticks and
    70.00 - 0.01 to 6998.999999999999. A naive ceil/floor pushes those stops a
    whole tick farther out, to 69.94 and 69.98. (Checked: the quotients
    really carry the noise; a fixture whose quotient comes out exact tests
    nothing.)"""
    mcl = {"symbol": "TST", "contract_spec": {"tick_size": 0.01, "tick_value": 1.0,
                                              "point_value": 100.0}}
    c = cost(mcl)
    up, down = 70.00 - 0.07, 70.00 - 0.01
    assert up / 0.01 != round(up / 0.01) and down / 0.01 != round(down / 0.01)
    assert c.round_stop(up, SHORT, ecfg()) == pytest.approx(69.93)
    assert c.round_stop(down, LONG, ecfg()) == pytest.approx(69.99)


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

def test_only_market_orders_slip():
    c = cost()
    assert c.slippage("entry_limit") == 0 and c.slippage("target_limit") == 0
    for kind in ("stop_market_exit", "breaker_flatten", "day_boundary_flatten"):
        assert c.slippage(kind) == 0.25, kind


def test_breaker_flatten_is_a_market_order_that_slips():
    f = market_exit(T0, 101.00, LONG, 1, "breaker_flatten", cost())
    assert f.side == -1 and f.fill_price == 100.75 and f.reference_price == 101.00


def test_fees_are_charged_on_entry_and_exit_per_contract():
    _, e, x = long_position([(101.50, 102.25, 101.50, 101.75)])
    rt = round_trip(e.fill, x.fill, direction=LONG, stop=99.00, cost=cost())
    assert rt["gross_usd"] == pytest.approx(10.00)            # 2.00 points x $5
    assert rt["fees_usd"] == pytest.approx(1.50)              # 2 sides x $0.75
    assert rt["net_usd"] == pytest.approx(8.50)
    assert rt["risk_usd"] == pytest.approx(5.00)              # 1.00 point to the stop
    assert rt["r_net"] == pytest.approx(1.70)
    assert rt["r_gross"] == pytest.approx(2.00)


def test_a_stopped_trade_loses_more_than_one_r_after_slippage_and_fees():
    _, e, x = long_position([(99.50, 99.75, 99.00, 99.25)])
    rt = round_trip(e.fill, x.fill, direction=LONG, stop=99.00, cost=cost())
    assert rt["gross_usd"] == pytest.approx(-6.25)            # 1.25 points x $5
    assert rt["net_usd"] == pytest.approx(-7.75)
    assert rt["r_net"] == pytest.approx(-1.55)


def test_an_undefined_thin_session_multiplier_is_refused():
    """Which sessions count as thin is undefined, so a multiplier other than 1
    has nothing to apply to yet. Refuse it rather than apply it everywhere."""
    costs = copy.deepcopy(COSTS_CFG)
    costs["slippage"]["thin_session_multiplier"] = 2.0
    with pytest.raises(ValueError, match="thin"):
        CostModel.from_config(costs, SYMBOL_CFG)


def test_unknown_policies_are_refused_not_defaulted():
    cfg = copy.deepcopy(EXEC_CFG)
    cfg["fills"]["same_bar_stop_and_target"] = "target"
    with pytest.raises(ValueError, match="same_bar_stop_and_target"):
        ExecConfig.from_config(cfg)


# ---------------------------------------------------------------------------
# the real config files
# ---------------------------------------------------------------------------

def _yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_real_execution_config_holds_the_signed_off_decisions():
    cfg = ExecConfig.from_config(_yaml("config/execution.yaml"))
    assert cfg.k_entry_expire_bars == 3
    assert cfg.entry_through_ticks == 1 and cfg.target_through_ticks == 1


@pytest.mark.parametrize("sym", ["MES", "MNQ", "MYM", "MCL", "MGC", "SIL", "MET", "MBT"])
def test_real_costs_build_for_every_instrument(sym):
    scfg = _yaml(f"config/symbols/{sym}.yaml")
    c = CostModel.from_config(_yaml("config/costs.yaml"), scfg)
    spec = scfg["contract_spec"]
    assert c.point_value == pytest.approx(spec["tick_value"] / spec["tick_size"])
    assert c.slippage("stop_market_exit") == pytest.approx(spec["tick_size"])
    assert c.slippage("entry_limit") == 0
    assert c.fee_per_side > 0
