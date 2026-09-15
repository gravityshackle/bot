"""S17 two-stage close confirmation.

The property that matters is that a Yellow Alert always resolves exactly once,
into exactly one of confirmed / failed / expired. Any path that lets an alert
resolve twice, or never, leaks setups into or out of the engine silently.
"""
from __future__ import annotations

import pandas as pd
import pytest

from features import confirmation_signal as cs
from features.schema import load_params

CT = "America/Chicago"
P = load_params("MES")          # k_confirm_bars = 4


def mk(rows, start="2026-03-02 09:00"):
    ts = pd.date_range(pd.Timestamp(start), periods=len(rows), freq="5min",
                       tz=CT).tz_convert("UTC")
    return pd.DataFrame({
        "ts": ts, "raw_symbol": "MESM6",
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [100] * len(rows),
    })


# --------------------------------------------------------------------------
# mode exclusivity
# --------------------------------------------------------------------------

def test_default_mode_is_buffer():
    assert cs.assert_single_mode(P) == "buffer"


def test_invoking_s17_under_buffer_mode_raises():
    """S4 and S17 must never both apply to one setup."""
    with pytest.raises(cs.ModeConflict, match="alternative to S4"):
        cs.require_mode(P, "confirmation_signal")


def test_invoking_s4_under_buffer_mode_is_fine():
    cs.require_mode(P, "buffer")          # no raise


def test_unknown_mode_rejected():
    bad = load_params("MES")
    bad.values["breakout"]["mode"] = "both"
    with pytest.raises(ValueError, match="buffer\\|confirmation_signal"):
        cs.assert_single_mode(bad)


# --------------------------------------------------------------------------
# Yellow Alert -> Red Alert
# --------------------------------------------------------------------------

def test_confirmation_measures_against_the_piercing_bars_high_not_the_level():
    """This is what makes S17 stricter than S4's buffer method."""
    rows = [
        (99, 99.5, 98.5, 99.0),        # 0 below
        (99.5, 102.0, 99.4, 100.5),    # 1 PIERCE: high 102, closes above level
        (100.5, 101.0, 100.2, 100.9),  # 2 above the LEVEL but below high[P]
        (101, 103.0, 100.8, 102.5),    # 3 closes above high[P]=102 -> confirmed
    ]
    out = cs.scan(mk(rows), 100.0, P)
    up = out[out["direction"] == cs.LONG].iloc[0]
    assert up["pierce_idx"] == 1
    assert up["pierce_extreme"] == pytest.approx(102.0)
    assert up["outcome"] == "confirmed"
    assert up["resolve_idx"] == 3
    # bar 2 cleared the level but not the piercing extreme -- correctly no
    # confirmation there
    assert up["bars_to_resolve"] == 2


def test_close_back_through_the_level_is_a_failure_not_an_expiry():
    rows = [
        (99, 99.5, 98.5, 99.0),
        (99.5, 102.0, 99.4, 100.5),    # pierce
        (100.5, 101.0, 98.0, 98.5),    # closes back below the level
    ]
    out = cs.scan(mk(rows), 100.0, P)
    up = out[out["direction"] == cs.LONG].iloc[0]
    assert up["outcome"] == "failed"
    assert up["resolve_idx"] == 2


def test_no_resolution_within_k_bars_expires():
    rows = [(99, 99.5, 98.5, 99.0), (99.5, 102.0, 99.4, 100.5)]
    rows += [(100.5, 101.0, 100.2, 100.6)] * 6     # drifts, never confirms
    out = cs.scan(mk(rows), 100.0, P)
    up = out[out["direction"] == cs.LONG].iloc[0]
    assert up["outcome"] == "expired"
    assert up["bars_to_resolve"] == 4              # k_confirm_bars


def test_every_alert_resolves_exactly_once():
    rows = [(99, 99.5, 98.5, 99.0)]
    rows += [(99.5, 102.0, 99.4, 100.5), (100.5, 103.0, 100.2, 102.9)]
    rows += [(102, 102.5, 97.0, 97.5), (97, 97.5, 96.0, 96.5)]
    rows += [(96, 101.0, 95.5, 100.8)]
    out = cs.scan(mk(rows), 100.0, P)
    assert not out.empty
    # one row per (pierce_idx, direction), no duplicates
    assert not out.duplicated(["pierce_idx", "direction"]).any()
    assert set(out["outcome"]) <= {"confirmed", "failed", "expired", "pending"}


def test_a_second_pierce_does_not_reset_the_reference_bar():
    """Re-arming on each poke would keep moving the bar confirmation is
    measured against, making S17 no stricter than S4."""
    rows = [
        (99, 99.5, 98.5, 99.0),
        (99.5, 103.0, 99.4, 100.2),    # pierce, high 103
        (100.2, 101.0, 100.1, 100.3),  # pokes again, lower high
        (100.3, 101.5, 100.2, 100.4),
    ]
    out = cs.scan(mk(rows), 100.0, P)
    up = out[out["direction"] == cs.LONG].iloc[0]
    assert up["pierce_idx"] == 1
    assert up["pierce_extreme"] == pytest.approx(103.0)


def test_breakdown_side_is_symmetric():
    rows = [
        (101, 101.5, 100.5, 101.0),
        (100.5, 100.6, 98.0, 99.5),    # pierce down, low 98
        (99.5, 99.8, 97.0, 97.5),      # closes below low[P] -> confirmed
    ]
    out = cs.scan(mk(rows), 100.0, P)
    dn = out[out["direction"] == cs.SHORT].iloc[0]
    assert dn["pierce_extreme"] == pytest.approx(98.0)
    assert dn["outcome"] == "confirmed"


def test_unresolved_alert_at_the_end_is_pending_not_silently_dropped():
    rows = [(99, 99.5, 98.5, 99.0), (99.5, 102.0, 99.4, 100.5)]
    out = cs.scan(mk(rows), 100.0, P)
    assert (out["outcome"] == "pending").any()


def test_confirmations_returns_only_confirmed():
    rows = [
        (99, 99.5, 98.5, 99.0),
        (99.5, 102.0, 99.4, 100.5),
        (101, 103.0, 100.8, 102.5),
    ]
    out = cs.confirmations(mk(rows), 100.0, P)
    assert len(out) == 1 and (out["outcome"] == "confirmed").all()


# --------------------------------------------------------------------------
# scoring input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bars,expected", [
    (1, 1.0),        # immediate confirmation caps at 1.0
    (4, 1.0),        # exactly K
    (8, 0.5),        # twice K -> half
])
def test_magnitude_factor_rewards_fast_confirmation(bars, expected):
    assert cs.magnitude_factor(bars, P) == pytest.approx(expected)


def test_magnitude_factor_is_bounded():
    assert cs.magnitude_factor(0, P) == 1.0
    assert 0.0 < cs.magnitude_factor(100, P) <= 1.0


# --------------------------------------------------------------------------
# a Yellow Alert is a CROSSING, not "being beyond"
# --------------------------------------------------------------------------

def test_price_already_above_the_level_does_not_re_arm_every_bar():
    """S17 says the pierce is the first bar to CROSS the level.

    Read as "high > level", a market trading above a level re-arms an alert on
    every bar as soon as the previous one resolves. On real MES 5m data that
    produced ~3,450 confirmations against one static level -- one every five
    bars.
    """
    rows = [(99, 99.5, 98.5, 99.0)]                 # below
    rows += [(105 + i, 106 + i, 104 + i, 105.5 + i) for i in range(20)]
    out = cs.scan(mk(rows), 100.0, P)
    # exactly one crossing happened, so at most one LONG alert exists
    assert (out["direction"] == cs.LONG).sum() == 1


def test_recrossing_arms_a_new_alert():
    rows = [
        (99, 99.5, 98.5, 99.0),        # below
        (99.5, 102.0, 99.4, 101.0),    # crosses up -> alert 1
        (101, 103.0, 100.8, 102.5),    # confirms alert 1
        (102, 102.5, 97.0, 98.0),      # back below the level
        (98, 101.5, 97.5, 101.0),      # crosses up again -> alert 2
        (101, 104.0, 100.9, 103.5),    # confirms alert 2
    ]
    out = cs.scan(mk(rows), 100.0, P)
    longs = out[out["direction"] == cs.LONG]
    assert len(longs) == 2
    assert list(longs["pierce_idx"]) == [1, 4]


def test_first_bar_cannot_be_a_pierce():
    """With no prior bar there is no crossing to observe."""
    out = cs.scan(mk([(105, 106, 104, 105.5), (105, 106, 104, 105.5)]), 100.0, P)
    assert out.empty or (out["pierce_idx"] > 0).all()
