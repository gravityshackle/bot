"""Roll-logic tests on synthetic data.

These matter more than most tests in the repo: a wrong roll silently corrupts
every feature computed downstream, and it does so in a way that looks
plausible on a chart. Everything here is deterministic and offline -- no
Databento credits are spent to run it.

The central invariant under test is the approved Phase 1 design: the stitched
series holds REAL, UNADJUSTED prices, so a seam at a roll boundary is correct
output rather than a defect. Adjustment is derived on demand and must never
mutate the stored bars.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.continuous_contract import (
    ContinuousSeries,
    RolloverConfig,
    build_roll_map,
    daily_volume_by_contract,
    stitch,
    trade_date,
)
from data.sources.base import ContractMeta

CT = "America/Chicago"
BARS_PER_DAY = 4


def _meta(raw: str, root: str, code: str, year: int, expiry: str) -> ContractMeta:
    return ContractMeta(raw_symbol=raw, root=root, month_code=code, year=year,
                        expiration=pd.Timestamp(expiry, tz="UTC"))


def make_bars(spec: dict[str, dict], start="2026-03-02", n_days=10) -> pd.DataFrame:
    """Build canonical bars for several contracts over the same date range.

    spec: {raw_symbol: {"price": float, "volume": callable(day_idx) -> int}}
    Bars are placed at 15:00-18:00 UTC (mid-CT-session), safely inside the
    trade date so that trade_date == calendar date and the roll tests are not
    entangled with boundary handling (covered separately below).
    """
    dates = pd.bdate_range(start, periods=n_days, tz="UTC")
    rows = []
    for raw, cfg in spec.items():
        px = cfg["price"]
        for di, d in enumerate(dates):
            vol = cfg["volume"](di)
            for b in range(BARS_PER_DAY):
                ts = d + pd.Timedelta(hours=15 + b)
                o = px + 0.25 * b
                rows.append({
                    "ts": ts, "raw_symbol": raw,
                    "open": o, "high": o + 0.5, "low": o - 0.5, "close": o + 0.25,
                    "volume": max(vol // BARS_PER_DAY, 0),
                })
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["trade_date"] = trade_date(df["ts"], CT, "17:00")
    return df.sort_values(["ts", "raw_symbol"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# trade date boundary
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "local_time,boundary,expect_next_day",
    [
        ("16:59", "17:00", False),   # just before Globex open -> same trade date
        ("17:00", "17:00", True),    # at open -> next trade date
        ("18:30", "17:00", True),
        ("09:00", "17:00", False),
        ("15:59", "16:00", False),   # MET: maintenance halt boundary
        ("16:00", "16:00", True),
        ("23:00", "16:00", True),
    ],
)
def test_trade_date_boundary(local_time, boundary, expect_next_day):
    day = pd.Timestamp("2026-03-10")
    ts = pd.Series(pd.to_datetime([f"2026-03-10 {local_time}"])
                   .tz_localize(CT).tz_convert("UTC"))
    got = trade_date(ts, CT, boundary).iloc[0]
    expected = (day + pd.Timedelta(days=1)).date() if expect_next_day else day.date()
    assert got == expected


def test_met_and_index_boundaries_differ_for_the_same_bar():
    """A 16:30 CT bar is 'today' for MES (17:00 cut) but 'tomorrow' for MET."""
    ts = pd.Series(pd.to_datetime(["2026-03-10 16:30"])
                   .tz_localize(CT).tz_convert("UTC"))
    assert trade_date(ts, CT, "17:00").iloc[0] == pd.Timestamp("2026-03-10").date()
    assert trade_date(ts, CT, "16:00").iloc[0] == pd.Timestamp("2026-03-11").date()


# --------------------------------------------------------------------------
# volume crossover
# --------------------------------------------------------------------------

def _two_contract_crossover_bars(cross_at=4):
    return make_bars({
        "MESH6": {"price": 5000.0,
                  "volume": lambda i: 100_000 if i < cross_at else 20_000},
        "MESM6": {"price": 5010.0,
                  "volume": lambda i: 10_000 if i < cross_at else 90_000},
    })


def test_volume_crossover_rolls_after_confirm_days():
    bars = _two_contract_crossover_bars(cross_at=4)
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-03-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-06-19")}
    cfg = RolloverConfig(confirm_days=2, calendar_backstop_days=8)

    roll_map, warns = build_roll_map(bars, metas, cfg)
    dates = sorted(bars["trade_date"].unique())

    assert len(roll_map) == 1, f"expected exactly one roll, got:\n{roll_map}"
    row = roll_map.iloc[0]
    assert (row.from_symbol, row.to_symbol) == ("MESH6", "MESM6")
    assert row.reason == "volume_crossover"
    # crossover begins on index 4; with confirm_days=2 the decision completes on
    # index 5 and takes effect on index 6
    assert row.effective_date == dates[6]
    assert warns == []


def test_confirm_days_requires_a_streak_not_a_single_day():
    """One isolated day of higher volume must not roll when confirm_days=2."""
    bars = make_bars({
        "MESH6": {"price": 5000.0, "volume": lambda i: 20_000 if i == 3 else 100_000},
        "MESM6": {"price": 5010.0, "volume": lambda i: 50_000 if i == 3 else 10_000},
    })
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-06-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-09-19")}
    roll_map, _ = build_roll_map(bars, metas,
                                 RolloverConfig(confirm_days=2, calendar_backstop_days=1))
    assert roll_map.empty, f"single-day blip should not roll:\n{roll_map}"


def test_roll_takes_effect_after_the_deciding_session_no_lookahead():
    """The deciding session's own bars must still come from the OLD contract."""
    bars = _two_contract_crossover_bars(cross_at=4)
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-03-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-06-19")}
    roll_map, _ = build_roll_map(bars, metas, RolloverConfig(confirm_days=2))
    series = stitch(bars, roll_map, symbol="MES")

    eff = roll_map.iloc[0].effective_date
    before = series.bars[series.bars["trade_date"] < eff]["raw_symbol"].unique()
    on_after = series.bars[series.bars["trade_date"] >= eff]["raw_symbol"].unique()
    assert list(before) == ["MESH6"]
    assert list(on_after) == ["MESM6"]


# --------------------------------------------------------------------------
# calendar backstop
# --------------------------------------------------------------------------

def test_calendar_backstop_fires_when_crossover_never_does():
    """Thin successor volume must not let the front ride into expiry."""
    bars = make_bars({
        "MCLF6": {"price": 70.00, "volume": lambda i: 100_000},
        "MCLG6": {"price": 70.50, "volume": lambda i: 1_000},   # never crosses
    })
    dates = sorted(bars["trade_date"].unique())
    expiry = dates[5]
    metas = {"MCLF6": _meta("MCLF6", "MCL", "F", 2026, str(expiry)),
             "MCLG6": _meta("MCLG6", "MCL", "G", 2026, "2026-04-20")}
    cfg = RolloverConfig(confirm_days=2, calendar_backstop_days=2)

    roll_map, warns = build_roll_map(bars, metas, cfg)
    assert len(roll_map) == 1
    row = roll_map.iloc[0]
    assert row.reason.startswith("calendar_backstop")
    assert (row.from_symbol, row.to_symbol) == ("MCLF6", "MCLG6")
    # Margin is measured at the date the roll TAKES EFFECT. Mar-6 (Fri) is
    # 3 calendar days out, but the next session (Mon Mar-9) is 0 days out,
    # so the backstop must fire then rather than after expiry has passed.
    assert row.effective_date == dates[5]
    assert any("calendar backstop" in w for w in warns), warns
    # the weekend compressed the achievable margin below the configured 2d
    assert any("weekend/holiday gap" in w for w in warns), warns


def test_backstop_warns_and_truncates_with_no_successor():
    bars = make_bars({"MCLF6": {"price": 70.0, "volume": lambda i: 100_000}})
    dates = sorted(bars["trade_date"].unique())
    metas = {"MCLF6": _meta("MCLF6", "MCL", "F", 2026, str(dates[4]))}
    roll_map, warns = build_roll_map(bars, metas,
                                     RolloverConfig(confirm_days=2, calendar_backstop_days=2))
    assert roll_map.empty
    assert any("no\nsuccessor" in w or "successor" in w for w in warns), warns


# --------------------------------------------------------------------------
# stitching: coverage, exclusivity, unadjusted prices
# --------------------------------------------------------------------------

def test_stitch_has_exactly_one_active_contract_per_trade_date():
    bars = _two_contract_crossover_bars()
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-03-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-06-19")}
    roll_map, _ = build_roll_map(bars, metas, RolloverConfig(confirm_days=2))
    series = stitch(bars, roll_map)

    per_day = series.bars.groupby("trade_date")["raw_symbol"].nunique()
    assert (per_day == 1).all(), f"overlapping contracts on:\n{per_day[per_day != 1]}"

    # no dropped sessions and no duplicated timestamps
    assert set(series.bars["trade_date"]) == set(bars["trade_date"])
    assert not series.bars["ts"].duplicated().any()
    assert series.bars["ts"].is_monotonic_increasing


def test_stitch_preserves_real_unadjusted_prices():
    """The core of the approved design: stitched prices are untouched originals."""
    bars = _two_contract_crossover_bars()
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-03-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-06-19")}
    roll_map, _ = build_roll_map(bars, metas, RolloverConfig(confirm_days=2))
    series = stitch(bars, roll_map)

    merged = series.bars.merge(
        bars, on=["ts", "raw_symbol"], suffixes=("", "_orig"), how="left")
    for col in ["open", "high", "low", "close"]:
        pd.testing.assert_series_equal(
            merged[col], merged[f"{col}_orig"], check_names=False)


def test_unadjusted_series_shows_a_seam_at_the_roll():
    """A visible seam is CORRECT for an unadjusted stitch, not a bug."""
    bars = _two_contract_crossover_bars()
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-03-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-06-19")}
    roll_map, _ = build_roll_map(bars, metas, RolloverConfig(confirm_days=2))
    series = stitch(bars, roll_map)

    eff = roll_map.iloc[0].effective_date
    last_before = series.bars[series.bars["trade_date"] < eff]["close"].iloc[-1]
    first_after = series.bars[series.bars["trade_date"] >= eff]["close"].iloc[0]
    seam = first_after - last_before
    assert abs(seam) > 1.0, "expected a real price discontinuity at the roll"


def test_recorded_offset_matches_the_observed_seam():
    """offset_difference must equal the actual gap it is meant to correct."""
    bars = _two_contract_crossover_bars()
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-03-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-06-19")}
    roll_map, _ = build_roll_map(bars, metas, RolloverConfig(confirm_days=2))
    row = roll_map.iloc[0]

    # both contracts' closes on the deciding session (the day before effective)
    dates = sorted(bars["trade_date"].unique())
    decide = dates[dates.index(row.effective_date) - 1]
    day = bars[bars["trade_date"] == decide].sort_values("ts")
    c_from = day[day["raw_symbol"] == row.from_symbol]["close"].iloc[-1]
    c_to = day[day["raw_symbol"] == row.to_symbol]["close"].iloc[-1]

    # Sign convention: the offset IS the seam, new-minus-old, so that adding
    # it to pre-roll bars lifts history onto the incoming contract's level.
    assert row.offset_difference == pytest.approx(c_to - c_from)
    assert row.offset_ratio == pytest.approx(c_to / c_from)
    # and it must actually close the gap, not widen it
    assert abs((c_from + row.offset_difference) - c_to) < 1e-9


# --------------------------------------------------------------------------
# on-demand adjustment
# --------------------------------------------------------------------------

def _series_with_one_roll() -> ContinuousSeries:
    bars = _two_contract_crossover_bars()
    metas = {"MESH6": _meta("MESH6", "MES", "H", 2026, "2026-03-20"),
             "MESM6": _meta("MESM6", "MES", "M", 2026, "2026-06-19")}
    roll_map, warns = build_roll_map(bars, metas, RolloverConfig(confirm_days=2))
    return stitch(bars, roll_map, symbol="MES", warnings=warns)


@pytest.mark.parametrize("method", ["difference", "ratio"])
def test_adjustment_removes_the_seam(method):
    series = _series_with_one_roll()
    adj = series.to_adjusted(method)
    eff = series.roll_map.iloc[0].effective_date

    before = adj[adj["trade_date"] < eff]
    after = adj[adj["trade_date"] >= eff]
    # compare like-for-like: last bar of the pre-roll session vs the same
    # intraday slot after adjustment
    seam = after["close"].iloc[0] - before["close"].iloc[-1]
    assert abs(seam) < 1.0, f"{method} adjustment left a seam of {seam}"


def test_to_adjusted_never_mutates_stored_bars():
    series = _series_with_one_roll()
    snapshot = series.bars.copy(deep=True)
    series.to_adjusted("difference")
    series.to_adjusted("ratio")
    pd.testing.assert_frame_equal(series.bars, snapshot)


def test_adjusted_most_recent_contract_keeps_true_prices():
    """Back-adjustment shifts history, never the live contract."""
    series = _series_with_one_roll()
    eff = series.roll_map.iloc[0].effective_date
    adj = series.to_adjusted("difference")
    live_raw = series.bars[series.bars["trade_date"] >= eff]["close"].reset_index(drop=True)
    live_adj = adj[adj["trade_date"] >= eff]["close"].reset_index(drop=True)
    pd.testing.assert_series_equal(live_raw, live_adj)


def test_unknown_adjustment_method_rejected():
    with pytest.raises(ValueError, match="unknown adjustment method"):
        _series_with_one_roll().to_adjusted("panama")


# --------------------------------------------------------------------------
# MET: monthly near months + quarterly further out
# --------------------------------------------------------------------------

def test_met_next_chronological_does_not_skip_a_thin_month():
    """MET lists 6 near monthlies plus further-out quarterlies simultaneously,
    so a quarterly can out-volume a nearby monthly. next_chronological (the
    shared default across all seven symbols) must still advance one listing at
    a time rather than jumping to the fat quarterly."""
    bars = make_bars({
        "METU6": {"price": 4000.0, "volume": lambda i: 50_000 if i < 4 else 5_000},
        "METV6": {"price": 4010.0, "volume": lambda i: 1_000 if i < 4 else 9_000},
        "METZ6": {"price": 4030.0, "volume": lambda i: 8_000 if i < 4 else 80_000},
    })
    metas = {
        "METU6": _meta("METU6", "MET", "U", 2026, "2026-09-25"),
        "METV6": _meta("METV6", "MET", "V", 2026, "2026-10-30"),
        "METZ6": _meta("METZ6", "MET", "Z", 2026, "2026-12-24"),
    }
    roll_map, _ = build_roll_map(
        bars, metas,
        RolloverConfig(confirm_days=1, calendar_backstop_days=1,
                       candidate="next_chronological"))
    assert not roll_map.empty
    first = roll_map.iloc[0]
    assert first.from_symbol == "METU6"
    assert first.to_symbol == "METV6", (
        "next_chronological must step to the adjacent monthly, not skip to the "
        f"higher-volume quarterly; got {first.to_symbol}")


def test_highest_volume_candidate_may_skip_a_thin_month():
    """The alternative mode, kept for comparison in backtest."""
    bars = make_bars({
        "METU6": {"price": 4000.0, "volume": lambda i: 50_000 if i < 4 else 5_000},
        "METV6": {"price": 4010.0, "volume": lambda i: 1_000},
        "METZ6": {"price": 4030.0, "volume": lambda i: 8_000 if i < 4 else 80_000},
    })
    metas = {
        "METU6": _meta("METU6", "MET", "U", 2026, "2026-09-25"),
        "METV6": _meta("METV6", "MET", "V", 2026, "2026-10-30"),
        "METZ6": _meta("METZ6", "MET", "Z", 2026, "2026-12-24"),
    }
    roll_map, _ = build_roll_map(
        bars, metas,
        RolloverConfig(confirm_days=1, calendar_backstop_days=1,
                       candidate="highest_volume"))
    assert roll_map.iloc[0].to_symbol == "METZ6"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_daily_volume_matrix_shape():
    bars = _two_contract_crossover_bars()
    vol = daily_volume_by_contract(bars)
    assert set(vol.columns) == {"MESH6", "MESM6"}
    assert len(vol) == len(bars["trade_date"].unique())
    assert (vol >= 0).all().all()
