"""Continuous contract construction: unadjusted series + explicit roll map.

DESIGN (approved Phase 1 decision)
----------------------------------
The strategy trades price *levels* -- prior day H/L, gap zones, S/R. A
back-adjusted price is not a price that ever traded, so a level computed on
back-adjusted history is fictional. This module therefore keeps UNADJUSTED,
real per-contract prices as the source of truth and records the roll boundaries
alongside them. Adjustment factors are computed but applied only on demand, for
the narrow set of consumers that genuinely need a continuous price path
(long-lookback ATR means, HTF EMA) rather than real levels.

Consequence to expect in Phase 1 validation plots: roll dates appear as visible
price seams in the unadjusted series. That is CORRECT output, not a defect --
an unadjusted stitch is discontinuous by construction. What the plots check is
that seams occur only on roll dates, that each seam matches the recorded
offset, and that no seam appears anywhere else.

No lookahead: a roll is decided from completed sessions only and takes effect
at the START of the following session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.sources.base import ContractMeta

_LAST = 1 << 30


@dataclass(frozen=True)
class RolloverConfig:
    rule: str = "volume_crossover"
    confirm_days: int = 2
    calendar_backstop_days: int = 8
    candidate: str = "next_chronological"   # or "highest_volume"

    @classmethod
    def from_symbol_config(cls, d: dict) -> "RolloverConfig":
        r = d["rollover"]
        return cls(
            rule=r["rule"],
            confirm_days=int(r["confirm_days"]),
            calendar_backstop_days=int(r["calendar_backstop_days"]),
            candidate=r["candidate"],
        )


@dataclass
class ContinuousSeries:
    """Unadjusted stitched bars plus everything needed to reconstruct adjustment."""
    bars: pd.DataFrame                    # canonical schema + trade_date, is_roll_bar
    roll_map: pd.DataFrame                # one row per roll
    symbol: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_adjusted(self, method: str = "difference") -> pd.DataFrame:
        """Derive a back-adjusted price path ON DEMAND.

        Never persisted, and never used for level detection -- only for
        indicators that need a continuous path across rolls.
        """
        if method not in ("difference", "ratio"):
            raise ValueError("unknown adjustment method: " + repr(method))
        out = self.bars.copy()
        price_cols = ["open", "high", "low", "close"]

        # Walk rolls newest -> oldest, accumulating the correction applied to
        # everything BEFORE each roll, so the most recent contract keeps true
        # market prices.
        for row in self.roll_map.sort_values("effective_date", ascending=False).itertuples():
            mask = out["trade_date"] < row.effective_date
            if method == "difference":
                out.loc[mask, price_cols] = out.loc[mask, price_cols] + row.offset_difference
            else:
                out.loc[mask, price_cols] = out.loc[mask, price_cols] * row.offset_ratio
        return out


def trade_date(ts_utc: pd.Series, tz: str, boundary: str) -> pd.Series:
    """Map bar timestamps to the trade date they belong to.

    A session opening 17:00 CT Sunday is Monday's trade date. MET uses a 16:00
    CT boundary (the maintenance halt) instead, since crypto has no natural
    close -- see the MET note in the symbol spec.
    """
    local = ts_utc.dt.tz_convert(tz)
    hh, mm = (int(x) for x in boundary.split(":"))
    after = (local.dt.hour > hh) | ((local.dt.hour == hh) & (local.dt.minute >= mm))
    shifted = local.dt.normalize() + pd.to_timedelta(after.astype("int64"), unit="D")
    return shifted.dt.date


def daily_volume_by_contract(bars: pd.DataFrame) -> pd.DataFrame:
    """trade_date x raw_symbol volume matrix, used for the crossover test."""
    return (bars.groupby(["trade_date", "raw_symbol"], observed=True)["volume"]
                .sum()
                .unstack(fill_value=0)
                .sort_index())


def _empty_roll_map() -> pd.DataFrame:
    return pd.DataFrame(columns=["effective_date", "from_symbol", "to_symbol",
                                 "reason", "offset_difference", "offset_ratio"])


def _next_chronological(front, order_idx, available):
    later = [s for s in available if order_idx.get(s, -1) > order_idx.get(front, _LAST)]
    return min(later, key=lambda s: order_idx[s]) if later else None


def _pick_candidate(front, day_vol, order_idx, cfg):
    """Which contract may we roll INTO?

    next_chronological only ever advances one listing at a time.
    highest_volume may skip a thin intervening month -- relevant for MET, which
    lists 6 near monthlies alongside further-out quarterlies (see MET.yaml).
    """
    later = [s for s in day_vol.index
             if order_idx.get(s, -1) > order_idx.get(front, _LAST)]
    if not later:
        return None
    if cfg.candidate == "highest_volume":
        best = max(later, key=lambda s: day_vol[s])
        return best if day_vol[best] > 0 else None
    return min(later, key=lambda s: order_idx[s])


def _make_roll(effective_date, frm, to, reason, closes, decide_date):
    """Offsets from the last session both contracts traded before the switch.

    Sign convention: the offset IS the seam, measured new-minus-old, so that
    adding it to the pre-roll bars lifts history onto the incoming contract's
    price level. Getting this backwards does not error -- it silently doubles
    the gap it was meant to close -- so it is pinned by
    test_recorded_offset_matches_the_observed_seam.
    """
    diff, ratio = 0.0, 1.0
    try:
        c_from = closes.at[decide_date, frm]
        c_to = closes.at[decide_date, to]
        if pd.notna(c_from) and pd.notna(c_to) and c_from != 0:
            diff = float(c_to - c_from)      # added to pre-roll bars
            ratio = float(c_to / c_from)     # multiplied into pre-roll bars
    except KeyError:
        pass
    return {"effective_date": effective_date, "from_symbol": frm, "to_symbol": to,
            "reason": reason, "offset_difference": diff, "offset_ratio": ratio}


def build_roll_map(bars, metas: dict[str, ContractMeta], cfg: RolloverConfig):
    """Decide roll dates from completed-session volume, with a calendar backstop.

    Returns (roll_map, warnings). roll_map columns:
        effective_date, from_symbol, to_symbol, reason,
        offset_difference, offset_ratio
    """
    warnings: list[str] = []
    vol = daily_volume_by_contract(bars)
    if vol.empty:
        return _empty_roll_map(), ["no volume data; cannot build roll map"]

    ordered = sorted(metas.values(), key=lambda m: m.sort_key)
    order_idx = {m.raw_symbol: i for i, m in enumerate(ordered)}

    closes = (bars.sort_values("ts")
                  .groupby(["trade_date", "raw_symbol"], observed=True)["close"]
                  .last().unstack())

    dates = list(vol.index)
    front = vol.loc[dates[0]].idxmax()   # seed: most-traded listing on day one
    rolls: list[dict] = []
    streak = 0

    for i, d in enumerate(dates[:-1]):
        nxt_date = dates[i + 1]
        if front not in vol.columns:
            break

        cand = _pick_candidate(front, vol.loc[d], order_idx, cfg)
        rolled = False

        if cand is not None:
            streak = streak + 1 if vol.at[d, cand] > vol.at[d, front] else 0
            if streak >= cfg.confirm_days:
                rolls.append(_make_roll(nxt_date, front, cand,
                                        "volume_crossover", closes, d))
                front, streak, rolled = cand, 0, True

        if not rolled:
            # Calendar backstop -- never ride a contract into expiry just
            # because the crossover test never fired (thin or erratic volume,
            # holiday weeks).
            meta = metas.get(front)
            if meta is not None:
                # Measure margin at the date the roll would TAKE EFFECT, not at
                # the deciding session. Sessions advance in trading days while
                # expiry margin is in calendar days, so a weekend can carry
                # days_left straight past the threshold (e.g. 3 -> 0 across a
                # Fri/Mon gap) and fire the backstop only after expiry has
                # already passed -- defeating its entire purpose.
                days_left = (meta.expiration.date() - nxt_date).days
                if days_left <= cfg.calendar_backstop_days:
                    fallback = cand or _next_chronological(front, order_idx, vol.columns)
                    if fallback is None:
                        warnings.append(
                            f"{front} within {days_left}d of expiry on {d} with no "
                            "successor contract in the data -- series truncates here")
                        break
                    rolls.append(_make_roll(nxt_date, front, fallback,
                                            f"calendar_backstop({days_left}d)",
                                            closes, d))
                    warnings.append(
                        f"{front} -> {fallback} rolled by calendar backstop on "
                        f"{nxt_date}, not volume crossover ({days_left}d to expiry)")
                    if days_left <= 0:
                        warnings.append(
                            f"{front}: backstop wanted {cfg.calendar_backstop_days}d "
                            f"of margin but the calendar allowed only {days_left}d "
                            f"(weekend/holiday gap) -- roll lands on or after "
                            f"expiry {meta.expiration.date()}; widen "
                            "calendar_backstop_days for this symbol")
                    front, streak = fallback, 0

    return (pd.DataFrame(rolls) if rolls else _empty_roll_map()), warnings


def stitch(bars, roll_map, symbol: str = "", warnings=None) -> ContinuousSeries:
    """Keep, for each trade date, only the bars of the then-active contract."""
    if bars.empty:
        return ContinuousSeries(bars=bars.copy(), roll_map=roll_map, symbol=symbol,
                                warnings=list(warnings or []))

    if roll_map.empty:
        # No roll in the window -- one contract is front throughout. It must be
        # chosen by VOLUME. Taking bars.iloc[0] picks whatever sorts first, and
        # since normalize() sorts by raw_symbol, that is the alphabetically
        # first contract: for MES that selected MESH7 (4,753 lots over the
        # window) instead of MESU6 (67 million). The series still looked
        # well-formed, just built from a nearly untraded contract.
        active = bars.groupby("raw_symbol", observed=True)["volume"].sum().idxmax()
        sel = bars[bars["raw_symbol"] == active].copy()
    else:
        rm = roll_map.sort_values("effective_date")
        segments = []
        current = rm.iloc[0]["from_symbol"]
        seg_start = None
        for row in rm.itertuples():
            m = bars["raw_symbol"] == current
            if seg_start is not None:
                m &= bars["trade_date"] >= seg_start
            m &= bars["trade_date"] < row.effective_date
            segments.append(bars[m])
            current, seg_start = row.to_symbol, row.effective_date
        tail = (bars["raw_symbol"] == current) & (bars["trade_date"] >= seg_start)
        segments.append(bars[tail])
        sel = pd.concat(segments)

    sel = sel.sort_values("ts").reset_index(drop=True)
    roll_dates = set(roll_map["effective_date"]) if not roll_map.empty else set()
    sel["is_roll_bar"] = sel["trade_date"].isin(roll_dates)
    return ContinuousSeries(bars=sel, roll_map=roll_map, symbol=symbol,
                            warnings=list(warnings or []))
