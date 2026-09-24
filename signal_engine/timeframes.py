"""Multi-timeframe orchestration -- the frames every detector runs on.

The feature layer is timeframe-agnostic by design: `three_tail()` clusters
tails on whatever bars it is handed, `trend_bias()` reads whatever frame it is
given. That is the right shape for the detectors, but it left the question of
*which* frame each one should see unanswered, and `params.timeframes` was
configured and read by nobody -- callers hardcoded `"5min"` and `"1D"`, so S19
ran on the entry frame while its own spec timeframe said 10min.

This module answers it once. It resolves the configured timeframes, builds each
distinct one from the 1m base, attaches the features that are well-defined at
that timeframe, and aligns higher-timeframe values back onto the entry frame
causally. Everything downstream -- gates, scoring, backtest -- asks for a role
rather than naming a frequency.

## Roles, not frequencies

A detector needs "the entry frame" or "the HTF", not "5min" or "1h". Those are
config decisions that Phase 4 will grid-search per instrument, and a detector
that names a frequency has silently frozen one. So callers ask by ROLE:

    tfs.frame("entry")        the bars entries are decided on
    tfs.frame("htf")          trend filter (S14) and swing sizing (S1)
    tfs.frame("daily")        gap threshold (S3), daily time count (S18)
    tfs.frame("three_tail")   S19's own chart, which Soloway puts at 10min

Two roles may resolve to the same frequency -- nothing stops `three_tail` being
set to `5min` -- and when they do they share one built frame rather than
resampling twice. A role is what a detector asks for; a frequency is how it
happens to be built this run.

## Causality across timeframes

Everything crossing a timeframe boundary goes through
`structure.align_htf()`, which merges on the HTF bar's CLOSE. Resampled bars
are stamped with their open, so aligning on that would hand every entry bar
inside a forming 1h bar a value that will not exist until the hour ends. This
module never reimplements that merge; there is exactly one copy of the rule.

The reverse direction is not offered. A lower timeframe cannot be aligned onto
a higher one without either leaking (using LTF bars past the HTF bar's open) or
inventing an aggregation the specs do not define, so `align()` only goes HTF ->
entry, which is the direction every spec section actually needs.

## What is NOT decided here

Which trigger fires, how setups are gated, how confidence is scored. This
builds the frames and hands them over; `gates.py` and `scoring.py` decide what
to do with them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.resample import FREQ_ALIASES, resample
from features import confirmation, risk_state, structure
from features.schema import ATR, ATR_MEAN, VOL_REGIME, Params, load_params

# Roles named in params.timeframes, with the spec sections that consume each.
# This is the mapping that was previously implicit in whatever each caller
# happened to hardcode.
ROLE_CONSUMERS = {
    "entry": "S4/S7/S8/S9/S10/S11/S17/S20 triggers, S1 pivots, S5 ranges",
    "htf": "S14 trend filter, S1 swing major/minor sizing (ATR)",
    "daily": "S3 gap threshold, S18 daily time count",
    "three_tail": "S19 tail clusters",
}
ENTRY = "entry"


class TimeframeError(RuntimeError):
    """A role was requested that the config does not define, or cannot build."""


def _is_intraday(freq: str) -> bool:
    """Whether a frequency is shorter than a day.

    Volume expansion (S12) compares a bar against a session-matched or
    hour-of-day baseline; both of those are intraday notions. On daily bars
    there is nothing to match against, so the column would be meaningless --
    and a meaningless column that looks like a real one is worse than an
    absent one, because something downstream will eventually read it.
    """
    return pd.Timedelta(FREQ_ALIASES[freq]) < pd.Timedelta("1D")


@dataclass(frozen=True)
class TimeframeSet:
    """Every frame one symbol's detectors need, built once and shared.

    `frames` is keyed by FREQUENCY and `roles` maps role -> frequency, so two
    roles pointing at the same frequency share one frame rather than holding
    two copies that could drift.
    """
    symbol: str
    params: Params
    symbol_cfg: dict
    roles: dict[str, str]
    frames: dict[str, pd.DataFrame] = field(repr=False)

    # -- lookup ------------------------------------------------------------
    def freq(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise TimeframeError(
                f"{self.symbol}: no timeframe role {role!r}. Configured roles: "
                f"{sorted(self.roles)}. Roles come from params.timeframes; add "
                "it there rather than naming a frequency at the call site."
            ) from None

    def frame(self, role: str) -> pd.DataFrame:
        """The bars for a role, features attached."""
        return self.frames[self.freq(role)]

    def by_freq(self, freq: str) -> pd.DataFrame:
        """A frame by literal frequency -- for S18, whose config names
        timeframes directly rather than roles."""
        if freq not in self.frames:
            raise TimeframeError(
                f"{self.symbol}: {freq!r} was not built. Built: "
                f"{sorted(self.frames)}")
        return self.frames[freq]

    @property
    def entry(self) -> pd.DataFrame:
        return self.frame(ENTRY)

    # -- alignment ---------------------------------------------------------
    def align(self, role: str, values: pd.Series,
              name: str | None = None) -> pd.Series:
        """Put a series computed on `role`'s frame onto the entry frame.

        Causal by construction: `structure.align_htf()` merges on the HTF
        bar's close, so a forming bar is never visible to the entry bars inside
        it. Aligning the entry role to itself is a no-op rather than an error,
        which keeps callers from special-casing the configuration where two
        roles share a frequency.
        """
        freq = self.freq(role)
        if freq == self.freq(ENTRY):
            return values.rename(name or values.name)
        return structure.align_htf(self.entry, self.frames[freq], values, freq,
                                   name=name or f"{role}_value")

    def align_events(self, role: str, values: pd.Series,
                     name: str | None = None) -> tuple[pd.Series, pd.Series]:
        """Put EVENTS detected on `role`'s frame onto the entry frame.

        `align()` forward-fills, which is right for a state like S14's bias and
        wrong for an event: every entry bar inside the next HTF bar would
        re-fire it, which is the S11 failure again. Here an event lands on
        exactly one entry bar -- the first at which its HTF bar is visible
        under the same close-time rule `align()` uses, since visibility is
        taken from `align()` itself rather than recomputed.

        An event whose next entry bar falls in a LATER SESSION is dropped, not
        carried over: a cluster completing on the last bar before the close
        would otherwise arrive at the next open reading as fresh. The bound is
        the session rather than a time window on purpose. A thin market can
        go ten minutes without a print mid-session, and no prints means price
        has not moved -- a one-HTF-step window dropped 81 of MET's 306 S19
        bars that way, all of them still current.

        Returns (value, source bar's open ts), both indexed on the entry
        frame. The source ts is kept because crossing the boundary otherwise
        destroys it, and anything judging WHERE the pattern formed -- Stage 1
        gate 2 -- needs the bar it formed on, not the entry bar it landed on.
        """
        name = name or f"{role}_event"
        freq = self.freq(role)
        src = self.frames[freq]
        if freq == self.freq(ENTRY):
            return (values.rename(name),
                    src["ts"].where(values.notna()).rename(f"{name}_ts"))

        carried = pd.Series(src["ts"].to_numpy(), index=src.index)
        seen = pd.to_datetime(self.align(role, carried), utc=True)
        fresh = seen.notna() & (seen != seen.shift(1))
        src_session = seen.map(pd.Series(src["trade_date"].to_numpy(),
                                         index=src["ts"]))
        land = fresh & (src_session == self.entry["trade_date"])

        by_open = pd.Series(values.to_numpy(), index=src["ts"])
        value = seen.map(by_open).where(land)
        value = value.astype("object").where(value.notna(), None)
        src_ts = seen.where(land & value.notna())
        return value.rename(name), src_ts.rename(f"{name}_ts")

    def atr(self, role: str) -> pd.Series:
        """`role`'s ATR, on `role`'s own frame."""
        return self.frame(role)[ATR]

    def describe(self) -> str:
        rows = [f"{r:11} {f:>6}  {len(self.frames[f]):>6} bars   "
                f"{ROLE_CONSUMERS.get(r, '')}"
                for r, f in sorted(self.roles.items())]
        return f"{self.symbol}\n" + "\n".join(rows)


def _attach_features(bars: pd.DataFrame, freq: str, params: Params,
                     symbol_cfg: dict) -> pd.DataFrame:
    """Attach what is well-defined at this frequency.

    ATR, candle anatomy and CLV are defined at any timeframe. Volume expansion
    is not (see `_is_intraday`), so it is attached only where it means
    something and simply absent elsewhere.
    """
    period = int(params.get("atr.period"))
    if _is_intraday(freq):
        out = confirmation.apply(bars, params, symbol_cfg)
    else:
        out = bars.copy()
        out = pd.concat([out, confirmation.candle_anatomy(out)], axis=1)
        out[confirmation.CLV] = confirmation.clv(out)

    out[ATR] = risk_state.atr(out, period)
    regime, baseline = risk_state.volatility_regime(
        out[ATR],
        int(params.get("atr.regime_mean_window")),
        float(params.get("atr.high_vol_ratio")),
        float(params.get("atr.low_vol_ratio")),
    )
    out[ATR_MEAN] = baseline
    out[VOL_REGIME] = regime
    return out


def required_frequencies(params: Params) -> tuple[dict[str, str], list[str]]:
    """(role -> frequency, every distinct frequency to build).

    S18's `time_count.timeframes` names frequencies directly rather than roles,
    so they are built too -- otherwise the one config that does drive
    multi-timeframe behaviour today would still have nothing behind it.
    """
    roles = {r: str(params.get(f"timeframes.{r}")) for r in ROLE_CONSUMERS}
    extra = [str(f) for f in params.get("time_count.timeframes", [])]

    wanted: list[str] = []
    for freq in list(roles.values()) + extra:
        if freq not in FREQ_ALIASES:
            raise TimeframeError(
                f"timeframe {freq!r} is not resampleable; known: "
                f"{sorted(FREQ_ALIASES)}")
        if freq not in wanted:
            wanted.append(freq)
    return roles, wanted


def build(symbol: str, base_bars: pd.DataFrame, symbol_cfg: dict,
          params: Params | None = None) -> TimeframeSet:
    """Build every configured frame for one symbol from its 1m bars.

    `base_bars` is the stitched continuous series from
    `data.pipeline.build_continuous()` -- this deliberately takes bars rather
    than fetching, so the backtest, the paper loop and the validation scripts
    all orchestrate the same frames over whatever series they were handed.
    """
    params = params or load_params(symbol)
    roles, wanted = required_frequencies(params)
    frames = {freq: _attach_features(resample(base_bars, freq, symbol_cfg),
                                     freq, params, symbol_cfg)
              for freq in wanted}
    return TimeframeSet(symbol=symbol, params=params, symbol_cfg=symbol_cfg,
                        roles=roles, frames=frames)
