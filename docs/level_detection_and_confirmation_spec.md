# Level Detection & Confirmation Math — Spec v1

All thresholds below are **starting defaults**, not fixed truths. They belong in a
`config/params.yaml` (or per-symbol override), never hardcoded in feature logic.
Backtesting will tell you which ones need tuning per instrument.

---

## 1. Swing Highs / Lows (Structure)

**Fractal pivot definition:**
A bar `i` is a **swing high** if `high[i] > high[i-N..i-1]` and `high[i] > high[i+1..i+N]`.
Symmetric for swing low using `low`.

- `N` (lookback/lookahead bars) — default:
  - Lower timeframe (5m/15m): `N = 2`
  - Higher timeframe (1h/4h/Daily): `N = 3–5`
- Confirmed swing points require `N` bars to close *after* the pivot bar (no repainting
  live — a swing high isn't valid until it's actually survived N bars).
- **Major** vs **minor** swing: classify by ATR-normalized size —
  a swing is "major" if its retracement depth `>= 1.0 × ATR(14, HTF)`.
  **Depth is measured from the prior opposite swing, not the following one**
  (e.g., depth of a swing high = that high minus the *preceding* confirmed
  swing low). This is a deliberate, backward-looking choice: it means a
  swing's major/minor classification is known the instant the pivot itself
  confirms, not dependent on a future swing forming first. The exit spec's
  target price (§15, "next major level") and the R:R gate (§16) both need
  that classification available *at signal time* — if majority depended on
  the following swing instead, the target wouldn't exist yet when the gate
  needs to evaluate it. Don't measure depth forward, even though "retracement"
  reads more naturally that way in isolation.

## 2. Prior Day / Week High-Low

- Defined **per session type from the symbol spec table** — RTH only by default
  (e.g., ES RTH = 08:30–15:15 CT), not blended with overnight (ETH) unless the
  symbol config explicitly flags `use_eth_range: true`.
- Prior day H/L = max/min of RTH bars from the most recently completed RTH session.
- Weekly H/L = max/min across all sessions Mon–Fri (rolls at each new week start).

**"Prior day" means the previous *liquid* session, not simply the previous
trade date.** Confirmed necessary by MET, which trades through weekends —
its Monday "prior day" would otherwise pull from Sunday's thin session
(81–144 1-minute bars) instead of Friday's genuinely liquid one (~734 bars),
since a naive "shift by one trade date" picks up whatever session happens to
be immediately prior, liquid or not. Rule: a candidate session qualifies as
"prior day" only if its bar count is `>= 50%` of that instrument's rolling
median trade-date bar count; if it doesn't qualify, skip further back until a
session does. This is a general rule, not a MET-specific patch — it just
happens to matter most for MET given its continuous weekend trading. The 50%
threshold is a starting default, tunable like everything else in this doc,
but *whether* thin sessions get skipped at all is a logic decision, not a
threshold to leave open — it's decided here as yes.

## 3. Gap Zones

- Gap exists if `abs(session_open - prior_session_close) >= gap_threshold`.
- Default `gap_threshold = 0.15 × ATR(14, Daily)`.
- Gap zone = the price interval `[min(open, prior_close), max(open, prior_close)]`.
- Tag as "filled" once price trades back through the entire zone.

**Confirmed asymmetry (Phase 2 real-data validation)**: this definition means
something structurally different depending on session scope. For RTH-scoped
instruments (MES/MNQ/MYM), `session_open`/`prior_session_close` bracket the
full ~17-hour overnight — and since these trade nearly continuously, a "gap"
by this definition fires on the large majority of days (confirmed: 58% of
sessions on MES) rather than flagging anything selective. For the four
continuous-session instruments, the gap only spans the ~1-hour maintenance
halt, so it stays rare and meaningful (4–9 gaps in the same window). Both are
correct applications of the spec as written — the definition just carries
very different selectivity depending on session scope, and this matters
because gap edges feed directly into `level_confluence` scoring (confidence
spec, Stage 2 component 3): a "confluent level type" that's present on most
days barely functions as evidence. **Deliberately not fixed by guessing a
tighter RTH-specific threshold now** — this is exactly the kind of thing to
resolve against actual backtest results in Phase 4, not by picking a new
constant blind. Flagging it here so the eventual tuning pass starts from a
known asymmetry instead of rediscovering it from scratch.

## 4. Breakout / Breakdown Levels

- **Breakout**: a *close* (not just wick) beyond a marked level by at least a buffer,
  to filter noise: `close > level + buffer`, where
  `buffer = max(2 ticks, 0.1 × ATR(entry_timeframe))`.
- **Breakdown**: symmetric, `close < level - buffer`.
- Distinguish **intrabar wick-through** (ignored on its own) from **confirmed close
  beyond** (counts as breakout) — this distinction feeds directly into "failed breakout"
  logic below.

## 5. Consolidation Range Detection

Range regime = true when **either**:
- Rolling `ATR(14) < 0.7 × ATR(14).rolling_mean(50)` (volatility compression), **or**
- `(max(high, last N bars) - min(low, last N bars)) < 1.5 × ATR(14)` for `N = 10–20` bars.

Range boundaries = the max/min of the compression window once flagged.

## 6. "At a Level" / Test Zone Tolerance

Price is considered "testing" a level when:
`abs(price - level) <= max(2 ticks, 0.15 × ATR(entry_timeframe))`.

This tolerance band is what triggers the bot to start watching for a confirmation
event — it does **not** itself trigger an entry.

## 7. Rejection Candle

Bullish rejection (at support):
- `close_location_value (CLV) >= 0.6` (see formula in §11)
- `lower_wick >= 2.0 × body`
- `body >= 0.25 × ATR(entry_timeframe)` (filters doji/noise bars out)

Bearish rejection (at resistance): mirror — `CLV <= -0.6`, `upper_wick >= 2.0 × body`.

## 8. Failed Breakout

- Price closes beyond a level (§4 breakout/breakdown), **then**
- Within `K = 1–3` bars, closes back on the original side of the level.
- Direction of the failed breakout signals the *opposite* trade (failed breakout above
  resistance → short setup).

## 9. Breakout / Retest

- Confirmed breakout close beyond level (§4).
- Price pulls back into the test-zone tolerance (§6) of that same level within
  `K_max = 10` bars, without closing back through it in the failure direction (§8).
- Retest bar shows rejection candle (§7) in the breakout's direction → entry trigger.

## 10. Range Reclaim

- Price closes outside a consolidation range boundary (§5), then
- Within `K = 1–3` bars, closes back inside the range.
- Functionally identical math to "failed breakout," applied to range edges instead of
  discrete S/R levels — **but the direction must match the specific boundary
  being tested.** A reclaim of the range *high* requires the failed-breakout
  leg to be a close *above* that high which then reverts back below it; a
  reclaim of the range *low* requires a close *below* it that reverts back
  above. A close on the "wrong" side of a boundary (e.g., closing below the
  range high without ever having exceeded it) is not a reclaim of that
  boundary and must be excluded — reusing §8's failed-breakout mechanism does
  not mean accepting either direction indiscriminately just because it shares
  the same underlying check. (Confirmed necessary: an unfiltered
  implementation counted both directions against a single boundary and
  overcounted real reclaims by roughly 2x on real data.)

## 11. Momentum Continuation

Trigger requires **all**:
- Trend filter aligned (see §14).
- `body_ratio = body / (high - low) >= 0.6` (strong-bodied candle, not indecisive).
- Candle closes beyond a *minor* level (§1) in the trend direction.
- Volume expansion confirmed (§12).

**This must fire as an event (a transition), not a persistent per-bar
predicate.** §4's breakout/breakdown logic already gets this right by
shifting a per-bar state so it fires once at the qualifying transition; §11
must use the identical pattern. Written as a bare condition check
(`close > minor_level`), this stays true for every subsequent bar that
happens to remain beyond the level, re-firing on any strong-bodied,
volume-expanded bar indefinitely — producing thousands of "signals" where a
selective trigger should produce tens. (Confirmed on real data: momentum
continuation fired 2,000–4,300 times per instrument vs. 150–460 for the
other selective trigger types — an order of magnitude difference that is
itself the symptom, not a sign the trigger is simply more common.) Fire only
on the bar where the condition first becomes true after not having been true;
do not re-fire while it remains true.

## 12. Volume Expansion (Confirmation)

`volume[i] >= multiplier × avg_volume(N, session_matched)`

- Default `multiplier = 1.5`, `N = 20`.
- **Futures-specific**: average must be session-matched (RTH-only average compared to
  RTH bar, ETH-only average compared to ETH bar) — blending session types badly skews
  the baseline given how much lighter overnight volume is.

## 13. Close Location Value (CLV) — used by §7 and §11

```
CLV = ((close - low) - (high - close)) / (high - low)
```
Range: -1 (closed at the low) to +1 (closed at the high). Guard against
division by zero on zero-range bars (skip / treat as CLV = 0).

## 14. Trend Filter

- One EMA only: `EMA(20)` or `EMA(50)` on the higher timeframe (config per symbol).
- Bias:
  - **Bullish**: `close > EMA` AND `EMA[i] > EMA[i - 5]` (EMA sloping up, not flat).
  - **Bearish**: mirror.
  - **Neutral/chop**: neither condition holds cleanly.

**Resolved (previously conflicted with the confidence-scoring spec's
Directional Context component — gates.py surfaced the conflict directly)**:
neutral/opposite HTF bias is a **hard gate for continuation-type triggers
only** (momentum continuation, trend-direction breakout/retest, Confirmation
Signal used as a continuation entry) — these fail outright unless HTF bias
matches trade direction, since a continuation trade fundamentally requires a
trend to continue; neutral or opposite bias means no trade, no exceptions.
**Reversal-type triggers (rejection candle, three-tail, failed breakout,
range reclaim, engulfing) are never gated by HTF state** — they pass through
to scoring regardless of bias, where the confidence-scoring spec's
Directional Context component (1.0 exhaustion-supported / 0.5 neutral / 0.2
fighting a strong trend) differentiates conviction. A rejection trade at a
clean level during a chop market is a normal, valid setup — hard-gating it
out just because the trend is flat would contradict the whole premise of
trading price action at levels. The earlier wording ("skips new setups
against HTF ambiguity") was accurate for continuation but not written with
reversal triggers in mind; this is the corrected, complete rule.

## 15. ATR & Risk State

- `ATR(14)` computed per trading timeframe (not HTF) — this is what sizes stops.
- **Stop** = invalidation level ± buffer, buffer default `0.1–0.25 × ATR`.
- **Target** = the next major level (§1) in trade direction when one exists;
  falls back to `2.0 × risk` (the R:R gate's own minimum, §16) when no
  confirmed major swing exists yet in the trade direction — never discard a
  setup just because a major level hasn't formed.

  **Resolved — corrects an internal contradiction in this doc's earlier
  wording.** The original phrasing ("whichever is nearer/more conservative")
  read as a cap, while naming the R:R gate's threshold `min_reward_risk` and
  using it purely as a floor in §16 ("reject if `RR < min_reward_risk`") reads
  as a floor. Those can't both be true — taking "nearer" literally means the
  major level only ever gets used as a target when it's *inside* 2R (which
  then fails the gate anyway), so every setup that ever passes has RR exactly
  2.0, permanently zeroing out the confidence-scoring spec's
  `reward_risk_quality` component. The floor reading is correct: the major
  level is the real target whenever one exists, 2R is strictly the gate's
  minimum bar to clear, and RR is allowed to run up to the scoring spec's
  `RR_cap` (4.0) when a farther major level genuinely supports it.
  If the major level and the 2R floor disagree by more than
  `disagreement_flag_ratio` (default `1.5`), flag it as a signal-quality note
  for the Signal Engine to log — this is diagnostic information about how far
  the actual target sits from the bare minimum, not a reason to alter the
  target itself.
- **Volatility regime**:
  - High vol: `ATR(14) > 1.3 × ATR(14).rolling_mean(50)`
  - Low vol: `ATR(14) < 0.7 × ATR(14).rolling_mean(50)`
  - Used to gate position sizing (reduce size in high-vol regime) and to optionally
    skip momentum-continuation setups in low-vol chop.

## 16. Reward/Risk Gate (Entry Filter)

Compute at signal time, before order submission:
```
RR = (target - entry) / (entry - stop)      # long
RR = (entry - target) / (stop - entry)      # short
```
Reject any setup where `RR < min_reward_risk` (default 2.0). This is a hard gate in
the Signal Engine, applied after trigger + confirmation both pass — it should be the
last check before a signal is emitted, not baked into the trigger logic itself.

## 17. Confirmation Signal (Two-Stage Close Confirmation)

Adapted from the Soloway "Yellow Alert / Red Alert" framework. This is a **stricter
alternative** to the simple buffer-based breakout in §4 — use it as a configurable
mode (`breakout_mode: "buffer" | "confirmation_signal"`) rather than replacing §4
outright, since the two modes trade off differently (confirmation-signal mode fires
later but with fewer false positives).

- **Yellow Alert (piercing bar `P`)**: the first bar where price crosses the level `L`
  — `high[P] > L` (resistance) or `low[P] < L` (support). This is a *flag*, not a
  signal. No trade is permitted on this bar alone.
- **Red Alert (confirmation)**:
  - Breakout confirmed when a subsequent bar `C` closes above `high[P]`:
    `close[C] > high[P]`.
  - Breakdown confirmed when `close[C] < low[P]`.
  - Note this is confirmation against the **piercing candle's own extreme**, not
    just the original level — this is what makes it stricter than §4's buffer method.
- **Expiration window**: if confirmation hasn't occurred within `K_confirm` bars of
  the piercing bar, discard the setup as stale (default `K_confirm = 3–5`, tune per
  timeframe — Soloway's own framing implies this often plays out over a full session
  on daily charts, so intraday timeframes need a proportionally shorter window).
- **Failure case**: if price closes back through `L` (the original level, not just
  fails to exceed `high[P]`/`low[P]`) before confirmation — treat as failed breakout
  per §8, which itself can become the opposite-direction setup.

## 18. Time Count (Exhaustion Counter)

- Direction of period `i`: **up** if `close[i] > close[i-1]`, **down** if
  `close[i] < close[i-1]` (flat/unchanged bars don't reset or extend the count —
  hold the count at its current value).
- `time_count` = number of consecutive same-direction periods ending at the current
  bar.
- **Exhaustion flag**: `time_count >= exhaustion_threshold` (default `6`, matching
  Soloway's stated 6–7 session observation — tune per instrument/timeframe since
  this was described in a daily-chart, single-stock context and futures may exhaust
  faster or slower).
- Usage: this is a **context/filter flag, not a standalone trigger**. Once
  `exhaustion=true`:
  - Reduce confidence score (or outright skip) on momentum-continuation triggers
    (§11) in the exhausted direction.
  - Increase confidence weight on reversal-type triggers (rejection candle §7,
    three-tail cluster §19, failed breakout §8) in the opposite direction, especially
    when they coincide with a marked level.
- Applicable per timeframe — compute independently on Daily, 1H, etc., since a daily
  exhaustion count and an intraday one answer different questions and shouldn't be
  blended into one number.

## 19. Three Tail Theory (Tail Cluster Reversal)

- **Tail bar definition** (reuses §7's rejection-candle math): a bar qualifies as a
  "tail bar" on the upper side if `upper_wick >= wick_body_ratio × body` (default
  ratio `2.0`) and `body <= 0.5 × ATR(entry_timeframe)` (keeps it a rejection wick,
  not a strong trend candle with a small wick). Mirror for lower-side tail bars.
- **Clustering**: within a lookback window of `N_lookback` bars (default `5–8`,
  typically evaluated on a lower timeframe like 10-minute per Soloway's usage),
  identify tail bars of the *same side* (all upper or all lower).
- **Alignment tolerance**: the extreme points of the candidate tail bars (the highs,
  for upper tails; the lows, for lower tails) must fall within
  `cluster_tolerance = max(2 ticks, 0.1 × ATR(entry_timeframe))` of each other —
  this is what makes them "aligned at a specific level" rather than three unrelated
  wicks at different prices.
- **Trigger condition**: `count(clustered tail bars) >= min_tails_required` (default
  `3`, per the theory's name — consider exposing a looser `2`-of-window variant as a
  tunable relaxation, but keep `3` as the true-to-name default).
- **Signal**: fires as a reversal trigger, functionally equivalent in weight to a
  rejection candle (§7) but with higher confidence given the multi-bar confirmation
  — feed it into the Signal Engine's confidence score at a higher weight than a
  single rejection candle, not as an entirely separate trigger type.
- Best used **at or near a marked level** (§2–§6) — a three-tail cluster in open
  space with no structural level nearby is weaker evidence and should score lower.

**Two-sided bars — a single bar qualifying as both an upper and a lower tail
in the same clustering window — resolve to "no trade," excluded from the
directional signal entirely** (not defaulted to either long or short).
Confirmed necessary on real data: an earlier implementation kept "whichever
direction ran last" in its own internal bookkeeping, which happened to always
be long — silently mislabeling every genuinely two-sided bar as a long signal
(149 of 300 MET events, not a rare edge case). This follows the same
principle as nullable `is_major` in §1 — a genuinely ambiguous state gets
excluded explicitly, never forced into a default bucket. `gates.py` must
handle a "both" label as a distinct, non-directional case.

**Zero-range bars (`open == high == low == close`) must not qualify as a
tail of either side.** The wick-to-body ratio check (`upper_wick >=
wick_body_ratio × body`) trivially passes when both sides are zero
(`0 >= 2.0 × 0`), so a bar with no real trading activity within it can
currently register as a "tail" — and on thin instruments this is the
dominant driver of spurious two-sided classifications, not genuine
conflicting rejection evidence. This guard belongs in the shared candle-
anatomy logic, not patched into §19 alone — §7 (rejection) and §20
(engulfing) both reuse the same wick/body math and are equally exposed to
this degenerate input, even though it was found via three-tail specifically.

## 20. Bullish / Bearish Engulfing (Trigger)

This definition existed only in prior chat discussion, never in a spec file —
that gap is now closed.

- **Bullish engulfing**: `close[i-1] < open[i-1]` (prior bar red) AND
  `close[i] > open[i]` (current bar green) AND `open[i] <= close[i-1]` AND
  `close[i] >= open[i-1]` — i.e., the current bar's body fully engulfs the
  prior bar's body (body-to-body, not wick-to-wick).
- **Bearish engulfing**: mirror image of the above.
- **Strength filter**: require `body[i] >= engulfing_strength_multiplier ×
  body[i-1]` (default `1.3`) so a marginal engulf of a tiny prior candle
  doesn't count as a real signal — a small body engulfing an even smaller one
  is noise, not conviction. **This relative check is not sufficient on its
  own — also require `body[i-1] >= engulfing_min_prior_body` (default
  `0.5 × ATR(entry_timeframe)`).** A purely relative multiplier has no floor:
  any ordinary-sized bar trivially "engulfs" a near-zero one-tick doji at
  1.3x or more, since 1.3x of almost nothing is still almost nothing.

  **Correction, confirmed against real data**: an earlier version of this
  note claimed one-tick-doji cases drove "the majority" of engulfing
  firings — they don't (20.7% on MES, 3.2% on MGC, 31.7% on MET; a real
  contributor, not a majority one). The actual problem is broader: even with
  a modest floor, engulfing remains far more frequent than a genuinely
  selective trigger should be. At `0.10 × ATR`, the floor retains 73–92% of
  the unfiltered firing count and engulfing still fires on ~11.5% of bars
  against ~1% for rejection candle (§7). At `0.50 × ATR`, retained count
  drops to 9–19% of the original — landing in the same order of magnitude as
  the other selective triggers, which is the actual bar this filter needs to
  clear. `0.50` is the corrected default for that reason. Both the
  multiplier and the floor remain tunable via backtest, same as everything
  else in this doc — but the floor needed to start in the range that
  actually makes the trigger selective, not an untested guess.
- Base confidence-score weight for this trigger type is `0.80` (already
  specified in the confidence scoring spec, §Stage 2 trigger-quality table) —
  that number was never in question; only this geometric definition was
  missing.

---

## Open parameters to tune via backtest (do not guess-and-freeze)

| Param | Default | Sensitivity |
|---|---|---|
| Swing pivot N | 2 (LTF) / 3–5 (HTF) | High — changes structure entirely |
| Gap threshold | 0.15 × ATR(D) | High for RTH-scoped instruments specifically — confirmed 58% of MES sessions flag as "gap," diluting level_confluence; low for the four continuous instruments (stays rare/selective) |
| Breakout buffer | max(2 ticks, 0.1×ATR) | High — noise vs. false breakout tradeoff |
| Test-zone tolerance | max(2 ticks, 0.15×ATR) | Medium |
| Rejection wick ratio | 2.0× body | Medium |
| Failed-breakout window K | 1–3 bars | Medium |
| Volume expansion multiplier | 1.5× | High — very instrument-dependent |
| Trend EMA slope lookback | 5 bars | Low |
| Min reward/risk | 2.0 | High — trade frequency vs. quality |
| Confirmation signal expiration (K_confirm) | 3–5 bars | High — too short = misses real confirms, too long = stale setups |
| Time count exhaustion threshold | 6 periods | Medium — very instrument/timeframe dependent |
| Three-tail lookback window | 5–8 bars | Medium |
| Three-tail cluster tolerance | max(2 ticks, 0.1×ATR) | High — too loose merges unrelated wicks |
| Three-tail min count required | 3 | Medium — consider 2-of-window as a relaxed variant |
| Engulfing strength multiplier | 1.3× | Medium — too low lets marginal engulfs through as noise |
| Engulfing minimum prior body (absolute floor) | 0.5 × ATR(entry_timeframe) | High — corrected from an initial 0.1× default that barely filtered anything; this is what actually makes the trigger selective |
| Prior-day liquidity floor (session qualifies as "prior day") | 50% of rolling median trade-date bar count | Medium — mainly affects MET's weekend sessions; low impact on the other six |

Every one of these should be a named field in the symbol/config spec so backtesting
can grid-search or walk-forward optimize them per instrument rather than using one
global constant across ES, NQ, CL, and GC.

## A note on integrating these with the existing feature groups

- **Confirmation Signal (§17)** slots into **Trigger** as an alternate breakout/
  breakdown detection mode — don't run both §4 and §17 simultaneously per setup;
  pick one mode per symbol/strategy variant and A/B them in backtest.
- **Time Count (§18)** slots into **Risk State / context**, not Trigger — it should
  never fire a trade on its own, only adjust confidence weighting on other triggers.
- **Three Tail Theory (§19)** slots into **Trigger**, alongside rejection candle
  (§7) and engulfing (from the prior message) — all three are "reversal-type"
  triggers and can share one confidence-weighting scheme in the Signal Engine
  (three-tail > engulfing > single rejection candle, as a starting confidence
  ordering, tunable by backtest results).
