# Confidence Scoring Formula — Signal Engine Spec v1

## Design principle

**Hard gates decide "is this a valid setup at all." The weighted score decides
"how good is it relative to other valid setups."** Don't blend these two — a
setup that fails a hard gate should never reach the scoring stage no matter how
well it might otherwise score. This mirrors discretionary trading logic (some
conditions are non-negotiable) and keeps the score interpretable rather than a
black box that occasionally launders a bad setup through a lucky score.

The score itself is a **rule-based weighted sum** to start — deliberately not ML.
Once you've logged enough labeled backtest/live trades (win/loss + R multiple),
this is the natural place to later fit a logistic regression or gradient-boosted
model against the same feature set, per your original "rule-based first, AI
classification second" architecture. Don't skip the rule-based version to jump
straight to ML — you need it as a baseline to know whether the ML version is
actually adding value or just overfitting to noise.

---

## Stage 1 — Hard Gates (boolean, all must pass)

A setup that fails any of these is discarded before scoring, full stop:

1. **Structure valid** — a recognized trigger type fired (§7–§11, §17, §19 from
   the level-detection spec): rejection candle, breakout/retest, failed breakout,
   range reclaim, momentum continuation, confirmation-signal breakout, engulfing,
   or three-tail cluster.
2. **Level present** — the trigger occurred at or near a marked level (§1–§6;
   §1 major swings count only while live, per §1's liveness rule),
   not in open space, with two exceptions: **momentum continuation** (it can
   fire off a minor level per §11's own definition — this was always a
   "minor level satisfies it" clarification, not a true no-level exemption),
   and **three-tail (§19)**, which genuinely can fire in open space. This
   second exemption resolves a real inconsistency found during gates.py
   construction: `params.yaml` already had `three_tail.requires_nearby_level:
   false`, and §19's own text always said an open-space cluster "should
   score lower," never that it should be excluded — but gate 2 as originally
   written didn't name three-tail in its exemption list, silently
   contradicting both. No separate penalty mechanism is needed for the
   "weaker evidence" case: `level_confluence` (Stage 2, component 3) already
   scores near zero when nothing is nearby, so the framing holds without
   extra logic.

   **When the level is checked.** For level-free triggers (rejection,
   engulfing, and three-tail when not exempt), the check is on the decision
   bar. For **level-defined** triggers (failed breakout §8, breakout/retest
   §9, range reclaim §10, Confirmation Signal §17), it is on **the bar
   immediately before the pattern's first bar**: the level must have been a
   live marked level when the pattern began. The reason is structural. These
   patterns are built on a close through their level, and under §1's
   liveness rule that close kills a swing; a range escape likewise drags the
   rolling range edge onto the escape bar. Checked at the decision bar, the
   level the pattern is about would already be gone, and gate 2 would fail
   every breakout-family trigger on a swing level and every range reclaim. A
   level that only became marked after the pattern began does not count.
3. **Confirmation present** — volume expansion (§12) is required on every
   trigger type except three-tail, evaluated on the bar the trigger actually
   completes on. CLV (§13) is not a substitute for volume expansion here; it
   remains part of specific trigger definitions where the level-detection
   spec already requires it (e.g., rejection candle §7's own CLV >= 0.6
   condition), but it does not satisfy gate 3 on its own for any trigger
   type.

   **Three-tail (§19) is exempt**, as it is from gate 2
   (`three_tail.requires_volume_expansion: false`). Its tail bars are
   small-bodied and quiet by construction. Measured on real data across all
   seven instruments, the bar that completes a cluster has a median volume
   ratio of 0.53, against 0.84 for all 10-minute bars. Requiring expansion
   on that bar kept 15 of 265 otherwise-passing S19 setups, and those 15
   did no better in a forward check than the rest (1 win in 15, −0.69R,
   against +0.29R for the 250 without expansion). The sample is small, so
   this shows no support for the requirement, not that volume hurts. Volume
   still counts for three-tail through Stage 2's `confirmation_strength`,
   so a quiet cluster scores lower rather than being excluded. If the flag
   is set true, the check applies to the 10-minute bar the cluster
   completes on (its own frame, not the entry frame).
4. **R:R meets minimum** — `RR >= min_reward_risk` (default 2.0), computed per
   §16 of the level-detection spec.
5. **Continuation triggers require matching HTF bias** (resolved — previously
   flagged here as "consider making this a hard gate," now decided). Momentum
   continuation, trend-direction breakout/retest, and Confirmation Signal used
   as a continuation entry all fail this gate if HTF bias (§14) is neutral or
   opposite to the trade direction — no exceptions, regardless of how well the
   rest of the setup scores. This does **not** apply to reversal-type triggers
   (rejection, three-tail, failed breakout, range reclaim, engulfing), which
   are never gated by HTF state — they proceed to Stage 2 regardless of bias,
   where the Directional Context component (below) differentiates conviction
   instead of gating outright.
6. **No conflicting risk-control veto** — max daily loss not hit, max open
   positions not exceeded, not in a post-loss cooldown window, symbol passes
   liquidity filter, no unresolved major news flag (per your original risk
   controls list).

If all six pass, proceed to Stage 2. If not, the setup is simply not logged as
a candidate — it doesn't get a low score, it doesn't exist as a signal.

---

## Stage 2 — Weighted Confidence Score (0–100)

```
score = 100 × [
    0.25 × trigger_quality
  + 0.20 × confirmation_strength
  + 0.20 × level_confluence
  + 0.15 × directional_context
  + 0.15 × reward_risk_quality
  + 0.05 × volatility_fit
]
```

Weights are a **starting default** — treat them as config, and once you have
enough labeled trades, this is exactly what you'd hand to a logistic regression
to re-derive empirically rather than trust hand-picked weights forever.

### 1. Trigger Quality (weight 0.25)

Each trigger type gets a **base score** reflecting how much evidence it
inherently represents, then a small magnitude-based adjustment — strength
should nudge the score, not let a strong weak-type trigger outrank a plain
strong-type trigger.

| Trigger type | Base score |
|---|---|
| Three-tail cluster (§19) | 1.00 |
| Confirmation Signal breakout (§17) | 0.90 |
| Breakout/retest (§9) | 0.85 |
| Bullish/bearish engulfing | 0.80 |
| Single rejection candle (§7) | 0.70 |
| Failed breakout (§8) / range reclaim (§10) | 0.70 |
| Momentum continuation (§11) | 0.60 |

```
trigger_quality = base_score × (0.85 + 0.15 × magnitude_factor)
```
- `magnitude_factor` = trigger-specific normalized strength, capped [0, 1]:
  - Rejection/tail/engulfing types: `min(wick_or_body_ratio / (2 × threshold), 1.0)`
  - Momentum continuation: `min(body_ratio / 0.8, 1.0)`
  - Confirmation Signal: inverse of bars-to-confirm, `min(K_confirm_max / bars_taken, 1.0)`
    (faster confirmation = more conviction, per Soloway's own emphasis on decisive closes)
  - **Resolved for the types the list above does not cover:**
    - Breakout/retest (§9): the retest bar's rejection magnitude (wick/body over
      2 × `rejection.wick_body_ratio`), since §9 defines that bar as a
      rejection candle.
    - Three-tail (§19): each clustered tail bar's ratio is capped first, then
      averaged across the cluster. Read off one bar, a zero-body tail (ratio
      infinite) saturated it; 17% of completing bars have a zero body.
    - Failed breakout (§8) / range reclaim (§10): no magnitude is defined, so
      they take the neutral midpoint (0.5), logged as such.
    - Momentum's `0.8` lives in config (`magnitude.momentum_full_body_ratio`).
  - **Known structural saturation:** a §17 confirmation resolves within
    `K_confirm_max` bars or expires, so `K_confirm_max / bars_taken` is never
    below 1 and its magnitude is always 1.0. Harmless while §17 mode is off.

This keeps the 0.85–1.00 multiplier range narrow on purpose — magnitude is a
tiebreaker within a trigger type, not a way to make a weak trigger type beat a
strong one.

### 2. Confirmation Strength (weight 0.20)

```
confirmation_strength = average(volume_score, clv_score)

volume_score = min(volume_ratio / (2 × volume_expansion_multiplier), 1.0)
clv_score    = max(0, CLV × direction)    # +1 long, -1 short
```
**Resolved: CLV is signed in the trade direction and floored at zero.** The
original `abs(CLV)` credited a long whose trigger bar closed at its own low as
fully as one that closed at its high, treating contradictory evidence as
confirming. That was 4.7% of candidates. Floored rather than negative: a close
against the trade earns nothing here, and the other components handle
overall quality. Both scores are read on the bar the trigger completes on
(the 10-minute bar for three-tail); no volume baseline leaves the score
unscored, not zero.

**Three-tail (§19) is scored on CLV alone:** `confirmation_strength =
clv_score`. Its tail bars are small-bodied and quiet by construction, the
reason it is exempt from gate 3, and averaging volume in cost it about 10
points against every other type (volume score 0.22 vs 0.70–0.80). The volume
score is still logged for it. This is expected to be a small move, since its
own CLV averages only 0.30; if three-tail still ranks at the bottom, the
better basis is the cluster's own tail character rather than any single
bar's volume or CLV (open_questions #17).

### 3. Level Confluence (weight 0.20)

Count distinct level *types* (prior day H/L, weekly H/L, major swing point, gap
zone edge, channel boundary, consolidation range edge) that fall within the
test-zone tolerance (§6) of the level being traded:

```
level_confluence = min(confluent_type_count / 3, 1.0)
```
Capped at 3 confluent types for full score — a level with 5 things stacked on
it isn't meaningfully better evidence than one with 3; don't let this term
runaway-dominate the total score.

**Resolved:** the traded level's **own type counts**, so an isolated real level
scores 1/3 rather than the same as open space. Types, not levels: a prior-day
high and low stacked together are one type. The marked levels are found with
exactly gate 2's test and on gate 2's bar (the bar before the pattern for
level-defined triggers). Rejection and engulfing use the levels their bars
reached; momentum's minor level is not itself a marked type, so only marked
levels near it count. Channel boundaries count once channels are defined.

### 4. Directional Context (weight 0.15)

This term handles trend alignment *and* the Time Count exhaustion flag (§18)
together, because they answer the same underlying question ("does the broader
context support this trade") from opposite trigger categories:

- **Continuation-type triggers never reach this component with a bad
  alignment** — resolved as a Stage 1 hard gate (see gate 5 above), not a
  Stage 2 score. A continuation trigger reaching scoring at all means HTF
  bias already matched trade direction, so this branch is `1.0` for
  continuation types; there's no `0.0` case to score here anymore.
  - **Except momentum firing into exhaustion: `0.6`.** When §18's exhaustion
    flag is active in momentum's own trade direction (any timeframe in
    `time_count.timeframes`, read on the entry bar before the pattern, the
    same anchor as the reversal branch), this is riding a possibly tiring
    trend. That is reduced, but not treated like fighting a trend (0.2): it is
    still trend-aligned by construction. This resolves §18's "reduce
    confidence on momentum-continuation triggers in the exhausted direction",
    which the fixed 1.0 had left applied nowhere. It applies to momentum only,
    which §18 names; breakout/retest and §17 keep 1.0. It is expected to touch
    about 6% of momentum setups. Whether fresh-but-unremarkable continuation
    should ever score below 1.0 is a separate, larger question left open.
- **If trigger is reversal-type** (rejection candle, three-tail, failed
  breakout, range reclaim, engulfing):
  - `1.0` if Time Count exhaustion flag (§18) is active in the direction being
    faded, on **any** timeframe in `time_count.timeframes`, read on the entry
    bar **before the pattern starts**. At the decision bar the reversal bar's
    own close has already broken the run being faded, so the branch was all
    but empty by construction (1 of 1,559), the same timing error as gate 2's
    level check. Higher-timeframe counts align causally.
  - `1.0` if the reversal is **with** the HTF trend (e.g. a long rejection at
    support while the HTF is bullish): buying a pullback with the trend, the
    textbook higher-probability reversal context. The original text had no
    case for it, and it is 31% of reversal setups.
  - `0.5` if HTF is neutral/chop (no strong trend to fight).
  - `0.2` if fading a strong, non-exhausted HTF trend — this is the lowest-
    conviction case (catching a falling knife with no exhaustion evidence) and
    should score accordingly, not be excluded outright, since reversals do
    occasionally work without a clean exhaustion signal.
  - HTF bias unknown (still warming up): unscored, not defaulted.

### 5. Reward/Risk Quality (weight 0.15)

```
reward_risk_quality = min((RR - min_reward_risk) / (RR_cap - min_reward_risk), 1.0)
```
Default `RR_cap = 4.0` — caps the benefit of very large theoretical R:R, since
targets far beyond the next couple of levels are increasingly unreliable
projections, not free extra conviction.

### 6. Volatility Fit (weight 0.05)

Small, deliberately low-weight modifier — volatility regime (§15) should
influence position sizing more than setup ranking, but it's worth a light touch
here for setup-type fit:

- Breakout/momentum triggers: `1.0` in normal/high vol, `0.5` in low vol
  (compressed volatility regimes produce more false breakouts).
- Reversal triggers (rejection, three-tail): `1.0` in normal/low vol, `0.6` in
  high vol (wide bars make wick-based rejection signals noisier).
- **Resolved:** the rows follow gate 5's classification. Breakout = the
  continuation triggers (momentum, breakout/retest, §17); reversal = its closed
  reversal list, which adds engulfing, failed breakout and range reclaim. The
  original rows named neither of those three.

---

## Using the score

- **Ranking**: when multiple symbols/setups qualify simultaneously, take the
  highest-scoring first, subject to max open positions. An unscored setup (a
  component whose input does not exist yet) is not ranked, and never treated
  as a zero.
- **Threshold filter**: consider a minimum score floor (e.g., `60`) below which
  a qualifying-but-mediocre setup is skipped entirely — start permissive (e.g.,
  `50`) and raise it once backtest data shows where the win-rate/score
  correlation actually breaks.
- **Position sizing scalar** (optional, use cautiously): a mild scalar like
  `size_multiplier = 0.75 + 0.5 × (score / 100)` gives modestly larger size to
  higher-confidence setups within your fixed max-risk-per-trade ceiling — do
  **not** let this become a large multiplier, since that reintroduces exactly
  the kind of overconfident sizing your fixed-percent-risk rule was meant to
  prevent.
- **Logging**: log every component sub-score alongside the final number and the
  trade outcome. This is the dataset that eventually lets you validate — or
  replace — these hand-picked weights with a fitted model.

## What NOT to do

- Don't let this score creep toward 15+ components "for thoroughness" — every
  added component is another thing to overfit to backtest noise. Six components
  is already a lot for a rule-based system; resist the urge to add an EMA-slope
  bonus, an RSI divergence bonus, etc.
- Don't let the score override the Stage 1 hard gates. If you ever find
  yourself tempted to let a 95-score setup skip the R:R minimum "just this
  once," that's a sign the gate belongs in the score instead — don't quietly
  bypass it in code.
