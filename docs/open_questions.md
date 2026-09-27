# Open questions

Items 1, 2 and 7 are resolved and kept for the record; 3-6 are live.
Items 7-9 were raised while building the trigger layer; 10-12 by the Phase 2
validation run (`scripts/plot_triggers.py`) and are resolved in spec and code.
Item 13 was the first piece of Signal Engine work and has landed.




Tracked here rather than in chat so they stay attached to the code. Each names

the config key that is parked on `UNRESOLVED` or `UNDEFINED_IN_SPEC`, so

nothing silently picks a default.



---



## 1. ~~"Major" swing depth reference~~ — RESOLVED (spec §1)



**Config:** `params.yaml > swings.major_depth_reference`



Level spec §1 classifies a swing as major when *"its retracement depth

`>= 1.0 × ATR(14, HTF)`"*, but never says what the depth is measured **from**.

Three readings, each producing different structure:



- **A — from the prior opposite swing.** Depth of a swing high = that high

  minus the preceding swing low. Backward-looking; known as soon as the pivot

  confirms.

- **B — to the following opposite swing.** Depth = the high minus the *next*

  swing low. Forward-looking, so a swing's majority is unknown until the next

  one forms — which delays every level derived from it.

- **C — the smaller of the two.** Requires meaningful movement on both sides.

  Strictest; also inherits B's delay.



**Why it matters beyond §1:** exit spec §15 sets the primary target to the

"next major level in trade direction," and the TRAILING state trails behind

"each newly confirmed swing point." If majority can only be known in

retrospect (B or C), a target may be unavailable at signal time, when §16's

R:R gate needs it.



**Resolved: A**, written into spec §1. Depth is measured from the prior

opposite swing, never the following one, so classification is known the

instant a pivot confirms. `params.yaml > swings.major_depth_reference:

prior_opposite_swing`.



---



## 2. ~~Engulfing undefined~~ — RESOLVED (spec §20)



**Config:** `params.yaml > engulfing.status`



The scoring spec gives engulfing a base score of **0.80** (between

breakout/retest at 0.85 and single rejection at 0.70), lists it in Stage 1

gate 1, and the repo structure assigns it to `features/triggers.py`. The

level-detection spec attributes it to *"the prior message"* — a document not

in this repo. So the only trigger with a scoring weight and no definition.



**Resolved** by a new spec §20: body-to-body (not wick-to-wick), opposite-

coloured bars, plus a RELATIVE strength filter `body[i] >= 1.3 x body[i-1]` --

not an ATR floor like §7 uses. `params.yaml > engulfing.strength_multiplier`.



---



## 3. Channels are a confluence type but never defined



**Config:** `params.yaml > channels.status`



Scoring §3 counts "channel boundary" as one of the distinct level types

feeding `level_confluence`, and the repo structure lists "parallel channel

logic" under `features/structure.py`. No spec section defines how a channel is

constructed or when it is valid.



Lower urgency than 1 and 2: confluence is a *count* capped at 3, so channels

can be omitted initially and the term still works — it just scores slightly

lower on levels where a channel would have counted.



---



## 4. ~~Neutral-HTF policy contradicts itself~~ — RESOLVED (spec §14, scoring gate 5)



**Config:** `params.yaml > trend.neutral_policy`



Level spec §14 says a neutral/chop HTF means the bot *"skips new setups"* — an

explicit no-trade filter. Scoring §4 assigns reversal triggers **0.5** when

*"HTF is neutral/chop"*, which presupposes those setups reach scoring at all.

Both cannot hold. Either neutral is a hard gate (and the 0.5 branch is dead

code), or it is a scoring penalty (and §14's "skips" is wrong).




**Resolved:** neutral or opposite HTF bias is a hard gate for
CONTINUATION-type triggers only: momentum, trend-direction breakout/retest,
and §17 used as a continuation entry. That is scoring Stage 1 gate 5.
Reversal-type triggers (rejection, three-tail, failed breakout, range reclaim,
engulfing) are never gated by HTF. They reach scoring, where Directional
Context's 1.0 / 0.5 / 0.2 branches differentiate them. Both statements survive:
§14's "skips" applies to continuation, and scoring's 0.5 branch applies to
reversals. `trend.neutral_policy` is no longer UNRESOLVED.

---



## 5. ~~Hard gate 3 overlaps gate 1~~ — RESOLVED (scoring gate 3)



Scoring Stage 1 gate 3 requires "volume expansion (§12) **or** CLV threshold

(§13) met, per the trigger type's own requirement." But §7's rejection candle

already requires `CLV >= 0.6` to fire at all, and §11's momentum continuation

already requires volume expansion. So for those two, gate 3 is satisfied by

definition the moment gate 1 passes.



Question: does a rejection candle need volume expansion *in addition* to its

own CLV, or does its internal CLV discharge gate 3? Materially changes trade

frequency.




**Resolved:** volume expansion (§12) is required on EVERY trigger type,
read on the bar the trigger completes on. For three-tail that is the 10min bar
the cluster completed on, not the entry bar. CLV never satisfies gate 3 on its
own. It stays inside the trigger definitions that already require it, like
§7's CLV >= 0.6. So a trigger's internal confirmation does not discharge gate
3, and rejection candles now also need volume expansion.

---



## 6. ~~§15 target wording~~ — RESOLVED (spec §15, exit spec Part 3)



§15 sets the target to the "next major level in trade direction, or `2.0 ×

risk` **minimum**, whichever is **nearer**." Taking the nearer of the two means

that when the next major level is closer than 2R, the target lands *below* 2R

and §16 then rejects the setup for `RR < min_reward_risk`. That is coherent

behaviour, but "minimum" describes the opposite of what the rule does.



`targets.disagreement_flag_ratio: 1.5` supplies the threshold §15 asks for

("flag if the two disagree by a lot") but never gives.




**Resolved, and it was not cosmetic.** Taken literally, "nearer" makes every
passing setup's RR exactly 2.0, which permanently zeroes scoring's
`reward_risk_quality` term. The target is now the next major level in trade
direction when one exists, falling back to 2R when none has confirmed yet. 2R
is the gate's floor, not a cap, so RR can run to `rr_cap` when a farther major
level supports it. `disagreement_flag_ratio` survives as a logged diagnostic
note, never a target adjustment. The exit spec's Part 3 carried the same stale
"nearer of" wording and is corrected too. The stop that RR depends on is now
defined per trigger in the exit spec's new Part 0 (the pattern's own
extreme).

---

## 7. ~~§20 engulfing fires on 12-15% of bars~~ — RESOLVED (spec §20)

§20's strength filter is purely RELATIVE: `body[i] >= 1.3 x body[i-1]`. Its
stated purpose is that "a marginal engulf of a tiny prior candle doesn't count
as a real signal — a small body engulfing an even smaller one is noise, not
conviction." But 1.3x of tiny is still tiny, so the rule catches *marginal*
engulfs while letting *small* ones straight through.

Measured on real 5m data over the 3-month window:

| symbol | bars flagged engulfing |
|---|---|
| MES | 14.5% |
| MGC | 12.1% |
| MET | 4.5% |

For comparison §7's rejection candle fires on ~1% of bars, because it carries
an ABSOLUTE floor (`body >= 0.25 x ATR`) alongside its ratio test. §20 has no
equivalent. A trigger present on one bar in seven is not evidence of anything,
and it carries a 0.80 base score — second only to three-tail and the
confirmation signal.

**Recommendation:** add an absolute body floor to §20, mirroring §7. Not
applied, because §20 is a defined spec rule and changing it is a spec decision,
not an implementation one.

**Resolved.** Spec §20 now requires both tests, and is explicit that the
combination is not optional: `body[i] >= 1.3 x body[i-1]` AND
`body[i-1] >= engulfing_min_prior_body` (default `0.50 x ATR(entry)`, config
key `engulfing.min_prior_body_atr_multiple`). The floor is on the PRIOR body —
what has to be meaningful is the body being swallowed.

The floor landed at 0.50 rather than the 0.10 first written into §20, and the
measurement behind that first value was corrected in the same pass. One-tick
dojis are a real contributor but not "the majority": 20.7% of MES firings,
3.2% MGC, 31.7% MET. The broader problem is frequency. Share of the
unfiltered count retained:

| floor (× ATR) | MES | MGC | MET |
|---|---|---|---|
| 0.10 | 79.3% | 73.5% | 92.1% |
| 0.30 | 34.5% | 29.1% | 43.7% |
| 0.50 | 13.8% | 9.1% | 18.7% |

At 0.10 engulfing still fired on 11.5% of bars against ~1% for §7 — the
one-bar-in-seven problem this item opened with, largely intact. 0.50 puts it in
the same order of magnitude as the other selective triggers. Still tunable, but
a trigger firing on most bars is not structurally functioning as a trigger
whatever a later backtest says, so the floor is a structural decision rather
than a Phase 4 one.

---

## 8. RESOLVED in code: a breakout is a transition, not a state

§4 defines a breakout as "a close beyond a marked level by at least a buffer".
Read as a per-bar predicate that makes every bar of an uptrend a breakout of
every level beneath it — verified directly: with price trading 104-108 and a
level at 100, all six bars returned `up_close == True`, and §8 then
manufactures a failed breakout from any oscillation.

§8 ("closes beyond a level, THEN within K bars closes back") and §9 ("confirmed
breakout ... price pulls back") both describe a discrete transition, so
`breakouts()` returns both readings: `up_close`/`down_close` as state, and
`up_break`/`down_break` as the transition event that §8 and §9 consume. The
first bar can never be an event, since there is no prior bar to transition
from.

---

## 9. RESOLVED in code: a Yellow Alert is a crossing, not "being beyond"

Same class of problem in §17. The spec says the piercing bar is "the first bar
where price crosses the level L — `high[P] > L`". Implemented as `high > L`
alone, a market trading above a level re-arms an alert on every bar the moment
the previous one resolves. On real MES 5m data against one static level that
produced 3,454 confirmations — one every five bars.

A crossing now requires the previous bar to have closed on the other side. Same
data, same level: 90 alerts, 34 confirmed / 54 failed / 2 expired. More
failures than confirmations, which is the expected shape for a mode whose whole
point is being stricter than §4.

---

## 10. RESOLVED in code: §11 momentum continuation is an event, not a state

Same class of problem as items 8 and 9, found by `scripts/plot_triggers.py`.
§11's level test was written as a bare predicate, so `close > minor_level`
stayed true for every bar that remained beyond the level and any later
strong-bodied, volume-expanded bar re-fired it.

The count was the tell, not the code reading. Over 66 sessions per instrument:

| trigger | firings per instrument |
|---|---|
| §11 momentum (before) | 2,232 - 4,379 |
| §7 rejection | 157 - 219 |
| §9 breakout/retest | 29 - 36 |
| §19 three tail | 35 - 457 |

An order of magnitude above every other selective trigger is a symptom, not a
sign that momentum is simply more common. §11 now shifts the level-beyond
state and fires on the transition, the identical pattern `breakouts()` uses.
The body, volume and trend filters apply to the transition bar. As with §4,
the first bar can never be an event.

---

## 11. RESOLVED in code: §10 range reclaim must match the boundary's side

`range_reclaims()` delegated to `failed_breakouts()`, which accepts a close
beyond the level in EITHER direction as the breakout leg. That is right for a
discrete S/R level, which can fail from either side, and wrong for a range
boundary, which cannot: a close *below* a range high without ever exceeding it
is ordinary trade inside the range, and the return above then scored as a
reclaim. Roughly 2x overcount on real data (MES 238 -> 116).

`range_reclaims(bars, edge, atr, params, side)` now takes the boundary's side
and is required, not inferred — nothing about a bare price says which edge it
is. `failed_breakouts()` gained an `only` parameter to restrict the leg, and
still defaults to both sides for §8.

---

## 12. RESOLVED in spec §2: "prior day" is the previous LIQUID session

**Config:** `params.yaml > prior_levels.liquidity_floor_ratio` (0.50),
`prior_levels.liquidity_median_window_sessions` (20)

MET trades through weekends. Over the same window it produces 93 trade dates
against 66 for the other six — 14 Saturdays and 13 Sundays carrying ~7.5% of
its volume, straight through the 16:00-17:00 halt its own config declares. A
shift-by-one therefore drew Monday's prior-day levels from Sunday's 81-144 bar
session instead of Friday's ~734 bar one, and every trigger keyed to those
levels inherited it.

Resolved as a general rule, not a MET-specific patch: a session qualifies as
someone's "prior day" only if its bar count reaches 50% of the rolling median
trade-date bar count; otherwise the search skips further back. Whether thin
sessions are skipped at all is decided (yes); the 50% is a tunable default.

**Still open, minor:** the spec says "rolling median" without fixing the
window. 20 sessions (~a trading month) is used and is a named config field. An
expanding median over all prior sessions is the parameter-free alternative;
either is defensible, and the choice barely moves the result for the six
weekday instruments. Worth settling when Phase 4 grid-searches the floor.

**Note:** MET's `session.globex_open` / `globex_close` / `maintenance_halt`
still describe a Sun 17:00 - Fri 16:00 week with a daily halt, which the bars
contradict. The fields are not read by the trade-date logic, so nothing is
currently wrong because of it, but they should be corrected or explicitly
marked as nominal before anything starts trusting them.

---

## 13. ~~Timeframe config is not wired to anything~~ — LANDED (signal_engine/timeframes.py)

**Config:** `params.yaml > timeframes.entry`, `timeframes.htf`,
`timeframes.daily`, `timeframes.three_tail`

Only `timeframes.htf` is read (by `structure.swings()`). The rest are set and
ignored: every detector runs on whatever frame its caller hands it, and callers
hardcode `"5min"` / `"1D"`. §19 is the visible case — its own spec timeframe is
10min, `candle_triggers()` runs it on the entry frame, and the two disagree
(MET: 457 firings on 10min).

This is deliberately NOT being patched into `triggers.py`. Deciding which
detector runs on which frame, aligning them causally, and reconciling their
outputs is multi-timeframe orchestration — the job the Signal Engine exists to
do, and the first real piece of that build rather than a fix folded into the
feature layer.

**Landed** as `signal_engine/timeframes.py`, the first piece of the Signal
Engine. `params.timeframes` now resolves into built, feature-attached frames;
detectors ask for a ROLE (entry / htf / daily / three_tail) instead of naming a
frequency, so Phase 4 can move a timeframe per instrument without touching
feature code. `time_count.timeframes` is built too, since it names frequencies
directly. `structure.htf_atr_at()` generalised to `structure.align_htf()` so
there is still exactly one implementation of the HTF-close alignment rule —
ATR was only its first caller, and S14's bias needed the same merge.

Frames carry what is well-defined at their frequency: ATR, anatomy and CLV
everywhere, volume expansion only on intraday frames, since S12's baseline is
session-matched or hour-of-day and neither exists on daily bars. An absent
column is safer than a meaningless one that something later trusts.

`scripts/plot_triggers.py` builds through it, which is the proof it works: all
counts unchanged except engulfing (the 0.50 floor), and S19 lands on 10min
because the config says so rather than because the script resampled it.

**~~Still open inside this~~ — RESOLVED before gates.py:** `candle_triggers()`
bundled S7/S19/S20 onto one frame, so it could not honour a `three_tail` role
that differs from `entry`. It had no production caller, so the bug was latent
until gates.py called it. It moved to `signal_engine/candles.py` and takes a
`TimeframeSet`: S7/S20 read the entry frame, and S19 reads the `three_tail`
frame and crosses back through `TimeframeSet.align_events()`. That method is
new, because `align()` forward-fills, which is right for a state like S14's bias
and wrong for an event: S19 would re-fire on every entry bar until the next
10min close, which is the §11 bug again. An event lands on exactly one entry bar,
the first at which its bar has closed, and it is dropped rather than carried into
the next session. The bound is the session, not a time window: a one-step window
dropped 81 of MET's 306 S19 bars during mid-session quiet spells when price had
not moved.

---

## 14. RESOLVED in spec §19 + code: two-sided §19 bars, and bare-side "tails"

`three_tail()` can emit a SHORT and a LONG event on the same bar. The old
`candle_triggers()` wrote them per bar with the last write winning, so every such
bar silently came out LONG. `signal_engine/candles.py` labels them `both`.
**Spec §19 now rules:** a two-sided bar is NO TRADE, excluded from the directional
signal, on the same principle as nullable `is_major`. gates.py treats `both` as
its own non-directional case and never falls through to a side.

**The degenerate input behind most of them is fixed at the shared anatomy
layer.** `wick >= ratio × body` passes for a wick of ZERO whenever the body is
zero (`0 >= 2.0 × 0`). So every doji "had" a dominant wick on its bare side, and
an O=H=L=C bar had one on both. The guard is broader than zero-range bars: a wick
must have length (`confirmation.wick_dominates()`), and both S7 and S19 go
through it. Real 10min data, before → after:

| | MCL | MES | MET | MGC | MNQ | MYM | SIL |
|---|---|---|---|---|---|---|---|
| S19 bars | 35 → 35 | 83 → 83 | **306 → 36** | 34 → 34 | 37 → 37 | 79 → 79 | 41 → 41 |
| two-sided | 0 → 0 | 9 → 9 | **151 → 0** | 4 → 4 | 2 → 2 | 8 → 8 | 0 → 0 |

MET's worst in-session delivery lag fell from 9h10m to 20min: that lag came from
the same degenerate bars, not a separate issue. S7 and S20 counts are unchanged
on every instrument. At defaults S7's body floor and CLV threshold already
excluded these bars, and S20 compares bodies with strict inequalities. Both are
pinned by tests that push the neighbouring tunables to 0, which a Phase 4 grid
could reach.

**What remains two-sided on liquid instruments is real wicks on BOTH sides of a
zero-body doji.** Any wick against a zero body is an infinite ratio, so the
ratio test cannot tell a 1-tick wick from a 3-point one. These are excluded as
no-trade under the §19 ruling above, which is the correct outcome. The open
question: should a tail also need a minimum wick length in ATR terms? That would
be a new tunable, so it is a spec decision.

**Minimum wick length — RESOLVED: none.** Swept on the 10min S19 frame (S19
bars, two-sided in brackets):

| Min wick | MES | MGC | MNQ | MYM |
|---|---|---|---|---|
| current (`> 0`) | 83 (9) | 34 (4) | 37 (2) | 79 (8) |
| 1 tick | 83 (9) | 34 (4) | 37 (2) | 79 (8) |
| 2 ticks | 83 (7) | 34 (4) | 37 (2) | 78 (8) |
| 3 ticks | 63 (2) | 34 (4) | 37 (2) | 70 (4) |
| 0.05 × ATR | 83 (9) | 34 (4) | 37 (2) | 78 (8) |
| 0.10 × ATR | 81 (7) | 33 (3) | 35 (2) | 73 (6) |
| 0.15 × ATR | 68 (4) | 30 (3) | 32 (2) | 51 (1) |

MCL, MET and SIL have no two-sided bars under any rule. Tick floors do not
carry across instruments: a typical 10min ATR is about 22 ticks on MES but 159
on MNQ. The leftover two-sided bars carry large wicks on both sides relative
to ATR, so they are genuine. Only 0.15 × ATR clears most of them, and it costs
18% of MES's S19 events and 35% of MYM's, mostly one-sided. The residual is
5-11% of S19 bars and already excluded from trading as no-trade under §19. A
floor would buy selectivity at the cost of clean events, and it would be one
more tunable to overfit, so none is added.

---

## 15. ~~Which confirmed major swings are live levels and targets?~~ — RESOLVED (spec §1 liveness)

**Affects:** gate 2 (a major swing is a marked level) and gate 4 (the S15
target is "the next major level in trade direction").

No spec section says when a confirmed major swing stops being a level.
`signal_engine/gates.py` currently counts every confirmed major swing in the
history, and at a typical trigger bar that is about 470 of them. At that
density almost any price is inside some old swing's test zone, and the "next
major level" is almost always a few ticks past entry. Real S7/S20 triggers,
MES / MNQ / MCL / MGC:

| Which major swings count | Gate 2 pass | Gate 4 pass | Median RR |
|---|---|---|---|
| all confirmed (current) | 90-95% | 3-8% | 0.10-0.18 |
| last 3 per side | 64-69% | 27-33% | 0.93-1.10 |
| last 5 per side | 70-74% | 19-27% | 0.74-0.82 |
| unbroken only | 30-42% | 61-63% | 2.41-2.84 |

"Unbroken" means no close beyond the swing since it confirmed: a swing high
price has since closed above is no longer overhead resistance. It adds no
tunable. A recency cap adds one (K) with no spec basis, and it performs worse
on both gates. `scripts/plot_triggers.py` used a recency cap only for
readable charts.

**Resolved: unbroken only** (spec §1). A confirmed major swing is live until a
bar closes beyond it. There is no recency window, and dead swings do not flip
sides (a deliberate v1 simplification). Implemented as
`structure.mark_swing_deaths()` / `live_major_swings()`, used by gate 2's
marked levels and gate 4's target.

---

## 16. RESOLVED in the scoring spec: Stage 2 gaps found building scoring.py

Each was measured on real data before being resolved (3,738 candidates then):

| Gap | Resolution |
|---|---|
| Reversal WITH the HTF trend had no context case (31% of reversals) | 1.0 |
| Exhaustion read at the decision bar: 1 of 1,559 by construction | the bar before the pattern, any configured timeframe |
| `abs(CLV)` credited closes against the trade (4.7%) | `max(0, CLV × direction)` |
| No magnitude for S8/S9/S10 (28%) | S9 = retest-bar rejection; S8/S10 = midpoint 0.5 |
| Three-tail magnitude off one bar saturates on dojis | mean of capped per-bar ratios |
| Does the traded level count itself in confluence? | yes, so an isolated level scores 1/3 |
| Volatility-fit rows silent on engulfing/S8/S10 | gate 5's reversal list |

Structural, no decision: S17 magnitude is always 1.0 (it resolves within
`K_confirm_max` bars or expires).

**S18 vs continuation, resolved:** S18 says to reduce confidence on momentum
in the exhausted direction, but continuation's directional context was fixed
at 1.0. Momentum fires into exhaustion in 6.4% of its candidates (136 of
2,133; 3.4% MNQ to 10.8% MGC), almost all intraday. Resolved as 0.6 context
for momentum only.

---

## 17. DEFERRED TO PHASE 4: three-tail ranks last despite the highest base score

With the scoring spec as first built, three-tail averaged 57.6, the lowest of
any trigger type, against momentum's 61.9, despite base scores of 1.00 vs
0.60. Its trigger-quality edge (+9.5 points) was cancelled by confirmation
strength (-10.1: quiet tail bars, volume 0.22 and CLV 0.30) and directional
context (-6.4: momentum's continuation context is a full 15 points by
construction). CLV-only confirmation is the first step. If three-tail stays
near the bottom, its confirmation needs a basis drawn from the cluster's own
tail character, since a single bar's volume or CLV measures the wrong thing
for a multi-bar pattern.

A separate, larger question: 94% of momentum setups score the full 15
points of directional context by construction. Whether fresh-but-unremarkable
continuation should ever score below 1.0 is left open.

**Measured after CLV-only (523cfb1):** three-tail averages 58.3, still 7th of
7, 3.2 points behind momentum.

**Cluster-level volume tested as a replacement basis; not implemented.** Both
the mean and the pooled volume ratio across the cluster's tail bars were
checked against the forward-outcome proxy, the same test the gate 3 sweep
used (263 setups, all resolved). Neither separates outcomes: rank correlation
with R is essentially zero (+0.000, p = 1.00; +0.001, p = 0.99), and the
high-volume third did no better than the low one. It would also have lowered
three-tail's confirmation further (0.24 vs CLV's 0.30).

**CLV-only stays as the basis.** It is the only basis tested with any signal:
top vs bottom third +1.70R (p = 0.015), with win rate rising 10% → 14% → 17%.
That signal is weak. The rank correlation is not significant (p = 0.16), and
the 0.015 does not survive correcting for four bases tested (about 0.06).

**Three-tail's own composite score does not order its outcomes at this
sample size either:** its middle third did best (+1.14R) and its high third
worst (-0.15R). This is a sample-fragility finding, not evidence that the
scoring design is wrong. **Five winning trades supply half of all positive R
across 263 setups at a 14% win rate**, so any tercile-based conclusion from
this dataset is weak evidence, not a verdict.

**Explicitly unresolved.** This does not establish that three-tail doesn't
deserve its base score. It is deferred to Phase 4, where a real fill model
and a multi-year window can test it with enough winning trades to mean
something (see docs/phase4_questions.md).

---

## 18. DEFERRED TO PHASE 4: MET rolls fall through to the calendar backstop

**Config:** `config/symbols/MET.yaml > rollover` (`rule: volume_crossover`,
`confirm_days: 2`, `calendar_backstop_days: 2`)

Recorded at the time only in commit `3d33d2b` (Phase 1 roll tuning), which
predates this file: MET's day-to-day volume swings keep resetting the
consecutive-day crossover streak, so its rolls fall through to the calendar
backstop instead of firing on the volume crossover by design. The proposed fix
was a smoothed volume comparison, deliberately not tuned blind on three months.

Re-measured on the current code: all 3 MET rolls are backstop rolls (06-24,
07-29, 08-26), and so is MCL's August roll (08-17). MGC and SIL roll by
crossover. The June backstop rolls on MES/MNQ/MYM (06-15) and MCL (06-16)
have a different, mundane cause: the data starts 06-12, leaving no room for a
two-day streak. They are a window-edge artifact, handled by documentation
alone, and self-resolving under a multi-year window, where only the first
roll meets the edge. This corrects `3d33d2b`'s claim that the index micros
rolled by crossover.

Tracked as docs/phase4_questions.md A11, to be validated with Phase 1's own
roll-quality metric (C7).

**Cause corrected (2026-09-26, found onboarding MBT).** The recorded cause is
wrong, and so is the proposed fix. On each of MET's three backstop roll days
its expiring contract still carried **61–72% of volume**, and **52–70%** the
day after. The incoming contract takes over only **on expiry day itself**
(76–91% on 06-26, 07-31, 08-28). Volume migrates at the last moment, so the
2-day crossover streak cannot complete before the 2-day backstop fires. The
weekend swings are real but secondary: one Sunday (07-26) briefly shows the
incoming contract ahead before weekday volume flips back. MBT, which rolls on
the same dates, shows the same pattern more sharply (80–84% still in the
expiring contract on roll day). Its 6 minority-contract sessions in three
months are exactly the Wednesday and Thursday before each Friday expiry.

So **a smoothed volume comparison is the wrong fix**: smoothing delays the
crossover further past the backstop. The real choice is between three
options. Roll the day before expiry (a 1-day backstop), which cuts the
minority-contract sessions from 2 per roll to 1. Accept 2 sessions per roll.
Or hold the expiring contract into its final day, which is what volume says,
but carries expiry and settlement risk.
