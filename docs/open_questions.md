# Open questions

Items 1, 2 and 7 are resolved and kept for the record; 3-6 are live.
Items 7-9 were raised while building the trigger layer; 10-12 by the Phase 2
validation run (`scripts/plot_triggers.py`) and are resolved in spec and code.
Item 13 is the first piece of Signal Engine work, deliberately not patched
into the feature layer.



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



## 4. Neutral-HTF policy contradicts itself — Phase 3



**Config:** `params.yaml > trend.neutral_policy`



Level spec §14 says a neutral/chop HTF means the bot *"skips new setups"* — an

explicit no-trade filter. Scoring §4 assigns reversal triggers **0.5** when

*"HTF is neutral/chop"*, which presupposes those setups reach scoring at all.

Both cannot hold. Either neutral is a hard gate (and the 0.5 branch is dead

code), or it is a scoring penalty (and §14's "skips" is wrong).



---



## 5. Hard gate 3 overlaps gate 1 — Phase 3



Scoring Stage 1 gate 3 requires "volume expansion (§12) **or** CLV threshold

(§13) met, per the trigger type's own requirement." But §7's rejection candle

already requires `CLV >= 0.6` to fire at all, and §11's momentum continuation

already requires volume expansion. So for those two, gate 3 is satisfied by

definition the moment gate 1 passes.



Question: does a rejection candle need volume expansion *in addition* to its

own CLV, or does its internal CLV discharge gate 3? Materially changes trade

frequency.



---



## 6. §15 target wording — Phase 3, cosmetic but worth settling



§15 sets the target to the "next major level in trade direction, or `2.0 ×

risk` **minimum**, whichever is **nearer**." Taking the nearer of the two means

that when the next major level is closer than 2R, the target lands *below* 2R

and §16 then rejects the setup for `RR < min_reward_risk`. That is coherent

behaviour, but "minimum" describes the opposite of what the rule does.



`targets.disagreement_flag_ratio: 1.5` supplies the threshold §15 asks for

("flag if the two disagree by a lot") but never gives.



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

## 13. Timeframe config is not wired to anything — Signal Engine, first piece

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
