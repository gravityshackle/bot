# Open questions

Items 1 and 2 are resolved and kept for the record; 3-6 are live.
Items 7-9 were raised while building the trigger layer.



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

## 7. §20 engulfing fires on 12-15% of bars — the filter does not do what
its own rationale says

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
