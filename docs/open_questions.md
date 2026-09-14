# Open questions blocking Phase 2 / Phase 3

Tracked here rather than in chat so they stay attached to the code. Each names
the config key that is parked on `UNRESOLVED` or `UNDEFINED_IN_SPEC`, so
nothing silently picks a default.

---

## 1. "Major" swing depth reference — BLOCKS `structure.classify_swings()`

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

**Recommendation: A.** It is the only reading knowable at pivot-confirmation
time, so it keeps targets computable when the R:R gate runs.

---

## 2. Engulfing is scored but never defined — BLOCKS `triggers.engulfing()`

**Config:** `params.yaml > engulfing.status`

The scoring spec gives engulfing a base score of **0.80** (between
breakout/retest at 0.85 and single rejection at 0.70), lists it in Stage 1
gate 1, and the repo structure assigns it to `features/triggers.py`. The
level-detection spec attributes it to *"the prior message"* — a document not
in this repo. So the only trigger with a scoring weight and no definition.

Needs: body-overlap rule (does the engulfing body have to cover the prior
body only, or the whole prior range including wicks?), whether a same-
direction close is required, and any minimum body size.

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
