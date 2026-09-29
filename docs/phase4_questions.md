# Phase 4 — questions the backtest must answer

Everything the build so far has explicitly deferred to backtest evidence,
collected in one place. The goal is for the backtest engine's data window,
fill model and metrics to be designed around answering all of these at once,
rather than discovering them one at a time mid-backtest. Each item names
where it is recorded.

Phase 1–3 evidence came from **three months** of data
(`config/data.yaml > window.months: 3`, pinned end `2026-09-12`) and a
**forward-outcome proxy** that assumes a fill at the signal price and counts a
bar touching both stop and target as a loss. Every "measured" figure below
carries both of those limits.

---

## F. Findings that re-plan Phase 4

### F1. No trigger has positive expectancy before costs (found 2026-09-29, Phase 4.4)

**This is the situation 4.5 re-plans around.** Before slippage and fees:
- **In-sample, zero of 8 instruments show positive signal R in either trigger family.** All 16 instrument-family cells have a negative mean, and 15 of them have an interval wholly below zero. The exception is MYM's reversal types, -0.045 [-0.118, +0.036].
- **Out-of-sample, exactly one instrument per family is positive, and neither is distinguishable from flat:** MGC momentum +0.064 [-0.047, +0.182] and SIL reversal types +0.025 [-0.334, +0.497], SIL being directional only.
- **No instrument-family cell is significantly positive in either period.**
- **Three-tail, the highest base-scored trigger in the system (1.00; momentum is the lowest at 0.60), is the single worst performer**: -0.330 R in-sample and -0.322 R out-of-sample.

So it is not one trigger dragging down an otherwise viable set: removing momentum leaves the reversal types just as negative. It is not cost drag alone, because cost drag comes on top of a negative signal. And the trigger quality ranking the score is built on runs opposite to outcome at its top.

**Measured on.** Every filled trade from a sized setup in the 5-year setup study (commit 2fd10a8): 44,683 trades across all 8 instruments, split at 2024-09-12. Discarded and unfilled setups have no outcome. The unit is **signal R**: `r_gross` with each market exit's modelled slippage (1 tick, from the study's own cost model) added back. Limit exits and entries carry none. Gap-through losses stay in, because a stop that a bar opens beyond fills from the open, and that is the market, not a cost. The identities are checked on all 8 instruments' data:
- entries fill at their order price;
- `r_gross` = (exit fill - entry fill) / risk;
- risk dollars = risk points × point value;
- `r_gross` - `r_net` = fees / risk dollars;
- non-gapped exits slip exactly the modelled tick.

(Correction: signal R was first computed by pricing the exit at `exit_reference`. For a gapped stop that is the stop price, so gap losses were being added back as if they were slippage: 0.097 R per trade on MET, where 31% of stops gapped, and 0.043 R on MBT. The exact definition above replaced it before this entry was written.)

Mean R per filled trade, pooled across instruments (95% bootstrap interval on signal R):

| Family | Period | Trades | Share | Signal R | Gross | Net | Instruments with positive signal R |
|---|---|---|---|---|---|---|---|
| Momentum | IS | 13,917 | 51% | **-0.196** [-0.223, -0.168] | -0.309 | -0.783 | 0 of 8 |
| Momentum | OOS | 9,468 | 55% | **-0.086** [-0.129, -0.036] | -0.167 | -0.748 | 1 of 8 (MGC) |
| Reversal types (5) | IS | 13,352 | 49% | **-0.148** [-0.174, -0.124] | -0.217 | -0.442 | 0 of 8 |
| Reversal types (5) | OOS | 7,769 | 45% | **-0.129** [-0.162, -0.095] | -0.180 | -0.478 | 1 of 8 (SIL, directional only) |
| Breakout/retest | IS / OOS | 99 / 78 | 0.4% | -0.273 / -0.137 (intervals span 0) | | | |

By trigger, signal R in-sample / out-of-sample:
- failed_breakout: -0.084 / -0.059 (the least negative)
- range_reclaim: -0.110 / -0.165
- rejection: -0.128 / -0.113
- engulfing: -0.131 / -0.094
- momentum: -0.196 / -0.086
- three_tail: **-0.330 / -0.322**, the worst in both periods (bears on A1)

Every trigger's interval excludes zero on the negative side in-sample. Out-of-sample, all do except engulfing, whose interval reaches +0.044. The balanced figure (mean of per-instrument means) tells the same story.

**The momentum hypothesis, tested and not supported.** The hypothesis was that momentum (52% of filled trades, directional context saturated at 1.0 on 93% of them against 33-46% for the reversal types) drags down a viable signal. The momentum-minus-reversal gap in signal R:
- in-sample: -0.048 R [-0.086, -0.008];
- out-of-sample: +0.043 R [-0.015, +0.105].

So it is not replicated. The per-instrument sign is mixed (in-sample 3 of 8 favour momentum, out-of-sample 5 of 8). With breakout/retest counted on the reversal side, or with SIL excluded, it is unchanged. **After fees** momentum is clearly worse, by -0.27 to -0.34 R in both periods. That is a tight-stop cost effect, not signal: momentum's stops are narrower (16 against 24 ticks at the median), so fees cost it 0.52 R against 0.25 R, and MET drives most of it (without MET the net gap is -0.11 R).

**What it does not say.** Signal R is the outcome of the whole trade plan: the limit entry, the stop, the target, breakeven at 1R, trailing, and the day-boundary flatten. It does not separate the entry triggers from the exit logic. Win rates before costs are 20% (momentum) and 27-30% (reversal types). Whether the problem is where trades enter or how they exit is the first thing 4.5 has to establish, for example from each trade's maximum favourable excursion against its target, before choosing what to change.

Source: `scripts/edge_by_trigger.py`, `analysis/edge.py`, results in `cache/analysis/edge_by_trigger.json`.

---

## A. Questions that need outcomes, not just counts

(A11 is a data-construction question, not a signal one, but it decides which
contract's bars the whole series is built from, so it belongs in the same pass.)

| # | Question | Evidence so far | Source |
|---|---|---|---|
| A1 | **Does three-tail deserve its 1.00 base score?** It ranks 7th of 7 (58.3 mean) | No confirmation basis separates its outcomes; its own composite score does not order them either; 5 winners supply half of all positive R across 263 setups | open_questions #17 |
| A2 | **Do the Stage 2 weights hold?** The spec calls them a starting default to refit by logistic regression on labelled trades | Every setup's full component breakdown is already logged (`engine.run()`) for exactly this | scoring spec, Stage 2 intro and "Logging" |
| A3 | **Where does the score floor go?** Start permissive (50), raise where the win-rate/score relationship breaks | Not measured | scoring spec, "Using the score" |
| A4 | **Is the RR cap (4.0) right?** 59% of candidates score reward/risk quality at the cap, so the component only separates the 41% with RR in [2, 4) | **ANSWERED 2026-09-29 (Phase 4.4): the component carries no measurable information about outcomes, and there is no evidence the cap costs anything.** Verdicts use `r_gross` (after slippage, before fees; decided 2026-09-29, replacing `r_net`, because on MET the fees exceed the risk on the median trade and the fee share grows with RR band). Signal R and `r_net` are reported beside it. Per instrument, in-sample and out-of-sample:<br>- **Component** (R per live RR band, 2-4): flat on all 8 instruments, in both periods, on every cost layer. On MNQ, MGC and SIL it is also flat under both stricter checks, with the strata covering 100% of OOS. The 95% intervals are about ±0.1 R per band, so an effect larger than that is ruled out.<br>- **Cap** (RR ≥ 4 against [3, 4)): no instrument shows RR above 4 doing better. MES, MNQ, MYM and MBT show no difference in either period. MCL and MET are worse in-sample only (not replicated). MGC and SIL are **unresolved: sizing**: the in-sample "worse" vanishes under both stricter checks, and on SIL slippage also moves it.<br>- **Score quintiles**: no instrument's composite score orders outcomes consistently. MES, MNQ, MYM, MCL and SIL are flat. MBT (in-sample down), MET (out-of-sample down) and MGC (out-of-sample up) are not replicated. This carries into A2/A3.<br>**Cost drag, a finding of its own:** MET's fees are 133-147% of risk at the median and exceed the whole risk on 63-70% of trades (net -2.5 to -3.3 R), so at 1 contract MET is untradeable, as SIL is at current prices. SIL has the largest slippage, 0.19-0.24 R per trade. See F1 for the larger result behind all of this: expectancy is negative before costs everywhere. Results: `scripts/a4_report.py` (commit 5f19574 on), `cache/analysis/a4/`. Earlier evidence: saturation measured; outcome value of RR above 4 not measured. **How it must be answered (decided 2026-09-28): per instrument first, then decide whether pooling is valid at all, never pooled by default.** The Part 1 sizing cap ($50) discards setups whose stop is too wide for the budget, so the survivors on some instruments are a narrow-stop-only sample by construction. **The discard rate depends on the price regime, not just the instrument** (corrected 2026-09-29 from the 5-year study; the figures first recorded here came from a 90-day check that turned out to be the most extreme stretch of the window). The budget buys a fixed number of ticks, and stops widen as prices and volatility rise. Discarded share of Stage 1 setups, whole window / in-sample (before 2024-09-12) / out-of-sample / last 90 days: MNQ 27 / 20 / 38 / 66%, MGC 21 / 5 / 42 / 60%, SIL 47 / 28 / 69 / 94%. MES, MYM, MCL, MET and MBT stay at 0-15% throughout. Pooled, the RR/win-rate relationship could reflect which setups survived sizing rather than anything about reward/risk quality. The same applies **across the in-sample/out-of-sample split** on MNQ, MGC and SIL: their survivors' makeup shifts between the two halves, so a difference there may come from sizing and not from the component. **For MNQ, MGC and SIL an in-sample/out-of-sample trend is only reported once it passes a stricter check (decided 2026-09-29).** Stop width is measured in multiples of the previous completed session's daily ATR(14), since the tick cap is the same in both halves and only the market around it changes. The survivors' score, RR and ATR-scaled stop width are compared across the split. The in-sample trend is then recomputed twice. First, restricted to survivors inside the ATR-scaled stop range (5th-95th percentile) that out-of-sample survivors cover. Second, stratified: RR bands are compared within out-of-sample stop-width decile slices, and the slices are weighted to out-of-sample's mix. Restriction alone does not remove the confound, because if high-RR setups have wider stops inside the range too, the link survives it. Stratification does remove it. The share of out-of-sample trades whose slice has in-sample trades is reported alongside. If its direction holds under both, and agrees with the raw in-sample and out-of-sample results within their confidence intervals, it can be trusted. If it flips or vanishes, that instrument's A4 answer is **unresolved**, a finding in its own right. MES, MYM, MCL, MET and MBT are reported directly, with the same distribution comparison shown as a control. **SIL** stays in the dataset, but every SIL-derived figure is flagged low-sample/directional-only and never answers A4, or any other decision, on its own. Every sized SIL setup has a stop of at most 10 ticks ($50 at $5/tick). SIL's median stop went from 5-7 ticks when silver was around $23 (2021-2024 Q1, 16-32% discarded) to 25-79 ticks at $51-83 (2025 Q4 onward, 86-98% discarded), which leaves 642 filled out-of-sample trades against 1,981 in-sample. **At current prices SIL is effectively untradeable at this account size. That is a property of the price regime, not of the instrument**, so it would change if silver fell back or the risk budget rose. Whether SIL stays in the traded list is a separate decision, revisited after 4.4/4.5's findings | scoring spec §5; Phase 3 scoring run |
| A5 | **Should fresh-but-unremarkable continuation always score 1.0 context?** 94% of momentum setups get the full 15 points by construction, and momentum is 57% of candidates | Only momentum into exhaustion (6.4%) is reduced (0.6) | open_questions #17; scoring spec §4 |
| A6 | **Is the disagreement flag threshold (1.5) informative?** It fires on 70–75% of trade plans | Diagnostic only; never moves the target | exit spec Part 3; `targets.disagreement_flag_ratio` |
| A7 | **Should broken levels flip polarity?** v1 drops a dead swing entirely | Deliberate v1 simplification | level spec §1; exit spec "NOT built into v1" |
| A8 | **Buffer breakout (§4/§9) or confirmation signal (§17), per symbol?** The spec says to A/B them; only `buffer` mode has been run | §17's magnitude is also structurally always 1.0. **Validation gap (2026-09-28): the fill simulator (4.1) and exit state machine (4.2) have been checked on real candidates in `buffer` mode only.** Of 16,213 real candidates across all 8 instruments, none were §17 confirmation signals, so §17 trades have only synthetic unit and mutation coverage (its Part 0 stop basis, pierce bar through resolve bar, is pinned in `test_part_0_pattern_bars_per_trigger`). Before any mode comparison or the final report treats the two modes as equally validated, re-run the 4.1/4.2 real-data checks with `breakout.mode` set to the confirmation signal | level spec §17 and "integrating" note; scoring spec §1 |
| A9 | **Is the gap threshold too loose for the RTH instruments?** It fires on 58% of MES sessions vs 4–9 gaps on the continuous four, and gap edges feed level confluence | Deliberately not re-thresholded blind | level spec §3 |
| A10 | **Which entry/HTF timeframe pairing per instrument?** Roles are config (`timeframes.*`), built to be grid-searched | Only 5min/1h/10min/1D has been run | params.yaml `timeframes`; config/data.yaml; open_questions #13 |
| A11 | **How should MET's (and MBT's) roll be timed? DECIDED 2026-09-26: a 1-day backstop for MET and MBT only; extended to MCL 2026-09-27 on the 5-year pull (lead only on the last session before expiry; minority sessions 51 → 18); MNQ checked, stays at 3**, not expiry day (settlement risk). *Diagnosis: volume migrates only on expiry day; it is not streak noise, and smoothing would delay it further (open_questions #18). Phase 4 still re-measures roll quality over 5 years (C7).* Originally recorded as: should MET's roll use a smoothed volume comparison? MET's day-to-day volume swings keep resetting the consecutive-day crossover streak (`rollover.confirm_days: 2`), so its rolls fall through to the calendar backstop instead of firing on the volume crossover by design. The fix proposed in Phase 1 was a smoothed volume comparison, deliberately not tuned blind on three months | Re-measured on the current code: all 3 MET rolls are backstop rolls (06-24, 07-29, 08-26, each 2 days before expiry), and so is MCL's August roll (08-17); MGC and SIL roll by crossover. The June backstop rolls on MES/MNQ/MYM (06-15) and MCL (06-16) are a window-edge artifact: the data starts 06-12, leaving no room for a two-day streak | Recorded only in commit `3d33d2b` (Phase 1 roll tuning), never in open_questions; `config/symbols/MET.yaml > rollover` |
| A12 | **How should thin-liquidity sessions be handled: in the stop buffer, in slippage, or both?** One liquidity effect shows up in two places. The stop buffer is `stops.buffer_atr_multiple` (0.20) × ATR(entry) with no floor in ticks, and overnight ATR is small, so stops get very tight. Meanwhile market-order slippage has a 2–4× thin-session multiplier that was named as a sensitivity case but never defined. The fill model refuses any value other than 1.0 because no doc says which sessions count as thin. Decide one definition of "thin" (by session, by volume or by ATR regime) and test the buffer (possibly with a minimum in ticks, like the breakout buffer's `buffer_min_ticks`) and slippage against it together | Phase 3 chart review (2026-09-26): 23% of resolved candidates stop out within 3 bars; 1–3-point stops appear on MES and MET overnight. Proposed for the Phase 4 grid in chat then, but never recorded until now. Slippage is 1 tick everywhere; a multiplier other than 1.0 raises (`execution/simulated_execution.py`, commit `7e85087`) | exit spec Part 0 and §15 buffer; `config/params.yaml > stops`; `config/costs.yaml > slippage.thin_session_multiplier`; phase4_plan.md §2 |

## B. Parameters to grid-search / walk-forward per instrument

The level spec's own tuning table, as named config fields
(`sensitivity:` marks in `config/params.yaml`):

- **HIGH:** `swings.pivot_n_ltf`, breakout buffer, `volume_expansion.multiplier`,
  `targets.min_reward_risk`, `confirmation_signal.k_confirm_bars`,
  `three_tail.cluster_tolerance_atr_multiple`,
  `engulfing.min_prior_body_atr_multiple`
- **MEDIUM:** `time_count.exhaustion_threshold`, `three_tail.lookback_bars`,
  `three_tail.min_tails_required`, `engulfing.strength_multiplier`,
  prior-day liquidity floor (50% of rolling median), and that median's window
  (20 sessions vs expanding; open_questions #12)
- **Scoring:** the six weights, base scores, context values and volatility-fit
  values in `config/scoring_weights.yaml` (A2)

The engulfing floor (0.50 × ATR) is a *structural* floor, not only a tunable:
below it the trigger fires on most bars (open_questions #7).

## C. What this implies for the backtest design

1. **Data window: multi-year, not three months.** A1 is limited by the number
   of winning trades, not setups (14% win rate). The window has to be sized so
   each trigger type, and each score tercile within it, carries enough *winners*
   to compare. `config/data.yaml` is pinned at 3 months by design; widening it
   is a billed Databento pull and needs explicit `confirm=True` approval.
2. **A real fill model.** Every Phase 3 outcome figure assumes a fill at the
   signal price. The exit spec already defines what to simulate: limit entries
   only (Part 2), the entry time-in-force, cancellation when the stop is hit
   before the fill, and the exit state machine (Part 3: breakeven at 1R,
   trailing after a confirmed swing). Unfilled limit orders matter directly
   for A1 and A4, since far targets and wide stops fill differently.
3. **Per-trade records carrying the full score breakdown.** A2 and A3 need
   the outcome (R multiple, exit path) joined to the six components that
   `engine.run()` already logs. Keep the row-level log as the unit of record.
4. **Metrics per trigger type and per score band, per instrument:** win rate,
   average R, R distribution (not just the mean — A1's fragility came from a
   heavy right tail), plus how many trades a window actually yields after the
   risk engine's daily limits (2 trades/day, stop after 2 losses).
5. **Walk-forward, not one in-sample fit.** Section B is a large grid, and the
   spec warns against overfitting; tune on one window and validate on the next.
6. **Runtime.** The engine takes 70–100 s per instrument per run (gate
   evaluation ~25 ms per setup). A grid over section B multiplies that, so the
   gate path likely needs vectorising before a full grid is practical.
7. **Roll quality is measurable, so measure it.** Phase 1 scored roll
   parameters on "days spent holding a contract with under 50% of that root's
   daily volume" (commit `3d33d2b`). Re-run that metric under any smoothed
   comparison for A11, over the multi-year window, where the start-of-window
   edge affects only the first roll.
8. **Resolve before trusting MET or MBT results** (MBT added 2026-09-26: its
   provisional session fields copy MET's and contradict its bars the same
   way — it trades 7 days with a weekly ~9-hour Friday-evening break and no
   weekday halt). MET's `session.globex_open` /
   `globex_close` / `maintenance_halt` fields still describe a weekday week
   with a daily halt, which its bars contradict (open_questions #12). They are
   not read by the trade-date logic today, but the risk engine's session clock
   will read them.

## D. Not in this list, and why

- **Minimum wick length for three-tail tails:** resolved, not deferred. No
  floor is the settled default (open_questions #14, level spec §19), because
  the only floor that clears the residual two-sided bars costs 18–35% of clean
  events and the two-sided exclusion already keeps them out of trades.
- **Channels as a confluence type:** still undefined in any spec
  (open_questions #3). That is a spec question, not a backtest one.
- **The $100/$300 daily figures:** reviewed manually by design, never
  auto-tuned (exit spec, "NOT built into v1").
