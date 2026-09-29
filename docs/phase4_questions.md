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

## A. Questions that need outcomes, not just counts

(A11 is a data-construction question, not a signal one, but it decides which
contract's bars the whole series is built from, so it belongs in the same pass.)

| # | Question | Evidence so far | Source |
|---|---|---|---|
| A1 | **Does three-tail deserve its 1.00 base score?** It ranks 7th of 7 (58.3 mean) | No confirmation basis separates its outcomes; its own composite score does not order them either; 5 winners supply half of all positive R across 263 setups | open_questions #17 |
| A2 | **Do the Stage 2 weights hold?** The spec calls them a starting default to refit by logistic regression on labelled trades | Every setup's full component breakdown is already logged (`engine.run()`) for exactly this | scoring spec, Stage 2 intro and "Logging" |
| A3 | **Where does the score floor go?** Start permissive (50), raise where the win-rate/score relationship breaks | Not measured | scoring spec, "Using the score" |
| A4 | **Is the RR cap (4.0) right?** 59% of candidates score reward/risk quality at the cap, so the component only separates the 41% with RR in [2, 4) | Saturation measured; outcome value of RR above 4 not measured. **How it must be answered (decided 2026-09-28): per instrument first, then decide whether pooling is valid at all, never pooled by default.** The Part 1 sizing cap ($50) discards setups whose stop is too wide for the budget, so the survivors on some instruments are a narrow-stop-only sample by construction. **The discard rate depends on the price regime, not just the instrument** (corrected 2026-09-29 from the 5-year study; the figures first recorded here came from a 90-day check that turned out to be the most extreme stretch of the window). The budget buys a fixed number of ticks, and stops widen as prices and volatility rise. Discarded share of Stage 1 setups, whole window / in-sample (before 2024-09-12) / out-of-sample / last 90 days: MNQ 27 / 20 / 38 / 66%, MGC 21 / 5 / 42 / 60%, SIL 47 / 28 / 69 / 94%. MES, MYM, MCL, MET and MBT stay at 0-15% throughout. Pooled, the RR/win-rate relationship could reflect which setups survived sizing rather than anything about reward/risk quality. The same applies **across the in-sample/out-of-sample split** on MNQ, MGC and SIL: their survivors' makeup shifts between the two halves, so a difference there may come from sizing and not from the component. **SIL** stays in the dataset, but every SIL-derived figure is flagged low-sample/directional-only and never answers A4, or any other decision, on its own. Every sized SIL setup has a stop of at most 10 ticks ($50 at $5/tick). SIL's median stop went from 5-7 ticks when silver was around $23 (2021-2024 Q1, 16-32% discarded) to 25-79 ticks at $51-83 (2025 Q4 onward, 86-98% discarded), which leaves 642 filled out-of-sample trades against 1,981 in-sample. **At current prices SIL is effectively untradeable at this account size. That is a property of the price regime, not of the instrument**, so it would change if silver fell back or the risk budget rose. Whether SIL stays in the traded list is a separate decision, revisited after 4.4/4.5's findings | scoring spec §5; Phase 3 scoring run |
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
