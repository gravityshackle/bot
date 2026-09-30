# Phase 4 plan — backtest engine

Answers the questions in `docs/phase4_questions.md`, designed around its
section C. Signed off 2026-09-26, with the decisions in §5. **No data is pulled
until the exact per-request costs have been shown again and approved,
separately, immediately before the pull.**

The one ordering rule that shapes everything else: **validate A4 first.**
Phase 3's validation plots showed win rate falling as score rises (Q1 38% →
Q5 22%), tracking planned RR almost exactly. If reward/risk quality is really
rewarding worse trades once real fills exist, most of the rest of the tuning
order changes. So the first backtest deliverable is an answer to A4, not a
full grid.

---

## 1. The data pull

**Priced via Databento `metadata.get_cost`** (no data fetched), all seven
instruments, `GLBX.MDP3`, 1-minute bars plus contract definitions, window
ending at the current pinned `2026-09-12`:

| Window | Total | Most expensive single request |
|---|---|---|
| 2 years | $34.48 | MGC $8.10 |
| 3 years | $48.03 | MGC $10.76 |
| **5 years (recommended)** | **$73.23** | MCL $16.55 |

Per instrument, 5 years: MCL $16.55, MGC $15.38, MNQ $10.58, MES $9.87,
SIL $9.33, MYM $7.81, MET $3.69. Every request stays under the existing
$25 per-request ceiling, so the ceiling does not need raising. MET launched in
late 2021, so its "5 years" is about 4¾; MCL's history starts mid-2021.

**Why 5 years:** A1 is limited by *winning trades*, not setups. Three months
gave 263 three-tail setups at a 14% win rate, and 5 winners carried half of
all positive R. Five years is about 20× that: roughly 700 three-tail winners
before the new stale-entry gate, and thousands per trigger type overall. That
is enough to compare score bands and still hold out a validation window for
walk-forward (for example, 3 years in-sample and 2 years out-of-sample).
Three years would give about 420 three-tail winners and a thin hold-out.

**Operational notes, fixed before pulling:**

- The cache key includes the window's start and end, so a new window is a
  **full re-fetch**, not a delta. `config/data.yaml`'s comment claiming
  otherwise is wrong and gets corrected. Re-fetching the three months already
  held costs about $5 and is included in the figures above.
- The pull runs one instrument at a time, each writing to disk as it
  completes. A killed process loses only that instrument, and a re-run reads
  finished instruments from cache for free.
- Every billed request still needs `confirm=True` on your explicit approval.
  This plan's sign-off is not that approval: I will show the exact per-request
  cost again immediately before the pull.

**Validation before anything reads the new data:** re-run Phase 1's checks
over 5 years: tick grid, seam report, and roll-map review. Rolls go from
1–3 per instrument to roughly 20 (quarterly) and 60 (monthly). Re-derive
empirical RTH from the longer sample. Fix MET's stale session fields (C8).

## 2. The fill model

Built from the exit spec (Parts 0–4); every number is a config field.

**Entry (Part 2):**
- A limit order is placed at the decision bar's close, priced by the Part 2
  table.
- It stays live for `K_entry_expire` = **3** bars (the conservative end of
  the spec's 3–5).
- It fills only if a later bar **trades through** the limit by at least one
  tick. A touch is not assumed to fill, since queue position is unknowable
  from OHLC, and assuming it would inflate the fill rate of every level
  entry.
- It is cancelled if a bar closes beyond the stop before the fill
  (invalidation before fill), or when the time-in-force expires.
- **Intrabar path from 1-minute bars.** Signals are decided on 5-minute bars,
  but fills are walked on the 1-minute bars inside them. That resolves most
  "did the stop or the target come first" cases. Where one 1-minute bar
  touches both, the adverse outcome is assumed.

**Sizing (Part 1):** `risk = min(equity × 1%, $50)`, contracts =
floor(risk / (stop ticks × tick value)), capped at `max_contracts` (1). A
setup that sizes to 0 contracts is discarded, never rounded up. MET's $0.05
tick value means the cap, not the risk budget, binds there.

**Exits (Part 3):**
- A stop-market order at the stop fills at the stop plus slippage, or at the
  open if price gaps through it.
- A limit order at the target fills only on trade-through.
- At 1R the stop moves to breakeven.
- Once a swing confirms in the trade's favour, the target is cancelled and
  the stop trails behind each newly confirmed swing. It uses the same causal
  swing confirmation as the gates, so the trailing logic cannot see a swing
  before it confirms.

**Day boundary: flatten, don't carry.** Any open position is closed at a
market price at its instrument's own day boundary (`day_boundary` in each
symbol config). Stop protection is weakest in thin overnight liquidity, and
this keeps every position inside one trading day as the Part 4 risk state
machine already defines it. Carrying overnight is a variant to test later
with evidence, not the default.

**Costs** (`config/costs.yaml`, per filled contract per side, on entry AND
exit):
- **Commission:** IBKR low-volume tier: $0.25 for the E-micros
  (MES/MNQ/MYM/MCL/MGC/SIL), $0.20 for MET, $0.85 for MBT. E-micros count as
  1/10 of a contract toward IBKR's volume tiers, so this tier is expected to
  hold.
- **Exchange/regulatory pass-through:** a $0.50 placeholder, deliberately
  conservative. It gets replaced from IBKR statement fills once a real sample
  exists.
- **Slippage: market orders only.** Entries and targets are limit orders with
  trade-through fills, so they fill at their price. The stop-market exit, the
  daily-breaker flatten and the day-boundary flatten are market orders, and
  each slips **1 tick** to start. Tick values come from the IB-verified
  symbol configs, not from `costs.yaml`. 2 ticks for MET/MBT and a 2–4× thin-
  session multiplier are sensitivity cases, not defaults.
- Every fill in the backtest records its reference price, order price and
  fill price, so realized slippage can later be measured by symbol, session
  and order type, and the placeholders calibrated from paper/live fills.

## 3. Two levels of backtest

1. **Setup-level study.** Every Stage 1 candidate is simulated independently
   through the fill model, with no portfolio or daily limits. Each row is one
   setup's realized outcome (filled or not, R, exit path) joined to its six
   score components. This answers A1–A6, the "is the signal any good"
   questions, with the most data, because it isn't thinned by 2 trades a day.
2. **Portfolio backtest.** The risk engine's Part 4 daily state machine
   (2 trades/day, stop after 2 losses, −$300 continuous breaker, +$100 target,
   one position, the correlated index group) runs on one account-wide clock.
   Setups are taken in score order. This answers "what would the account have
   done", and it is where gate 6 stops being `not_evaluated`.

## 4. Build order

Each step is test-first and has its own commits and real-data check, the
same discipline as Phases 1–3.

| Step | What | Output | Validates |
|---|---|---|---|
| **4.0** | Data pull (after your spend approval), cache-comment fix, Phase 1 validators over 5 years, MET session fields | clean 5-year series for all seven | C1, C8 |
| **4.1** | `execution/simulated_execution.py`: limit entry, time-in-force, invalidation before fill, trade-through fills on the 1-minute path, stop/target fills, costs | fill simulator, unit-tested on hand-built paths | C2 |
| **4.2** | Exit state machine: breakeven at 1R, causal trailing | trade lifecycle | exit spec Part 3 |
| **4.3** | Setup-level study harness; per-trade records with the full score breakdown | one row per setup, outcome plus components | C3, C4 |
| **4.4** | **A4 first**, per instrument before any pooling (see A4 in phase4_questions.md; SIL flagged low-sample/directional-only): reward/risk quality and RR band vs realized win rate, expectancy and R distribution, in-sample and out-of-sample separately | a yes/no answer on whether the component rewards better or worse trades | A4 |
| 4.5 | Re-plan the rest of Phase 4's order from 4.4's answer (see **4.5 planning inputs** below). Then A1 (three-tail), A5, A6, and the weight refit (A2) with the score floor (A3) | revised tuning order | A1–A3, A5, A6 |
| 4.6 | `risk_engine/controls.py` and `sizing.py`: the Part 4 state machine, plugged into gate 6; `backtest/engine.py` portfolio mode | account-level results | gate 6, Part 4 |
| 4.7 | Performance work (vectorise the gate path), then walk-forward over section B, plus A8–A11 | tuned per-instrument parameters | B, C5, C6, A8–A11 |
| 4.8 | `backtest/report.py`: metrics per trigger type, score band and instrument, with R distributions | the Phase 4 report | C4 |

Steps 4.1–4.4 answer A4 without needing the risk engine at all, which is why
it comes later.

### 4.5 planning inputs (from 4.4, 2026-09-29)

4.4's answers are in `docs/phase4_questions.md`: A4 (answered), F1 (no trigger
has positive expectancy before costs) and F2 (target placement is now the
primary suspect). 4.5 plans around these, and specifically:

1. **Target selection and stop width are two distinct candidate fixes, to be
   evaluated separately, not as one "fix target placement" change.** A
   target's distance in R is (distance to the nearest live major level) /
   (stop width), and F2 found that for most triggers stop width drives it
   more directly than level distance does. Changing how the target is chosen
   (which level, or a cap on its distance) and changing the stop's width or
   basis (the pattern extreme plus 0.10 × the entry-timeframe ATR) move RR,
   fill and outcome in different ways.
   - Evaluate each on its own first, against the current engine, per
     instrument, in-sample then out-of-sample.
   - Only then test them together, so any gain can be traced to one lever.
2. **Three-tail is a separate track.** Its problem is upstream of both
   scoring and target placement. Three independent findings converge on
   its detection logic itself being suspect:
   - A1: the Phase 3 ranking;
   - F1: the worst signal R in both periods, despite the highest base score;
   - F2: the smallest moves of any trigger, and 8 R targets that come from
     its tight stops, not far levels.

   It is a candidate for revisiting level spec §19's actual trigger
   definition in a later phase. That stays separate from whatever general
   target/stop fix 4.5 lands on for the other six trigger types. It is
   neither judged by that fix nor used to tune it.
3. **Open, not blocking: gate-4 survivorship.** Gate 4 drops setups whose
   nearest level is under 2R, so the survivors lean toward far levels by
   construction. The study saved only Stage 1 setups, so measuring this
   needs an engine run that also logs gate-4 failures. It is done when such
   a run is needed for something else, or when a target/stop candidate's
   evaluation depends on it.

## 5. Decisions (signed off 2026-09-26)

| Decision | Resolution |
|---|---|
| Window | **5 years** ($73.23 as priced). The spend is approved separately at pull time, against exact per-request figures shown again |
| `K_entry_expire` | **3 bars** |
| Fill rule | **Trade-through by one tick**; a touch does not fill |
| Overnight | **Flatten at each instrument's own day boundary**; carrying is a later variant |
| Costs | IBKR low-volume commission + $0.50 placeholder pass-through, per filled contract per side, entry and exit (`config/costs.yaml`) |
| Slippage | **Market orders only**, 1 tick to start |

**One interpretation to confirm:** slippage was specified for stop-market
exits. The day-boundary and daily-breaker flattens are market orders too, so
by the same reasoning they are included. Say so if they should be excluded.

## 6. MBT (Micro Bitcoin): a separate track, not in this pull

MBT joins the project through its own Phases 1–3 before any backtest, as
`docs/symbol_spec_and_repo_structure.md` records. Its contract spec (tick
size, multiplier, contract months) comes only from IB's `reqContractDetails`,
the same discipline that caught the SIL error. Architecturally it follows
MET: continuous session, no RTH/ETH split, its own `day_boundary`,
`volume_crossover` rollover with `highest_volume` candidates. Its Phase 1
needs its own short Databento pull, priced and approved separately. Once it
is clean through Phase 3, it either joins the Phase 4 pull or follows as its
own addition.
