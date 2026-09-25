# Symbol Spec Table & Repo Structure — v1

## Part 1 — Symbol Spec Schema

Each instrument gets its own config file. **Tick size/value/multiplier are
exchange-fixed facts — hardcode these as verified constants.** Margin is
broker/volatility-dependent and changes without notice — **fetch it from the
Interactive Brokers API at runtime**, never hardcode it here, or you'll silently
trade on stale numbers after CME raises margins overnight.

```yaml
# config/symbols/ES.yaml
symbol: ES
exchange: CME
asset_class: equity_index
tick_size: 0.25
tick_value: 12.50          # verified, exchange-fixed
point_value: 50.00          # tick_value / tick_size
contract_months: [H, M, U, Z]   # Mar, Jun, Sep, Dec (quarterly)
rollover_rule: highest_volume    # NOT a simple pairwise chronological crossover —
                                   # must compare against ALL listed months to avoid
                                   # hopping through a thin/dead month that happens to
                                   # sit next in calendar order (found via Phase 1
                                   # plotting on MGC/SIL: chronological-adjacent
                                   # comparison rolled onto contracts holding <1% of
                                   # actual volume before reaching the real liquid month)
calendar_backstop_days: 3         # tuned empirically against real data — 8 was too
                                   # loose and preempted the volume crossover entirely
                                   # (confirmed on MES: forced a roll onto a contract
                                   # holding only 1.3% of daily volume). 3 lets the
                                   # volume-based rule do its job; re-validate this
                                   # number if a different instrument set behaves
                                   # differently in Phase 4.
session:
  timezone: America/Chicago
  globex_open: "17:00"           # prior day, nearly 23h/day
  globex_close: "16:00"
  rth_open: "08:30"              # equity-index RTH aligned to cash market
  rth_close: "15:15"
  use_eth_range: false           # whether §2 "prior day H/L" includes overnight range
liquidity_filter:
  min_avg_volume: <set from backtest — don't guess a number here>
margin_source: ibkr_api           # never hardcode — fetch live
```

Repeat per instrument. **Verify session times before going live** — RTH
conventions for non-equity-index futures (energies, metals) are less
standardized post-pit-closure than for ES/NQ/YM, and different platforms define
them differently.

**Update, verified against IB's `liquidHours` field**: IB reports the same
generic `08:30–16:00 CT` window for every one of these instruments —
including MET, which has no RTH concept at all by definition. That's a strong
signal `liquidHours` is a broad platform default rather than instrument-
specific truth, which is exactly the "different platforms define them
differently" problem this spec anticipated — it just turned out IB's own API
is one more inconsistent source, not a reliable tiebreaker. **Don't take IB's
liquidHours as settled fact for any of these six.** Use the empirical
highest-volume-window approach (originally scoped just for GC/SI) uniformly
across all six non-MET instruments instead — measure it directly from
volume distribution in the historical data rather than trusting any single
platform's reported session.

| Symbol | Exchange | Tick size | Tick value | Point value | Contract months | Notes |
|---|---|---|---|---|---|---|
| ES | CME | 0.25 | $12.50 | $50 | H,M,U,Z (quarterly) | Equity index, RTH 08:30–15:15 CT is well-standardized |
| NQ | CME | 0.25 | $5.00 | $20 | H,M,U,Z (quarterly) | Same session convention as ES |
| YM | CBOT | 1.00 | $5.00 | $5 | H,M,U,Z (quarterly) | Same session convention as ES |
| CL | NYMEX | 0.01 | $10.00 | $1,000 | Monthly | Legacy pit RTH ~08:00–13:30 CT commonly used but confirm — rolls monthly, 3–4 days before expiry, more frequent rollover logic needed than the quarterly index contracts |
| GC | COMEX | 0.10 | $10.00 | $100 | G,J,M,Q,V,Z (Feb/Apr/Jun/Aug/Oct/Dec) | RTH convention weakest of this group post-floor-closure — treat as needing empirical confirmation (e.g., highest-volume window) rather than assumed pit hours |
| SI (micro, traded as SIL) | COMEX | 0.005 | $5.00 | $1,000/pt | F,H,K,N,U,V,X,Z | **IB lookup trap — see note below** |
| MET | CME | 0.50 (index pts) | $0.05/contract | $0.10/index pt | Monthly (6 near months) + quarterlies further out | **Different asset class — see note below** |

**Verified against IB's `reqContractDetails` — two corrections from the original table above:**
- **IB lists Micro Silver under the symbol `SI`, not `SIL`.** The `SIL` Globex
  code only shows up in `localSymbol`, not the lookup symbol itself. Worse,
  querying `SI` on COMEX through IB returns **both the full-size (5,000oz,
  $25/tick) and micro (1,000oz, $5/tick) contracts mixed together** — roughly
  40 contracts came back in one query. **`ib_multiplier: 1000` must be set as
  an explicit disambiguator in the SIL config**, or the wrong contract (5x the
  intended size, 5x the intended dollar risk per tick) can silently get
  selected instead. This is exactly the kind of silent wrong-instrument risk
  that matters with real capital on the line — treat this disambiguator as
  mandatory, not optional, and add a startup assertion that checks the
  resolved contract's multiplier matches 1000 before the bot ever trades it.
- **Contract months were wrong.** Corrected list per IB: `F,H,K,N,U,V,X,Z`
  (not the `H,K,N,U,Z` originally guessed).
- The original `SI` row (COMEX full-size, 5,000oz, $25/tick, H,K,N,U,Z) was
  never wrong on its own terms — it's a real, valid contract — it just isn't
  the one this bot is actually meant to trade at this account size. Keep both
  entries distinct in config if you ever want the full-size contract
  available later; don't let the two get merged under one symbol key.

**MET (Micro Ether) is a genuine outlier in this table, not just another micro.**
A few things that don't carry over from the other six instruments:
- **No RTH concept at all.** Crypto trades essentially continuously (Globex
  Sun 5pm CT – Fri 4pm CT with only the standard daily maintenance halt) —
  there's no cash-market session to anchor a "day session" to. Recommend
  treating the full Globex session as the only session for MET (`use_eth_range:
  true` effectively becomes the default, not an override) rather than forcing
  an artificial RTH window onto it.
- **"Prior day high/low" (§2) needs a redefined day boundary** — pick a fixed
  UTC or CT rollover time (e.g., the daily maintenance halt) as the day cutoff,
  since there's no natural close.
- **Financially settled, cash notional is small**: at ETH ≈ $4,000, one MET
  contract ≈ $400 notional — meaningfully smaller dollar exposure per contract
  than any of the other five even at 1 lot, which changes your position-sizing
  math (you may find `contracts` computed from the sizing formula in the
  exit/risk spec comes out larger than 1–2 for MET at the same dollar risk).
- **Dynamic price limits** (CME applies a 10% dynamic variant on crypto
  futures) — worth a specific liquidity/volatility sanity check in the Risk
  Engine before treating MET identically to the traditional futures in the
  no-trade-on-illiquid-symbols rule.
- Contract months are **monthly**, not quarterly like the index futures —
  rollover logic needs the same "monthly, more frequent" treatment already
  noted for CL, not the ES/NQ/YM quarterly pattern.

**Micro contract note**: MES, MNQ, MYM, MCL, MGC, SIL, and MET exist as
smaller-size versions (1/10, 1/5 for SI→SIL, or the only size at all for MET,
which has no full-size "ETH" retail equivalent in this bot's scope) of their
respective instruments. Worth including as alternate configs from day one —
during paper trading / early live testing, trading micros lets you validate
the full pipeline with real order flow and real slippage at a fraction of the
dollar risk, before sizing up to full contracts.

## Part 2 — Repo Structure

```
trading-bot/
├── config/
│   ├── symbols/
│   │   ├── ES.yaml, NQ.yaml, YM.yaml, CL.yaml, GC.yaml, SI.yaml
│   │   ├── MET.yaml           # crypto — different session/rollover handling, see spec note
│   │   └── micros/  (MES.yaml, MNQ.yaml, ...)
│   ├── params.yaml            # all tunables from the level-detection spec
│   ├── scoring_weights.yaml   # confidence-scoring weights (§Stage 2)
│   └── risk.yaml              # daily loss limit, max positions, cooldown rules
│
├── data/
│   ├── sources/
│   │   ├── base.py            # shared OHLCV+metadata schema both sources normalize to
│   │   ├── databento_client.py
│   │   └── ibkr_client.py
│   └── continuous_contract.py # roll logic per rollover_rule in symbol spec
│
├── features/
│   ├── schema.py               # shared feature output dataclass
│   ├── structure.py            # swings (§1), channels (parallel channel logic)
│   ├── levels.py                # prior day/week H-L (§2), gaps (§3), ranges (§5)
│   ├── triggers.py              # rejection (§7), engulfing, breakout/retest (§9),
│   │                             # failed breakout (§8), three-tail (§19), momentum (§11)
│   ├── confirmation.py          # volume expansion (§12), CLV (§13)
│   ├── confirmation_signal.py   # Soloway two-stage close confirmation (§17)
│   ├── time_count.py            # exhaustion counter (§18)
│   └── risk_state.py            # ATR, volatility regime (§15)
│
├── signal_engine/
│   ├── gates.py                 # Stage 1 hard gates
│   ├── scoring.py                # Stage 2 weighted confidence score
│   └── engine.py                  # orchestrates gates → scoring → ranked output
│
├── risk_engine/
│   ├── sizing.py                  # position sizing off symbol spec + ATR stop distance
│   └── controls.py                 # daily loss, max positions, cooldown, liquidity/news veto
│
├── execution/
│   ├── ibkr_execution.py            # live/paper order placement via Interactive Brokers
│   └── simulated_execution.py       # backtest fill simulation (slippage model)
│
├── backtest/
│   ├── engine.py                     # replays Data→Feature→Signal→Risk identically to live
│   └── report.py                      # performance attribution, walk-forward validation
│
├── monitoring/
│   ├── logger.py                       # structured logs: every signal + all sub-scores
│   └── comparison.py                    # diffs backtest vs. paper behavior on same dates
│
├── tests/
│   └── (unit tests per feature function — these are the most important tests in
│         the repo, since a wrong swing-detection or CLV formula silently corrupts
│         everything downstream)
│
├── run_backtest.py
├── run_paper.py
└── run_live.py   # do not build/enable until backtest + paper both validate cleanly
```

## Part 3 — Build Order

Build in this sequence — each phase should be independently testable before
the next begins:

1. **Data Layer + continuous contracts.** Get clean, correctly-rolled OHLCV for
   all seven instruments (including MET's monthly rollover and continuous
   overnight session) from Databento. Validate visually — plot each continuous
   series and confirm no artificial gaps/spikes at roll dates.
2. **Feature Layer.** Implement structure/levels/triggers/confirmation/risk-state
   against historical data. Validate by plotting detected swings, levels, and
   trigger flags over real charts and sanity-checking against what you'd mark
   by eye — this is the step most worth spending real time on, since every
   downstream layer inherits its errors silently.
3. **Signal Engine** (gates + scoring). Run against Phase 2 output in
   log-only mode — no execution yet. Review the log of flagged setups by hand
   against charts before trusting the scoring.
4. **Backtest Engine.** Replay signals through simulated execution with a
   realistic slippage/fill model and the full Risk Engine. This produces your
   first real performance numbers — expect several iterations of parameter
   tuning (the tables in both prior specs) here.
5. **Live data + paper execution via Interactive Brokers.** Only after backtest results
   are stable. Run paper trading in parallel with backtest on overlapping
   dates specifically to confirm they produce the *same* signals — any
   divergence here means the "shared feature/signal logic" principle got
   violated somewhere and needs fixing before going further.
6. **Monitoring/dashboard refinement**, then a deliberate, gradual step from
   paper to small live size (micros first, per the note above) — not a jump
   straight to full-size contracts on day one of live trading.
