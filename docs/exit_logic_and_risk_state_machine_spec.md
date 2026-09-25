# Exit Logic, Order Mechanics & Daily Risk State Machine — v1

Account context this spec is built around: **$5,000 account, micro contracts,
1–2 trades/day, $50 risk/trade, $100/day stop-for-day cooldown after 2 losses,
$300/day hard circuit breaker, single contract (no partials yet).**
All dollar figures below are config values, not hardcoded constants — but
they're written as real numbers here rather than placeholders, since you now
have an actual account to build against.

---

## Part 1 — Position Sizing

```
risk_dollars = min(account_equity × risk_pct, max_risk_per_trade)
             = min(equity × 0.01, $50)          # default config

stop_distance_ticks = abs(entry_price - stop_price) / tick_size
contracts = floor(risk_dollars / (stop_distance_ticks × tick_value))
contracts = min(contracts, max_contracts_config)   # default: 1

if contracts == 0:
    discard the setup — do not round up to 1 contract just to take the trade.
    A stop distance too wide for your risk budget at your current size is
    itself useful information (skip it, don't force it).
```

**Design note on scaling**: `risk_pct = 1.0%` is the config value; the dollar
amount recalculates naturally as `account_equity` grows, so you don't need to
manually bump $50 → $75 as the account compounds. Your **$100 daily target and
$300/$100 daily-loss numbers are currently fixed dollar amounts, not
percentages** — I'd deliberately leave them fixed rather than auto-scaling,
and revisit them yourself periodically (e.g., quarterly, or after a clear
account-size milestone) rather than letting the system silently increase your
daily risk tolerance as balance grows. Auto-scaling everything at once is how
overconfidence creeps back into a rules-based system without a real decision
ever being made about it.

---

## Part 2 — Order Mechanics (Entry)

**All entries are limit orders in v1. No market-order entry path exists in the
bot at all** — this is deliberate, not a placeholder to fill in later. A
discretionary "sudden opportunity" market order is exactly the kind of
impulsive deviation the system is meant to protect you from; if you want to
take one, do it manually, outside the bot, so it's never confused with "the
system said so."

**Entry price by trigger type** (confirmation must have already occurred on a
closed bar before any order is placed — never place an entry order intrabar
before the confirmation candle closes):

| Trigger type | Entry limit price |
|---|---|
| Breakout/retest (§9), Confirmation Signal (§17) | At the level being retested, ± 1 tick in your favor |
| Rejection candle (§7), three-tail (§19), engulfing | At or near the confirmation candle's close (price is already at the level for these) |
| Momentum continuation (§11) | At the minor level referenced in the trigger |

**Time-in-force**: entry limit order stays live for `K_entry_expire` bars
(default 3–5, config) after being placed. If unfilled, cancel it — per your
own instinct: if it doesn't hit the level, you don't take the trade, and you
wait for the next qualifying setup instead of chasing.

**Invalidation-before-fill**: if price closes beyond the *stop/invalidation*
level before the entry limit order fills, cancel the pending order
immediately — the setup's thesis is already dead even though you were never
actually in the trade.

---

## Part 3 — Exit State Machine

Single-contract path (partials deliberately excluded for now — add back the
1R-partial branch later once you're consistently trading 2+ contracts; the
hook for it is noted at the bottom).

```
State: ENTRY_PENDING
  → limit order live, TIF = K_entry_expire bars
  → on fill: transition to IN_POSITION
  → on expiry or invalidation-before-fill: transition to CANCELLED (no trade)

State: IN_POSITION
  → immediately place STOP-MARKET order at invalidation level ± ATR buffer (§15)
    (stop-market, not stop-limit — a guaranteed exit matters more here than a
    few ticks of slippage precision)
  → immediately place resting LIMIT order at primary target:
      target_price = nearer of (next major level in trade direction, 2R price)
  → monitor for 1R reached:
      if price reaches 1R in favor → move stop to breakeven (entry price),
      cancel/replace the stop-market order → transition to BREAKEVEN

State: BREAKEVEN
  → stop now at breakeven, target limit order still resting
  → on EVERY new confirmed swing point (§1) that forms in the trade's favor
    since entry (not just at target-touch — check this continuously, since a
    swing only counts as confirmed once N bars have closed after the pivot,
    so this can never be evaluated at the instant of a fill):
      cancel the resting target limit order immediately, transition to
      TRAILING (structure has confirmed continuation — let it run per your
      original "optional trailing stop only after structure confirms
      continuation" rule, now made concrete, and now decided *before* target
      is ever reached rather than at the same instant a resting order would
      fill)
  → if NO qualifying swing has confirmed and price reaches the target price:
      the resting limit simply fills at the exchange → transition to CLOSED
      (full planned exit) — this path never requires an inspect-then-decide
      step, since by the time we'd be inspecting, the order has already filled
  → if stop (breakeven) is hit first, with no swing having confirmed and no
    target fill: transition to CLOSED (scratch, ~$0)

State: TRAILING
  → stop trails behind each newly confirmed swing point (§1) in trade
    direction, offset by the same ATR buffer used for the initial stop
  → no re-entry of a fixed target — this state only exits via the trailing
    stop being hit
  → transition to CLOSED when trailing stop is hit

State: CLOSED
  → log full trade record: entry, exit, R multiple, which state path was
    taken (planned exit vs. trailing vs. scratch vs. stopped out), and every
    confidence sub-score from signal time — this is the dataset for later
    parameter tuning and eventual ML scoring.
```

**Partial-profit hook (for later, once trading 2+ contracts)**: insert a
branch in `IN_POSITION` at the 1R checkpoint — close 1 contract at 1R via
limit, move stop to breakeven on the remainder, then continue exactly as
above for the remaining contract(s). Nothing else in this state machine needs
to change; it's an additive branch, not a redesign.

---

## Part 4 — Daily Risk State Machine

```
Daily state, reset at session start (17:00 CT, account-wide, across all
instruments/exchanges including continuous MET — a single shared daily clock,
not per-instrument):
  consecutive_losses = 0
  realized_pnl_today = 0
  trades_today = 0

On each trade CLOSED:
  realized_pnl_today += trade_pnl
  trades_today += 1
  if trade_pnl < 0: consecutive_losses += 1
  else: consecutive_losses = 0

On EVERY bar/price update while a position is open (not just on trade close):
  daily_pnl = realized_pnl_today + unrealized_pnl_of_open_position
  if daily_pnl <= -$300: hard circuit breaker fires immediately (see #3 below)

  This check must run continuously against realized + unrealized P&L. The
  original version of this spec only evaluated daily_pnl on trade CLOSED
  events, which meant a single open position drawing down past -$300 while
  still open would never trip the breaker — nothing had "closed" yet, so the
  check was never reached. That defeats the entire purpose of a circuit
  breaker, which exists specifically to cut off an open loss, not just to
  audit closed ones. Continuous evaluation while in a position is mandatory,
  not an optimization.

Checks BEFORE allowing any new entry:
  1. if trades_today >= max_trades_per_day (default 2): no new entries today
  2. if consecutive_losses >= 2: no new entries today (stop-for-day cooldown,
     typical realized loss ≈ $100 at current sizing)
  3. if daily_pnl <= -$300 (per the continuous check above): hard circuit
     breaker —
       - cancel all pending entry orders immediately
       - flatten any open position at MARKET (this is the one place a market
         order is correct — exiting for capital preservation is a different
         action than opportunistic entry, and speed matters more than price
         here)
       - no new entries for the remainder of the day, no exceptions
  4. if realized_pnl_today >= $100 (daily target reached, using realized P&L
     specifically — don't stop new entries just because an open position is
     currently showing an unrealized $100+ gain that could still give back):
     no new entries for the rest of the day by default (`stop_after_daily_
     target: true`, config flag if you later want to keep taking A+ setups
     after hitting target — start with it OFF, i.e., stop once you've hit it,
     since locking in the win protects against exactly the give-back pattern
     that's easy to fall into on a small account)
  5. max_open_positions = 1 for now. The ES/NQ/YM correlated-group rule below
     is currently unreachable as a result (you can't hold two positions of
     any kind while capped at one) — leave the rule in the codebase anyway
     rather than deleting it, since it becomes load-bearing the moment
     max_open_positions is raised above 1 later, and you don't want to have
     to remember to re-add it then. Additionally, treat ES/NQ/YM as a
     correlated group — never hold concurrent positions across more than one
     instrument in that group at a time, even if each individually sizes
     fine, since at $5,000 an index-wide move against a "diversified" pair of
     correlated positions is really just one concentrated bet with extra
     steps.
```

**Sequencing note**: checks run in the order listed — the hard circuit
breaker (#3) always wins if multiple conditions are true simultaneously
(e.g., don't let a same-day "well I already hit my target" logic suppress the
emergency flatten).

---

## What's intentionally NOT built into v1

- No partial profit-taking (hook left in place for when you're ready).
- No market-order entry path (only the emergency daily-flatten uses a market
  order, and that's an exit, not an entry).
- No auto-scaling of the $100/$300 dollar figures as equity grows — revisit
  manually.
- No overriding the daily target/loss/cooldown limits from within the bot
  itself under any condition, including a high confidence score. If you ever
  want to override one of these limits, that has to be a manual, outside-the-
  bot decision on your part — never a code path the system can trigger on its
  own reasoning.
