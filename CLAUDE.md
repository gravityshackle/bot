# Project conventions

## File writes: always `encoding="utf-8"`

Every write must pass it explicitly:

```python
path.write_text(s, encoding="utf-8")
open(path, "w", encoding="utf-8")
```

Windows defaults to cp1252, which silently mangles `§`, `—`, `×`, `→` — all of
which appear throughout the spec references in this repo's configs and docs. It
already corrupted `docs/open_questions.md` once. The failure is silent at write
time and only shows up later as replacement characters, so the habit has to be
unconditional rather than applied when non-ASCII is expected.

Set `PYTHONIOENCODING=utf-8` when a script prints non-ASCII to the console.

## No hardcoded thresholds

Every threshold, multiplier, window and weight lives in `config/` — `params.yaml`
for feature-layer tunables, `risk.yaml` for risk controls, `scoring_weights.yaml`
for confidence weights, `config/symbols/*.yaml` for per-instrument values. Feature
and signal logic reads them through `features.schema.load_params()`. If a number
appears in a `.py` file, it is a bug.

Parameters the specs do not actually define are parked on `UNRESOLVED` /
`UNDEFINED_IN_SPEC` sentinels; `Params.get()` raises rather than substituting a
guess. See `docs/open_questions.md`.

## Causality — no lookahead, ever

- A feature at bar `i` may use only bars `<= i`, and bar `i` only once closed.
- Rolling baselines exclude the current bar. A bar must never contribute to the
  threshold it is tested against.
- Swing pivots are invisible until `confirmed_idx = idx + N`. Use
  `structure.last_confirmed_swings()`, never raw pivot indices.
- Higher-timeframe values align on the HTF bar's **close**, not its open.
- Rolls are decided from completed sessions and take effect the next session.

## Unknown is a state, not a default

Never collapse "not yet knowable" into `False` or `0`. Use nullable dtypes
(`pd.NA`) or explicit `unknown` labels. Silent demotion of unknown state caused
real bugs here already.

## Money and orders

- Any Databento fetch that would bill requires `confirm=True` on explicit
  operator approval. Cache reads are free and ungated.
- IB connections are paper only (port 4002, `DU*` accounts). `assert_paper_account()`
  refuses anything else. `run_live.py` is deliberately not built.
- Entries are limit orders only. The sole market order in the system is the
  daily circuit-breaker flatten, which is an exit.
- The daily risk state machine is a hard limit with no override path.

## Testing

- The suite must run offline and spend nothing. Stub at the client boundary.
- Unit tests on feature functions are the most important tests in the repo — a
  wrong swing or CLV formula corrupts everything downstream silently.
- When a bug is found on real data, add a regression test with the exact fixture
  that reproduced it.

## Environment

Python 3.12 at `../venv` (outside the repo). Run as
`../venv/Scripts/python.exe -m pytest`.
