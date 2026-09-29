"""Read-only analyses of backtest results (Phase 4.4 on).

Nothing here is an input to the setup study, which is why this is its own
package: scripts/run_setup_study.py hashes the engine's packages into its
result stamps, and an analysis module must not mark finished results stale.
"""
