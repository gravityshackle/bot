"""The full-window setup study (Phase 4.3), resumable, one instrument at a time.

Usage (from the repo root):
    python scripts/run_setup_study.py [--symbols MES MNQ ...] [--days N] [--out DIR]

With no --symbols it runs every instrument in config/symbols/. With no --days it
uses each instrument's whole cached series. Results land in --out (default
cache/setup_study/), one pair of files per instrument:
    SYM.pkl   the study() DataFrame
    SYM.json  the manifest: stamp, the data file's sha256, rows, seconds

RESUMABLE. An instrument counts as done only when its manifest's stamp equals
the stamp this run would write (a hash of every source file in the engine's
packages, every config file, the symbol and the window) AND the data file
matches the sha256 the manifest records. Anything else is missing, stale or
corrupt, and is re-run. Files are written under a temporary name and then
renamed, and the manifest goes last, so a run killed at any point leaves either
a complete result or none. Re-running the same command after an interruption
redoes only what is not done. A worker that fails stops the job (it is not
retried on its own), so re-running is always the user's call.

ONE INSTRUMENT AT A TIME, BY CONSTRUCTION. This machine has 3.9 GB of RAM, and
a single 5-year worker peaks near 1 GB.
  - The orchestrator never imports the engine. It starts one worker process per
    instrument and waits for it (subprocess.run). There is no pool, and the
    worker's command line takes exactly one symbol.
  - Two OS locks, at a fixed path whatever --out is: a job lock the
    orchestrator holds for the whole run, and a worker lock each worker holds
    while it computes. A second orchestrator, or a worker started by hand,
    exits with LOCKED and computes nothing. The locks sit on open file
    handles, so the OS releases them when a process dies. A kill never leaves
    a stale lock behind.
  - Each worker is a fresh process, so its memory goes back to the OS before
    the next instrument starts.

Test seam: SETUP_STUDY_COMPUTE="module:function" replaces the compute step
(tests/fake_study_compute.py). Everything else, including the locks, the
stamps, the writes and the skip logic, is the real code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CODE_DIRS = ("backtest", "data", "execution", "features", "risk_engine", "signal_engine")
CONFIG_DIR = "config"
DEFAULT_OUT = Path("cache") / "setup_study"
LOCK_DIR = Path("cache")
JOB_LOCK = "setup_study.job.lock"
WORKER_LOCK = "setup_study.worker.lock"

OK, USAGE, LOCKED, WORKER_FAILED, CODE_CHANGED = 0, 2, 3, 4, 5


# --- stamp and status -------------------------------------------------------

def _tree_sha(root: Path, rel_dirs, pattern: str) -> str:
    h = hashlib.sha256()
    for d in rel_dirs:
        for p in sorted((root / d).rglob(pattern)):
            if "__pycache__" in p.parts or not p.is_file():
                continue
            h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
            h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def stamp(symbol: str, days: int | None, root: Path | None = None) -> dict:
    """What a result depends on. A result is reusable only if this is equal."""
    root = root or ROOT
    return {"symbol": symbol, "days": days,
            "code_sha": _tree_sha(root, CODE_DIRS, "*.py"),
            "config_sha": _tree_sha(root, (CONFIG_DIR,), "*")}


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _paths(out: Path, symbol: str) -> tuple[Path, Path]:
    return out / f"{symbol}.pkl", out / f"{symbol}.json"


def status(out: Path, symbol: str, st: dict) -> str:
    """'done', or why not: 'missing', 'stale' or 'corrupt'."""
    data, man = _paths(out, symbol)
    if not man.exists():
        return "missing"
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "corrupt"
    if m.get("stamp") != st:
        return "stale"
    if not data.exists() or _file_sha(data) != m.get("data_sha256"):
        return "corrupt"
    return "done"


def _replace_atomically(target: Path, write) -> None:
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "wb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def write_result(out: Path, symbol: str, df, st: dict, seconds: float) -> None:
    """Data first, manifest last. Until the manifest lands, the result is not done."""
    import pickle

    out.mkdir(parents=True, exist_ok=True)
    data, man = _paths(out, symbol)
    man.unlink(missing_ok=True)
    _replace_atomically(data, lambda fh: pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL))
    body = {"stamp": st, "data_sha256": _file_sha(data), "rows": int(len(df)),
            "seconds": round(seconds, 1), "commit": _git_commit(), "written": time.time()}
    _replace_atomically(man, lambda fh: fh.write(json.dumps(body, indent=2).encode("utf-8")))


def _git_commit() -> str | None:
    """For the record only; never compared (a docs-only commit changes nothing)."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --- locks --------------------------------------------------------------------

def _try_lock(name: str):
    """An exclusive lock held by this process until it exits, or None if taken."""
    d = ROOT / LOCK_DIR
    d.mkdir(parents=True, exist_ok=True)
    fh = open(d / name, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


# --- the study itself -----------------------------------------------------------

def compute(symbol: str, days: int | None):
    """study() over the instrument's cached series (cache only; run from the repo root)."""
    import pandas as pd
    import yaml

    from backtest.setup_study import StudyConfig, study
    from data.pipeline import build_continuous, load_data_config
    from features.schema import load_params
    from signal_engine import gates, timeframes

    series, scfg, _ = build_continuous(symbol, load_data_config())
    bars = series.bars
    if days:
        bars = bars[bars["ts"] >= bars["ts"].max() - pd.Timedelta(days=days)].reset_index(drop=True)
    del series, _
    params = load_params(symbol)
    ctx = gates.GateContext.build(timeframes.build(symbol, bars, scfg, params), scfg)

    def y(p):
        with open(p, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    sc = StudyConfig.from_configs(costs_cfg=y("config/costs.yaml"), exec_cfg=y("config/execution.yaml"),
                                  risk_cfg=y("config/risk.yaml"), symbol_cfg=scfg,
                                  timeframe=str(params.get("timeframes.entry")))
    return study(ctx, bars, sc)


def _compute_fn():
    spec = os.environ.get("SETUP_STUDY_COMPUTE")
    if not spec:
        return compute
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def run_worker(symbol: str, out: Path, days: int | None) -> int:
    lock = _try_lock(WORKER_LOCK)
    if lock is None:
        print(f"{symbol}: LOCKED -- another study worker is running; nothing computed", flush=True)
        return LOCKED
    st = stamp(symbol, days)
    t0 = time.time()
    df = _compute_fn()(symbol, days)
    if stamp(symbol, days) != st:
        print(f"{symbol}: code or config changed while computing; result discarded", flush=True)
        return CODE_CHANGED
    write_result(out, symbol, df, st, time.time() - t0)
    print(f"{symbol}: wrote {len(df):,} rows in {time.time() - t0:.0f}s", flush=True)
    return OK


# --- orchestrator ---------------------------------------------------------------

def default_symbols() -> list[str]:
    return sorted(p.stem for p in (ROOT / CONFIG_DIR / "symbols").glob("*.yaml"))


def _worker_cmd(symbol: str, out: Path, days: int | None) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", symbol, "--out", str(out)]
    return cmd + (["--days", str(days)] if days else [])


def run_job(symbols: list[str], out: Path, days: int | None) -> int:
    lock = _try_lock(JOB_LOCK)
    if lock is None:
        print("LOCKED -- a setup-study job is already running; nothing started", flush=True)
        return LOCKED
    t_job = time.time()
    for sym in symbols:
        st = stamp(sym, days)
        why = status(out, sym, st)
        if why == "done":
            print(f"{sym}: done already, skipped", flush=True)
            continue
        print(f"{sym}: {why}, running ...", flush=True)
        t0 = time.time()
        rc = subprocess.run(_worker_cmd(sym, out, days), cwd=ROOT).returncode
        after = status(out, sym, st)
        if rc != OK or after != "done":
            print(f"{sym}: FAILED (exit {rc}, result {after}) after {time.time() - t0:.0f}s. "
                  f"Job stopped; re-run the same command to resume from {sym}.", flush=True)
            return LOCKED if rc == LOCKED else WORKER_FAILED
    print(f"all {len(symbols)} done in {time.time() - t_job:.0f}s -> {out}", flush=True)
    return OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", metavar="SYM", help="internal: compute exactly one instrument")
    ap.add_argument("--symbols", nargs="+", metavar="SYM")
    ap.add_argument("--days", type=int, default=None, help="last N days only (default: whole series)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = (a.out or ROOT / DEFAULT_OUT).resolve()
    if a.days is not None and a.days <= 0:
        ap.error("--days must be positive")
    if a.worker:
        if a.symbols:
            ap.error("--worker takes exactly one symbol and no --symbols")
        return run_worker(a.worker, out, a.days)
    symbols = a.symbols or default_symbols()
    known = set(default_symbols())
    bad = [s for s in symbols if s not in known or not re.fullmatch(r"[A-Z0-9]+", s)]
    if bad or len(set(symbols)) != len(symbols):
        ap.error(f"unknown or repeated symbols: {bad or symbols}")
    return run_job(symbols, out, a.days)


if __name__ == "__main__":
    sys.exit(main())
