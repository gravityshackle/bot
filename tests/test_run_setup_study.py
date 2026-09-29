"""The full-window setup-study runner: resume after a kill, one instrument at a time.

These run the REAL runner, with its orchestrator, worker processes, OS locks,
stamps, atomic writes and skip logic. Only the compute step is swapped, for
tests/fake_study_compute.py (via SETUP_STUDY_COMPUTE), which logs start and
end events and can hang on demand.
  - A worker is really killed partway through an instrument. The job stops
    there, the killed instrument leaves no result, and the one after it never
    starts. The same command then re-runs only the killed and unstarted
    instruments, in order, and leaves the finished one byte for byte alone.
  - Each way a result can be not-done (stale code, stale config, a different
    window, a missing manifest, damaged data, a leftover temp file) makes
    exactly that instrument re-run, and nothing else.
  - A second job, or a worker started by hand, is refused while a worker runs.
    It computes nothing, and the worker command takes exactly one symbol.
    Across every run in this file, no two compute intervals overlap.

The locks are machine-wide (under cache/), so these tests fail with LOCKED if
a real study job is running. That is the intended behaviour.

conftest's network guard does not reach subprocesses. The workers get no API
key, and the fake compute never loads data.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_setup_study.py"
SYMS = ["MES", "MNQ", "MYM"]

_spec = importlib.util.spec_from_file_location("run_setup_study", RUNNER)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

ALL_EVENTS: list[dict] = []     # every compute event in this file, for the overlap check


def _env(log: Path, **extra) -> dict:
    env = os.environ.copy()
    env.pop("DATABENTO_API_KEY", None)
    env.pop("FAKE_STUDY_HANG", None)
    env.update(SETUP_STUDY_COMPUTE="tests.fake_study_compute:compute", FAKE_STUDY_LOG=str(log),
               PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def _cmd(*args) -> list[str]:
    return [sys.executable, str(RUNNER), *map(str, args)]


def _run(env, *args, timeout=120) -> subprocess.CompletedProcess:
    return subprocess.run(_cmd(*args), cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout)


def _events(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def _wait_for(log: Path, ev: str, sym: str, timeout=60) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        hits = [e for e in _events(log) if e["ev"] == ev and e["sym"] == sym]
        if hits:
            return hits[-1]
        time.sleep(0.1)
    raise AssertionError(f"no {ev} event for {sym} within {timeout}s")


def _started(log: Path) -> list[str]:
    return [e["sym"] for e in _events(log) if e["ev"] == "start"]


def _kill(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)    # TerminateProcess on Windows: no cleanup runs
    ALL_EVENTS.append({"ev": "killed", "pid": pid, "t": time.time()})   # ends its interval


def _record(log: Path) -> None:
    ALL_EVENTS.extend(_events(log))


def _sha(p: Path) -> str:
    return rs._file_sha(p)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """One complete, untouched run of all three instruments."""
    d = tmp_path_factory.mktemp("baseline")
    log = d / "events.jsonl"
    r = _run(_env(log), "--symbols", *SYMS, "--out", d / "out")
    assert r.returncode == rs.OK, r.stdout + r.stderr
    assert _started(log) == SYMS
    _record(log)
    return d / "out"


# --- the kill-and-resume test -----------------------------------------------------

def test_worker_killed_mid_instrument_then_resume_redoes_only_the_unfinished(tmp_path):
    out, log = tmp_path / "out", tmp_path / "events.jsonl"
    job = subprocess.Popen(_cmd("--symbols", *SYMS, "--out", out), cwd=ROOT,
                           env=_env(log, FAKE_STUDY_HANG="MNQ"),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        hung = _wait_for(log, "start", "MNQ")
        assert rs.status(out, "MES", rs.stamp("MES", None)) == "done"    # finished before the kill
        _kill(hung["pid"])
        stdout, _ = job.communicate(timeout=60)
    finally:
        if job.poll() is None:
            job.kill()
    _record(log)

    assert job.returncode == rs.WORKER_FAILED, stdout
    assert "MNQ: FAILED" in stdout
    assert _started(log) == ["MES", "MNQ"]                        # MYM never started
    assert rs.status(out, "MNQ", rs.stamp("MNQ", None)) == "missing"
    assert not (out / "MNQ.pkl").exists() and not (out / "MNQ.json").exists()
    assert rs.status(out, "MYM", rs.stamp("MYM", None)) == "missing"
    mes = {p: (_sha(out / p), (out / p).stat().st_mtime_ns) for p in ("MES.pkl", "MES.json")}

    log.unlink()
    r = _run(_env(log), "--symbols", *SYMS, "--out", out)          # the same command, again
    _record(log)
    assert r.returncode == rs.OK, r.stdout + r.stderr
    assert "MES: done already, skipped" in r.stdout
    assert _started(log) == ["MNQ", "MYM"]                         # only the unfinished, in order
    assert {p: (_sha(out / p), (out / p).stat().st_mtime_ns) for p in mes} == mes
    for s in SYMS:
        assert rs.status(out, s, rs.stamp(s, None)) == "done"
        df = pd.read_pickle(out / f"{s}.pkl")
        assert list(df["symbol"].unique()) == [s]


# --- each not-done state re-runs exactly that instrument ------------------------------

def _stale_code(out):
    m = json.loads((out / "MNQ.json").read_text(encoding="utf-8"))
    m["stamp"]["code_sha"] = "0" * 64                  # written by other code
    (out / "MNQ.json").write_text(json.dumps(m), encoding="utf-8")


def _stale_config(out):
    m = json.loads((out / "MNQ.json").read_text(encoding="utf-8"))
    m["stamp"]["config_sha"] = "0" * 64                # written under other configs
    (out / "MNQ.json").write_text(json.dumps(m), encoding="utf-8")


def _data_without_manifest(out):                       # killed between the two renames
    (out / "MNQ.json").unlink()


def _data_damaged(out):
    p = out / "MNQ.pkl"
    p.write_bytes(p.read_bytes()[:-10])


def _manifest_damaged(out):
    (out / "MNQ.json").write_text("{not json", encoding="utf-8")


def _data_missing(out):
    (out / "MNQ.pkl").unlink()


@pytest.mark.parametrize("damage", [_stale_code, _stale_config, _data_without_manifest,
                                    _data_damaged, _manifest_damaged, _data_missing],
                         ids=lambda f: f.__name__.lstrip("_"))
def test_each_not_done_state_reruns_only_that_instrument(baseline, tmp_path, damage):
    out, log = tmp_path / "out", tmp_path / "events.jsonl"
    shutil.copytree(baseline, out)
    (out / ".MYM.pkl.tmp").write_bytes(b"half a file")    # a leftover temp never counts
    untouched = {s: _sha(out / f"{s}.pkl") for s in ("MES", "MYM")}
    damage(out)
    assert rs.status(out, "MNQ", rs.stamp("MNQ", None)) != "done"

    r = _run(_env(log), "--symbols", *SYMS, "--out", out)
    _record(log)
    assert r.returncode == rs.OK, r.stdout + r.stderr
    assert _started(log) == ["MNQ"]
    assert {s: _sha(out / f"{s}.pkl") for s in untouched} == untouched
    assert rs.status(out, "MNQ", rs.stamp("MNQ", None)) == "done"


def test_a_different_window_reruns_everything(baseline, tmp_path):
    out, log = tmp_path / "out", tmp_path / "events.jsonl"
    shutil.copytree(baseline, out)
    r = _run(_env(log), "--symbols", *SYMS, "--days", 30, "--out", out)
    _record(log)
    assert r.returncode == rs.OK, r.stdout + r.stderr
    assert _started(log) == SYMS
    assert all(pd.read_pickle(out / f"{s}.pkl")["days"].eq(30).all() for s in SYMS)


def test_the_stamp_follows_real_code_and_config_bytes(tmp_path):
    """The re-run tests above edit a manifest's hash. This test shows a real
    byte change in a code or config file changes the hash just the same."""
    for d in (*rs.CODE_DIRS, "config/symbols"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "signal_engine" / "gates.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "config" / "risk.yaml").write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    base = rs.stamp("MES", None, tmp_path)
    assert rs.stamp("MES", None, tmp_path) == base
    assert rs.stamp("MES", 365, tmp_path) != base
    assert rs.stamp("MNQ", None, tmp_path) != base
    (tmp_path / "scripts" / "notes.py").write_text("# not engine code\n", encoding="utf-8")
    (tmp_path / "signal_engine" / "__pycache__").mkdir()
    (tmp_path / "signal_engine" / "__pycache__" / "gates.cpython-312.pyc").write_bytes(b"\0")
    assert rs.stamp("MES", None, tmp_path) == base               # neither is an input
    (tmp_path / "signal_engine" / "gates.py").write_text("x = 2\n", encoding="utf-8")
    code_changed = rs.stamp("MES", None, tmp_path)
    assert code_changed["code_sha"] != base["code_sha"] and code_changed["config_sha"] == base["config_sha"]
    (tmp_path / "config" / "risk.yaml").write_text("a: 2\n", encoding="utf-8")
    config_changed = rs.stamp("MES", None, tmp_path)
    assert config_changed["config_sha"] != base["config_sha"]
    # read-only analysis settings are not a study input, but a look-alike name is
    (tmp_path / "config" / "analysis").mkdir()
    (tmp_path / "config" / "analysis" / "a4.yaml").write_text("split: x\n", encoding="utf-8")
    assert rs.stamp("MES", None, tmp_path) == config_changed
    (tmp_path / "config" / "analysis_extra.yaml").write_text("b: 1\n", encoding="utf-8")
    assert rs.stamp("MES", None, tmp_path)["config_sha"] != config_changed["config_sha"]


def test_code_changed_while_computing_discards_the_result(tmp_path, monkeypatch):
    for d in (*rs.CODE_DIRS, "config"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    src = tmp_path / "signal_engine" / "gates.py"
    src.write_text("x = 1\n", encoding="utf-8")

    def edits_code_midway(symbol, days):
        src.write_text("x = 2\n", encoding="utf-8")
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(rs, "ROOT", tmp_path)
    monkeypatch.setattr(rs, "_compute_fn", lambda: edits_code_midway)
    out = tmp_path / "out"
    assert rs.run_worker("MES", out, None) == rs.CODE_CHANGED
    assert not out.exists() or not any(out.iterdir())


# --- one at a time, by construction ------------------------------------------------------

def test_while_a_worker_runs_a_second_job_and_a_hand_started_worker_are_refused(tmp_path):
    out, log = tmp_path / "out", tmp_path / "events.jsonl"
    env = _env(log, FAKE_STUDY_HANG="MES")
    job = subprocess.Popen(_cmd("--symbols", "MES", "MNQ", "--out", out), cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        hung = _wait_for(log, "start", "MES")
        other_out = tmp_path / "other"          # locks are machine-wide, not per --out
        second = _run(_env(log), "--symbols", "MYM", "--out", other_out, timeout=60)
        assert second.returncode == rs.LOCKED, second.stdout + second.stderr
        assert "a setup-study job is already running" in second.stdout
        by_hand = _run(_env(log), "--worker", "MNQ", "--out", other_out, timeout=60)
        assert by_hand.returncode == rs.LOCKED, by_hand.stdout + by_hand.stderr
        assert _started(log) == ["MES"]         # neither computed anything
        assert not other_out.exists() or not any(other_out.iterdir())
        _kill(hung["pid"])
        job.communicate(timeout=60)
    finally:
        if job.poll() is None:
            job.kill()
    _record(log)
    assert job.returncode == rs.WORKER_FAILED

    # the killed worker's lock went with it: the next job starts cleanly
    log.unlink()
    r = _run(_env(log), "--symbols", "MES", "MNQ", "--out", out)
    _record(log)
    assert r.returncode == rs.OK, r.stdout + r.stderr
    assert _started(log) == ["MES", "MNQ"]


def test_a_job_refuses_to_start_workers_while_a_hand_started_worker_runs(tmp_path):
    out, log = tmp_path / "out", tmp_path / "events.jsonl"
    worker = subprocess.Popen(_cmd("--worker", "MES", "--out", out), cwd=ROOT,
                              env=_env(log, FAKE_STUDY_HANG="MES"),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        hung = _wait_for(log, "start", "MES")
        r = _run(_env(log), "--symbols", "MNQ", "MYM", "--out", out, timeout=60)
        assert r.returncode == rs.LOCKED, r.stdout + r.stderr
        assert "MNQ: LOCKED" in r.stdout and "MYM" not in r.stdout
        assert _started(log) == ["MES"]
        _kill(hung["pid"])
        worker.communicate(timeout=60)
    finally:
        if worker.poll() is None:
            worker.kill()
    _record(log)


def test_the_worker_command_takes_exactly_one_symbol(tmp_path):
    log = tmp_path / "events.jsonl"
    for extra in (["MES", "MNQ"], ["MES", "--symbols", "MNQ"]):
        r = _run(_env(log), "--worker", *extra, "--out", tmp_path / "out", timeout=60)
        assert r.returncode == rs.USAGE, r.stdout + r.stderr
    assert _started(log) == []


def test_no_two_computes_ever_overlapped():
    """Every compute interval recorded by the tests above, from any process, in time order."""
    if not ALL_EVENTS:
        pytest.skip("run with the rest of this file: it checks their recorded computes")
    open_, n = {}, 0
    for e in sorted(ALL_EVENTS, key=lambda e: e["t"]):
        if e["ev"] == "start":
            assert not open_, f"{e['sym']} started while {list(open_.values())} was computing"
            open_[e["pid"]] = e["sym"]
            n += 1
        else:                                   # "end", or "killed" for a hung worker
            open_.pop(e["pid"])
    assert not open_ and n >= 10
