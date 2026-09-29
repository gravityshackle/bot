"""A stand-in for run_setup_study.compute, for tests/test_run_setup_study.py.

It is selected through SETUP_STUDY_COMPUTE, runs inside the real worker
process, and never loads data. It appends an event to FAKE_STUDY_LOG (JSON
lines) when it starts and ends. If FAKE_STUDY_HANG names this symbol, it
hangs after "start" so the test can kill the worker in the middle of an
instrument. The frame it returns carries the worker's pid, so a re-run gives
different bytes and a test can tell whether a result was redone or left alone.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd


def _log(ev: str, symbol: str) -> None:
    with open(os.environ["FAKE_STUDY_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ev": ev, "sym": symbol, "pid": os.getpid(), "t": time.time()}) + "\n")


def compute(symbol: str, days):
    _log("start", symbol)
    if os.environ.get("FAKE_STUDY_HANG") == symbol:
        time.sleep(120)
    time.sleep(float(os.environ.get("FAKE_STUDY_SLEEP", "0.3")))
    _log("end", symbol)
    return pd.DataFrame({"symbol": [symbol] * 3, "days": [days] * 3, "pid": [os.getpid()] * 3,
                         "r_net": [0.5, -1.0, 2.0]})
