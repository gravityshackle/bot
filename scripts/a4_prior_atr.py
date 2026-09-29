"""Prior-session daily ATR for one instrument, for the A4 stricter check (cache only).

Usage (from the repo root; one process per instrument, nothing else heavy running):
    python scripts/a4_prior_atr.py SYM

Builds the engine's own daily frame (timeframes.build over the whole cached
series, as the setup study did) and writes analysis.a4.prior_session_atr of
it, the daily ATR known at each session's open, to
cache/analysis/prior_atr/SYM.pkl, keyed by trade date.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import a4  # noqa: E402
from data.pipeline import build_continuous, load_data_config  # noqa: E402
from features.schema import load_params  # noqa: E402
from signal_engine import timeframes  # noqa: E402

OUT = Path("cache") / "analysis" / "prior_atr"


def main(sym: str) -> int:
    t0 = time.time()
    series, scfg, _ = build_continuous(sym, load_data_config())
    del _
    params = load_params(sym)
    period = int(params.get("atr.period"))
    daily = timeframes.build(sym, series.bars, scfg, params).frame("daily")
    prior = a4.prior_session_atr(daily, period)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".{sym}.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(prior, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(OUT / f"{sym}.pkl")
    info = {"symbol": sym, "atr_period": period, "sessions": int(len(prior)),
            "with_value": int(prior.notna().sum()), "first": str(prior.index[0]), "last": str(prior.index[-1]),
            "bars": int(len(series.bars)), "seconds": round(time.time() - t0, 1)}
    (OUT / f"{sym}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{sym}: {info['sessions']:,} sessions ({info['with_value']:,} with a prior ATR), "
          f"{info['first']} .. {info['last']}, {info['seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/a4_prior_atr.py SYM")
    sys.exit(main(sys.argv[1]))
