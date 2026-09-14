"""Shared feature-layer contract: config loading and column names.

Every feature module reads its thresholds through load_params() so that no
number is ever written into feature logic. Per-symbol overrides merge over the
global defaults, which is what lets Phase 4 grid-search a parameter for MCL
without disturbing MES.

Causality rule for this whole layer: a feature at bar i may depend only on
bars <= i, and on bar i only through data available once it has CLOSED. Any
rolling statistic used as a *baseline* for bar i excludes bar i itself --
otherwise the bar being tested contributes to the threshold it is tested
against, which quietly flatters every signal in backtest and cannot be
reproduced live.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path("config")

# canonical OHLCV columns carried through from the data layer
OHLCV = ["ts", "raw_symbol", "open", "high", "low", "close", "volume"]

# feature columns this layer adds
TRUE_RANGE = "true_range"
ATR = "atr"
ATR_MEAN = "atr_mean"
VOL_REGIME = "vol_regime"        # "high" | "normal" | "low"
CLV = "clv"
BODY = "body"
UPPER_WICK = "upper_wick"
LOWER_WICK = "lower_wick"
BODY_RATIO = "body_ratio"
VOLUME_BASELINE = "volume_baseline"
VOLUME_RATIO = "volume_ratio"
VOLUME_EXPANDED = "volume_expanded"
SESSION = "session"              # "rth" | "eth" | "continuous"


class UnresolvedParameter(RuntimeError):
    """A parameter parked pending a spec decision was actually read.

    Better to stop here than to invent a default for something the spec does
    not define -- see docs/open_questions.md.
    """


@dataclass(frozen=True)
class Params:
    """Merged parameter view for one symbol."""
    values: dict
    symbol: str = ""

    def get(self, path: str, default=None):
        """Fetch a dotted path, refusing sentinels for undecided questions."""
        node = self.values
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not None:
                    return default
                raise KeyError(f"{self.symbol}: no parameter {path!r}")
            node = node[part]
        if isinstance(node, str) and node in ("UNRESOLVED", "UNDEFINED_IN_SPEC"):
            raise UnresolvedParameter(
                f"{self.symbol}: parameter {path!r} is {node} -- the spec does "
                "not define it. See docs/open_questions.md; do not guess."
            )
        return node


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_params(symbol: str | None = None, config_dir: Path | None = None) -> Params:
    """Global params.yaml, with config/symbols/<SYM>.yaml `params:` merged over."""
    cdir = config_dir or CONFIG_DIR
    with open(cdir / "params.yaml") as fh:
        values = yaml.safe_load(fh)

    if symbol:
        p = cdir / "symbols" / f"{symbol}.yaml"
        if not p.exists():
            p = cdir / "symbols" / "full_size" / f"{symbol}.yaml"
        with open(p) as fh:
            scfg = yaml.safe_load(fh)
        values = _deep_merge(values, scfg.get("params", {}))
        values["_symbol"] = scfg
    return Params(values=values, symbol=symbol or "")


def tick_size(params: Params) -> float:
    return float(params.values["_symbol"]["contract_spec"]["tick_size"])


def atr_buffer(params: Params, atr_value: float, min_ticks: int,
               atr_multiple: float) -> float:
    """The spec's recurring `max(N ticks, k x ATR)` construction (S4, S6, S19)."""
    return max(min_ticks * tick_size(params), atr_multiple * atr_value)
