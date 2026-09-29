"""Phase 4.4: A4. Does reward/risk quality reward better trades, and is the RR cap right?

The question, and how it must be answered, is in docs/phase4_questions.md (A4).
This is the per-instrument analysis of the setup study's results
(cache/setup_study/), done in-sample and out-of-sample separately:

  - RR bands [2, 2.5) [2.5, 3) [3, 4) [4, 6) [6, inf). Bands below
    targets.rr_cap are the component's live range. The rest is saturated:
    the component scores them all 1.0.
  - Score quintiles, with edges from in-sample sized setups only.
  - Per cell: fill rate over sized setups. Win rate, expectancy, scratch rate
    and the R distribution over filled trades, on the verdict metric.
    Discarded setups (0 contracts) are counted but never enter an outcome.

The verdict metric is r_gross: after slippage, before fees (decided 2026-09-29;
the question is whether the component picks better setups, whatever the cost
structure). It sits between two other layers, and each cell and trend reports
all three:
  signal R  exit at its intended price (exit_reference): no slippage, no fees
  r_gross   the actual fill: slippage in, fees out
  r_net     fees in too: the tradeable result at 1 contract
Cost attribution per trend: "slippage" when signal R and r_gross directions
differ, "fees" when r_gross and r_net differ. Both are tight-stop effects, but
they are different effects and are kept separate from each other and from the
sizing check.

Three answers per instrument and period, each a trend with a bootstrap
interval ("flat" when the interval spans zero):
  component  R on RR band index, over the live bands
  cap        R, saturated bands vs the top live band (a 0/1 trend: the
             difference in means)
  score      R on score quintile index

For the sizing-shift instruments (MNQ, MGC, SIL) an answer only stands if it
also survives the stricter check. Stop width is measured in multiples of the
previous completed session's daily ATR. The in-sample trend is recomputed:
  restricted  on in-sample trades inside the OOS stop-width range
  stratified  compared within OOS stop-width quantile slices, the slices
              weighted to the OOS mix
If either loses the raw in-sample direction, the verdict is "unresolved:
sizing". Restriction alone is not enough: if high-RR setups have wider stops
inside the range too, the confound survives it. Stratification is what
removes it.

Every number comes from config/analysis/a4.yaml or params.yaml.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.continuous_contract import trade_date
from features.levels import daily_atr_by_date

CONFIG_PATH = Path("config") / "analysis" / "a4.yaml"
IS, OOS = "IS", "OOS"
LAYERS = ("r_signal", "r_gross", "r_net")


@dataclass(frozen=True)
class A4Config:
    split: pd.Timestamp
    score_quantiles: int
    rr_edges: tuple
    rr_cap: float
    draws: int
    seed: int
    confidence: float
    min_cell_trades: int
    tail_top_fraction: float
    sizing_shift_symbols: tuple
    directional_only_symbols: tuple
    support_quantiles: tuple
    reweight_bins: int
    atr_period: int
    outcome: str = "r_gross"

    def validate(self) -> "A4Config":
        e = list(self.rr_edges)
        if e != sorted(e) or len(set(e)) != len(e):
            raise ValueError(f"rr_band_edges must increase strictly: {e}")
        if self.rr_cap not in e:
            raise ValueError(f"rr_cap {self.rr_cap} must be one of rr_band_edges {e}")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if self.outcome not in LAYERS:
            raise ValueError(f"outcome must be one of {LAYERS}")
        return self

    @classmethod
    def load(cls, symbol: str, path: Path = CONFIG_PATH) -> "A4Config":
        from features.schema import load_params

        with open(path, encoding="utf-8") as fh:
            c = yaml.safe_load(fh)
        p = load_params(symbol)
        edges = tuple(float(x) for x in c["rr_band_edges"])
        if edges[0] != float(p.get("targets.min_reward_risk")):
            raise ValueError("the first RR band edge must be targets.min_reward_risk")
        sc = c["stricter_check"]
        return cls(split=pd.Timestamp(c["split_date"], tz="UTC"), score_quantiles=int(c["score_quantiles"]),
                   rr_edges=edges, rr_cap=float(p.get("targets.rr_cap")),
                   draws=int(c["bootstrap"]["draws"]), seed=int(c["bootstrap"]["seed"]),
                   confidence=float(c["bootstrap"]["confidence"]),
                   min_cell_trades=int(c["min_cell_trades"]), tail_top_fraction=float(c["tail_top_fraction"]),
                   sizing_shift_symbols=tuple(c["sizing_shift_symbols"]),
                   directional_only_symbols=tuple(c["directional_only_symbols"]),
                   support_quantiles=tuple(float(q) for q in sc["support_quantiles"]),
                   reweight_bins=int(sc["reweight_bins"]),
                   atr_period=int(p.get("atr.period")), outcome=str(c["outcome"])).validate()


# --- bands ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x:g}"


def _labels(edges) -> list[str]:
    b = list(edges) + [math.inf]
    return [f"[{_fmt(lo)}, {_fmt(hi)})" for lo, hi in zip(b[:-1], b[1:])]


def rr_band(rr: pd.Series, edges) -> pd.Series:
    """Left-closed RR bands with an open top. RR below the first edge is an error:
    gate 4 never passes one, so its presence means the input is wrong."""
    rr = pd.Series(rr, dtype="float64")
    if rr.isna().any() or (rr < edges[0]).any():
        raise ValueError(f"RR below the first band edge {edges[0]} (or missing)")
    return pd.cut(rr, bins=list(edges) + [math.inf], right=False, labels=_labels(edges))


def live_bands(cfg: A4Config) -> list[str]:
    lab = _labels(cfg.rr_edges)
    return [l for l, lo in zip(lab, cfg.rr_edges) if lo < cfg.rr_cap]


def saturated_bands(cfg: A4Config) -> list[str]:
    lab = _labels(cfg.rr_edges)
    return [l for l, lo in zip(lab, cfg.rr_edges) if lo >= cfg.rr_cap]


def score_edges(is_scores: pd.Series, q: int) -> np.ndarray:
    return np.quantile(np.asarray(is_scores, dtype="float64"), np.linspace(0, 1, q + 1)[1:-1])


def score_band(scores: pd.Series, edges: np.ndarray) -> pd.Series:
    """1..q against fixed edges; values outside the in-sample range clamp to the ends."""
    return pd.Series(1 + np.searchsorted(edges, np.asarray(scores, dtype="float64"), side="right"),
                     index=getattr(scores, "index", None))


# --- flags and populations -------------------------------------------------------------

def signal_r(df: pd.DataFrame) -> pd.Series:
    """R had the exit filled at its intended price (exit_reference), on the same
    initial risk r_gross uses (entry fill to the resting stop order). Direction
    needs no sign: it flips the move and the risk alike, so the ratio holds."""
    return (df["exit_reference"] - df["entry_fill"]) / (df["entry_fill"] - df["stop_order"])


def add_flags(df: pd.DataFrame, outcome: str = "r_gross") -> pd.DataFrame:
    out = df.copy()
    out["sized"] = out["contracts"] > 0
    out["filled"] = out["entry_status"].astype("object").eq("filled").fillna(False).astype(bool)
    out["win"] = out["filled"] & (out[outcome] > 0)
    out["scratch"] = out["filled"] & out["path"].astype("object").eq("scratch").fillna(False).astype(bool)
    return out


def prepare(df: pd.DataFrame, cfg: A4Config) -> pd.DataFrame:
    df = df.copy()
    df["r_signal"] = signal_r(df).where(df["entry_status"].astype("object").eq("filled").fillna(False))
    p = add_flags(df, cfg.outcome)
    bad = int((p["filled"] & ~p["sized"]).sum())
    if bad:
        # the study never fills a discarded setup; if one appears, the input is wrong
        raise ValueError(f"{bad} setups sized to 0 contracts are marked filled")
    p["period"] = np.where(p["ts"] < cfg.split, IS, OOS)
    p["rr_band"] = rr_band(p["rr"], cfg.rr_edges)
    base = p.loc[(p["period"] == IS) & p["sized"], "score"]
    if base.empty:
        raise ValueError("no in-sample sized setups to set score edges from")
    p["score_q"] = score_band(p["score"], score_edges(base, cfg.score_quantiles)).to_numpy()
    return p


def tail_share(r: pd.Series, top_fraction: float) -> float:
    """Share of all positive R from the top fraction of winners (at least one winner)."""
    pos = np.sort(np.asarray(r, dtype="float64")[np.asarray(r) > 0])[::-1]
    if pos.size == 0:
        return float("nan")
    k = max(1, math.ceil(top_fraction * pos.size))
    return float(pos[:k].sum() / pos.sum())


def _interval(stats: np.ndarray, confidence: float) -> tuple[float, float]:
    a = (1 - confidence) / 2
    lo, hi = np.quantile(stats, [a, 1 - a])
    return float(lo), float(hi)


def _boot_mean(r: np.ndarray, cfg: A4Config, rng: np.random.Generator) -> tuple[float, float]:
    if r.size < 2:
        return float("nan"), float("nan")
    means = np.empty(cfg.draws)
    chunk = max(1, 2_000_000 // r.size)
    for s in range(0, cfg.draws, chunk):
        k = min(chunk, cfg.draws - s)
        means[s:s + k] = r[rng.integers(0, r.size, (k, r.size))].mean(axis=1)
    return _interval(means, cfg.confidence)


def cell_stats(g: pd.DataFrame, cfg: A4Config, rng: np.random.Generator) -> dict:
    sized = g[g["sized"]]
    f = sized[sized["filled"]]
    r = f[cfg.outcome].to_numpy(dtype="float64")
    fee_share = (f["fees_usd"] / f["risk_usd"]).to_numpy(dtype="float64")
    lo, hi = _boot_mean(r, cfg, rng)
    n = len(f)
    return {
        "setups": len(g), "discarded": int((~g["sized"]).sum()), "sized": len(sized), "filled": n,
        "fill_rate": n / len(sized) if len(sized) else float("nan"),
        "win_rate": float(f["win"].mean()) if n else float("nan"),
        "scratch_rate": float(f["scratch"].mean()) if n else float("nan"),
        "exp_r": float(r.mean()) if n else float("nan"), "exp_lo": lo, "exp_hi": hi,
        "median_r": float(np.median(r)) if n else float("nan"),
        "p10_r": float(np.quantile(r, 0.1)) if n else float("nan"),
        "p90_r": float(np.quantile(r, 0.9)) if n else float("nan"),
        "tail_share": tail_share(r, cfg.tail_top_fraction),
        "thin": n < cfg.min_cell_trades,
        "exp_signal": float(f["r_signal"].mean()) if n else float("nan"),
        "exp_net": float(f["r_net"].mean()) if n else float("nan"),
        "usd_net": float(f["net_usd"].mean()) if n else float("nan"),
        "fee_over_risk_median": float(np.median(fee_share)) if n else float("nan"),
        "fees_ge_risk": float((fee_share >= 1).mean()) if n else float("nan"),
    }


def band_table(p: pd.DataFrame, by: str, cfg: A4Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for period in (IS, OOS):
        sub = p[p["period"] == period]
        for band, g in sub.groupby(by, observed=False, sort=True):
            rows.append({"period": period, by: band, **cell_stats(g, cfg, rng)})
    return pd.DataFrame(rows)


# --- stop width in ATR ---------------------------------------------------------------------

def prior_session_atr(daily: pd.DataFrame, period: int) -> pd.Series:
    """Daily ATR known at each session's open: the engine's own daily ATR (S3),
    by trade date, shifted one session so a session never scales its own stops."""
    return daily_atr_by_date(daily, period).shift(1)


def stop_width_atr(df: pd.DataFrame, prior: pd.Series, scfg: dict) -> pd.Series:
    td = trade_date(df["ts"], scfg["session"]["timezone"], scfg["day_boundary"])
    atr_at = prior.reindex(pd.Index(td)).to_numpy(dtype="float64")
    return pd.Series((df["entry"] - df["stop_order"]).abs().to_numpy() / atr_at, index=df.index)


def support_mask(is_x: pd.Series, oos_x: pd.Series, quantiles) -> pd.Series:
    lo, hi = np.quantile(np.asarray(oos_x, dtype="float64"), quantiles)
    return (is_x >= lo) & (is_x <= hi)


def _bins(x, edges, n):
    return np.clip(np.searchsorted(edges, np.asarray(x, dtype="float64"), side="right") - 1, 0, n - 1)


def strata(is_x: pd.Series, oos_x: pd.Series, n: int) -> np.ndarray:
    edges = np.quantile(np.asarray(oos_x, dtype="float64"), np.linspace(0, 1, n + 1))
    return _bins(is_x, edges, n)


def reweight(is_x: pd.Series, oos_x: pd.Series, n: int) -> np.ndarray:
    """Per in-sample trade: OOS share of its stop-width bin / in-sample share."""
    edges = np.quantile(np.asarray(oos_x, dtype="float64"), np.linspace(0, 1, n + 1))
    b_is, b_oos = _bins(is_x, edges, n), _bins(oos_x, edges, n)
    s_is = np.bincount(b_is, minlength=n) / len(b_is)
    s_oos = np.bincount(b_oos, minlength=n) / len(b_oos)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s_is > 0, s_oos / s_is, 0.0)
    return ratio[b_is]


def oos_coverage(is_x: pd.Series, oos_x: pd.Series, n: int) -> float:
    """Share of OOS trades whose stop-width bin holds any in-sample trade. The
    stratified trend says nothing about the rest, so this is reported with it."""
    edges = np.quantile(np.asarray(oos_x, dtype="float64"), np.linspace(0, 1, n + 1))
    has_is = np.bincount(_bins(is_x, edges, n), minlength=n) > 0
    return float(has_is[_bins(oos_x, edges, n)].mean())


def ks(a: pd.Series, b: pd.Series) -> dict:
    """Two-sample Kolmogorov-Smirnov D, with the asymptotic p-value."""
    a = np.sort(np.asarray(a, dtype="float64")[~np.isnan(np.asarray(a, dtype="float64"))])
    b = np.sort(np.asarray(b, dtype="float64")[~np.isnan(np.asarray(b, dtype="float64"))])
    grid = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size
                            - np.searchsorted(b, grid, side="right") / b.size)))
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 1.0 if d == 0 else 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return {"D": d, "p": float(min(1.0, max(0.0, p))), "n_a": int(a.size), "n_b": int(b.size)}


# --- trends --------------------------------------------------------------------------------

def _demean(v, w, s):
    if s is None:
        return v - np.sum(w * v) / np.sum(w)
    sw = np.bincount(s, weights=w)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.bincount(s, weights=w * v) / sw
    return v - np.nan_to_num(m)[s]


def slope(x, y, w=None, s=None) -> float:
    """Weighted least-squares slope of y on x, within strata `s` if given."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    w = np.ones_like(x) if w is None else np.asarray(w, dtype="float64")
    xd, yd = _demean(x, w, s), _demean(y, w, s)
    den = np.sum(w * xd * xd)
    return float(np.sum(w * xd * yd) / den) if den > 0 else float("nan")


def trend(x, y, cfg: A4Config, rng: np.random.Generator, w=None, s=None) -> dict:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    w = np.ones_like(x) if w is None else np.asarray(w, dtype="float64")
    s = None if s is None else np.asarray(s)
    keep = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep]
    s = None if s is None else s[keep]
    vals, counts = np.unique(x, return_counts=True)
    out = {"n": int(x.size), "levels": {float(v): int(c) for v, c in zip(vals, counts)}}
    if (counts >= cfg.min_cell_trades).sum() < 2:
        return {**out, "slope": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "direction": "insufficient"}
    point = slope(x, y, w, s)
    boots = np.empty(cfg.draws)
    for i in range(cfg.draws):
        k = rng.integers(0, x.size, x.size)
        boots[i] = slope(x[k], y[k], w[k], None if s is None else s[k])
    boots = boots[np.isfinite(boots)]
    lo, hi = _interval(boots, cfg.confidence)
    direction = "up" if lo > 0 else "down" if hi < 0 else "flat"
    return {**out, "slope": point, "lo": lo, "hi": hi, "direction": direction}


def cap_gap(band: pd.Series, r, cfg: A4Config, rng: np.random.Generator) -> dict:
    """Mean r_net of the saturated bands minus the top live band (a 0/1 trend)."""
    t = trend(_cap_x(band, cfg), r, cfg, rng)
    return {**t, "gap": t["slope"]}


def _cap_x(band: pd.Series, cfg: A4Config) -> np.ndarray:
    b = pd.Series(band).astype("object")
    return np.where(b == live_bands(cfg)[-1], 0.0, np.where(b.isin(saturated_bands(cfg)), 1.0, np.nan))


def _live_x(band: pd.Series, cfg: A4Config) -> np.ndarray:
    code = {l: float(i) for i, l in enumerate(live_bands(cfg))}
    return pd.Series(band).astype("object").map(code).to_numpy(dtype="float64")


def _score_x(q: pd.Series, cfg: A4Config) -> np.ndarray:
    return pd.Series(q).to_numpy(dtype="float64") - 1


def verdict(is_dir: str, oos_dir: str, matched=None) -> str:
    if matched is not None and any(m != is_dir for m in matched):
        return "unresolved: sizing"
    if "insufficient" in (is_dir, oos_dir):
        return "insufficient data"
    return f"consistent: {is_dir}" if is_dir == oos_dir else "not replicated"


def _answer(p: pd.DataFrame, cfg: A4Config, symbol: str, x_of, salt: int) -> dict:
    f = p[p["filled"]]
    fi, fo = f[f["period"] == IS], f[f["period"] == OOS]
    rng = lambda k: np.random.default_rng([cfg.seed, salt, k])          # noqa: E731
    xi, yi = x_of(fi), fi[cfg.outcome].to_numpy(dtype="float64")
    res = {"IS": trend(xi, yi, cfg, rng(0)),
           "OOS": trend(x_of(fo), fo[cfg.outcome].to_numpy(dtype="float64"), cfg, rng(1)),
           "directional_only": symbol in cfg.directional_only_symbols, "outcome": cfg.outcome}
    layers = {}
    for k, (period, g) in enumerate(((IS, fi), (OOS, fo))):
        layers[period] = {m: (res[period] if m == cfg.outcome else
                              trend(x_of(g), g[m].to_numpy(dtype="float64"), cfg, rng(10 + 3 * k + j)))
                          for j, m in enumerate(LAYERS)}
    res["layers"] = layers
    res["attribution"] = {
        "slippage": {pd_: layers[pd_]["r_signal"]["direction"] != layers[pd_]["r_gross"]["direction"]
                     for pd_ in (IS, OOS)},
        "fees": {pd_: layers[pd_]["r_gross"]["direction"] != layers[pd_]["r_net"]["direction"]
                 for pd_ in (IS, OOS)},
        "sizing": None,
    }
    matched = None
    if symbol in cfg.sizing_shift_symbols:
        si, so = fi["stop_atr"].to_numpy(dtype="float64"), fo["stop_atr"].to_numpy(dtype="float64")
        ok_i, so = np.isfinite(si), so[np.isfinite(so)]
        m = ok_i & support_mask(pd.Series(si), pd.Series(so), cfg.support_quantiles).to_numpy()
        res["IS_restricted"] = trend(xi[m], yi[m], cfg, rng(2))
        st = strata(pd.Series(si[ok_i]), pd.Series(so), cfg.reweight_bins)
        w = reweight(pd.Series(si[ok_i]), pd.Series(so), cfg.reweight_bins)
        res["IS_stratified"] = {**trend(xi[ok_i], yi[ok_i], cfg, rng(3), w=w, s=st),
                                "oos_coverage": oos_coverage(pd.Series(si[ok_i]), pd.Series(so),
                                                             cfg.reweight_bins)}
        matched = (res["IS_restricted"]["direction"], res["IS_stratified"]["direction"])
        res["attribution"]["sizing"] = any(m != res["IS"]["direction"] for m in matched)
    res["verdict"] = verdict(res["IS"]["direction"], res["OOS"]["direction"], matched)
    return res


def rr_component_answer(p: pd.DataFrame, cfg: A4Config, symbol: str) -> dict:
    return _answer(p, cfg, symbol, lambda f: _live_x(f["rr_band"], cfg), salt=1)


def cap_answer(p: pd.DataFrame, cfg: A4Config, symbol: str) -> dict:
    return _answer(p, cfg, symbol, lambda f: _cap_x(f["rr_band"], cfg), salt=2)


def score_answer(p: pd.DataFrame, cfg: A4Config, symbol: str) -> dict:
    return _answer(p, cfg, symbol, lambda f: _score_x(f["score_q"], cfg), salt=3)


def survivor_shift(p: pd.DataFrame) -> dict:
    """Filled trades' score, RR and ATR-scaled stop width: in-sample vs out-of-sample."""
    f = p[p["filled"]]
    out = {}
    for col in ("score", "rr", "stop_atr"):
        if col not in f:
            continue
        a, b = f.loc[f["period"] == IS, col], f.loc[f["period"] == OOS, col]
        q = lambda v: [float(x) for x in np.nanquantile(v, [0.1, 0.5, 0.9])]   # noqa: E731
        out[col] = {"IS_p10_50_90": q(a), "OOS_p10_50_90": q(b), **ks(a, b)}
    return out
