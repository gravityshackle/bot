"""Derive each instrument's RTH window from where its volume actually is.

IB's liquidHours reports the same generic 08:30-16:00 CT for every one of these
instruments -- including MET, which has no session concept at all -- so it is a
platform default, not instrument-specific truth. The spec's answer, originally
scoped to GC/SI and now applied to all six non-MET symbols, is to measure the
session directly.

Method: bucket every bar by its offset from the session open, average volume
per bucket across all sessions, then find the SHORTEST contiguous run of
buckets holding at least target_volume_share of an average session's volume.
Shortest-window-for-a-target beats highest-volume-fixed-window because it does
not require assuming a session length up front -- the data decides both the
length and the placement.

Offsets are measured from the session open rather than from midnight so the
search space is linear instead of wrapping, and so DST never shifts a bucket.

Run with --write to persist results into the symbol configs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pipeline import build_continuous, load_data_config  # noqa: E402
from data.resample import session_open_utc  # noqa: E402

OUT = Path("plots")


def volume_profile(bars: pd.DataFrame, scfg: dict, granularity: int):
    """Mean volume per bucket, indexed by minutes since the session open."""
    tz, boundary = scfg["session"]["timezone"], scfg["day_boundary"]
    origin = session_open_utc(bars["trade_date"], tz, boundary).reset_index(drop=True)
    elapsed = (bars["ts"].reset_index(drop=True) - origin).dt.total_seconds() / 60.0
    bucket = (elapsed // granularity).astype("int64") * granularity

    df = pd.DataFrame({"bucket": bucket.values,
                       "volume": bars["volume"].values,
                       "trade_date": bars["trade_date"].values})
    n_sessions = df["trade_date"].nunique()
    profile = df.groupby("bucket")["volume"].sum() / n_sessions
    full = pd.Series(0.0, index=range(int(profile.index.min()),
                                      int(profile.index.max()) + granularity,
                                      granularity))
    full.update(profile)
    return full, n_sessions


def shortest_window_for_share(profile: pd.Series, target: float):
    """Smallest contiguous bucket run holding >= target of total volume."""
    v = profile.to_numpy(dtype=float)
    total = v.sum()
    if total <= 0:
        return None
    need = target * total
    csum = np.concatenate([[0.0], np.cumsum(v)])
    for length in range(1, len(v) + 1):
        sums = csum[length:] - csum[:-length]
        best = int(np.argmax(sums))
        if sums[best] >= need:
            return best, best + length, float(sums[best] / total)
    return None


def offset_to_local(minutes: float, scfg: dict) -> str:
    """Minutes since session open -> local wall-clock HH:MM."""
    hh, mm = (int(x) for x in scfg["day_boundary"].split(":"))
    total = (hh * 60 + mm + int(minutes)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def analyse(symbol: str, cfg: dict):
    series, scfg, _ = build_continuous(symbol, cfg)
    e = cfg["empirical_rth"]
    gran = int(e["granularity_minutes"])
    profile, n_sessions = volume_profile(series.bars, scfg, gran)

    if n_sessions < int(e["min_sessions_required"]):
        return {"symbol": symbol, "error":
                f"only {n_sessions} sessions, need {e['min_sessions_required']}"}

    res = shortest_window_for_share(profile, float(e["target_volume_share"]))
    if res is None:
        return {"symbol": symbol, "error": "no window reached the target share"}

    i0, i1, share = res
    buckets = profile.index.to_numpy()
    start_off, end_off = float(buckets[i0]), float(buckets[min(i1, len(buckets) - 1)])
    return {
        "symbol": symbol, "profile": profile, "n_sessions": n_sessions,
        "rth_open": offset_to_local(start_off, scfg),
        "rth_close": offset_to_local(end_off, scfg),
        "start_off": start_off, "end_off": end_off,
        "share": share, "length_min": end_off - start_off, "scfg": scfg,
    }


def plot_profiles(results, cfg):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return None
    fig, axes = plt.subplots(len(ok), 1, figsize=(13, 2.3 * len(ok)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, ok):
        p = r["profile"]
        ax.fill_between(p.index / 60.0, p.values, color="steelblue", alpha=0.35, lw=0)
        ax.plot(p.index / 60.0, p.values, color="steelblue", lw=0.8)
        ax.axvspan(r["start_off"] / 60.0, r["end_off"] / 60.0,
                   color="darkorange", alpha=0.22)
        ax.axvline(r["start_off"] / 60.0, color="darkorange", lw=1.3)
        ax.axvline(r["end_off"] / 60.0, color="darkorange", lw=1.3)
        ax.set_ylabel(r["symbol"], fontweight="bold")
        ax.text(0.995, 0.88,
                f"{r['rth_open']}–{r['rth_close']} CT   "
                f"{r['length_min'] / 60:.1f}h holds {r['share']:.0%} of volume",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="darkorange", fontweight="bold")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("hours since session open (17:00 CT)")
    tgt = cfg["empirical_rth"]["target_volume_share"]
    fig.suptitle(f"Empirical RTH — shortest window holding {tgt:.0%} of average "
                 f"session volume (IB liquidHours NOT used)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "empirical_rth_profiles.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


def write_back(results):
    import re
    for r in results:
        if "error" in r:
            continue
        p = Path(f"config/symbols/{r['symbol']}.yaml")
        s = p.read_text()
        s = re.sub(r"  rth_open: null", f"  rth_open: \"{r['rth_open']}\"", s)
        s = re.sub(r"  rth_close: null", f"  rth_close: \"{r['rth_close']}\"", s)
        s = re.sub(r"  rth_source: empirical_pending",
                   f"  rth_source: empirical_highest_volume_window   "
                   f"# {r['share']:.0%} of volume, {r['n_sessions']} sessions", s)
        p.write_text(s)
        print(f"  wrote {r['symbol']}: {r['rth_open']}-{r['rth_close']} CT")


def main() -> int:
    cfg = load_data_config()
    symbols = cfg["empirical_rth"]["applies_to"]
    results = [analyse(s, cfg) for s in symbols]

    print(f"target share: {cfg['empirical_rth']['target_volume_share']:.0%}   "
          f"granularity: {cfg['empirical_rth']['granularity_minutes']}min\n")
    print(f"{'sym':5} {'empirical RTH (CT)':<22} {'length':>7} {'share':>7} "
          f"{'sessions':>9}   vs IB liquidHours 08:30-16:00")
    print("-" * 95)
    for r in results:
        if "error" in r:
            print(f"{r['symbol']:5} ERROR: {r['error']}")
            continue
        same = r["rth_open"] == "08:30" and r["rth_close"] == "16:00"
        print(f"{r['symbol']:5} {r['rth_open'] + '-' + r['rth_close']:<22} "
              f"{r['length_min'] / 60:>6.1f}h {r['share']:>6.0%} {r['n_sessions']:>9}   "
              f"{'matches IB' if same else 'DIFFERS from IB'}")

    path = plot_profiles(results, cfg)
    if path:
        print(f"\nprofile plot -> {path}")
    if "--write" in sys.argv:
        print("\nwriting into symbol configs:")
        write_back(results)
    else:
        print("\n(dry run — pass --write to persist into config/symbols/*.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
