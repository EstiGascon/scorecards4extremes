#!/usr/bin/env python3
"""Independent verification of plot_qq_ens_members.py correctness.

Reads ONE forecast-day parquet (memory-bounded), then for a chosen model +
orography reproduces the tool's control / member1 / full series TWO ways:
  (A) via the tool's own functions (discover/filter/_forecast_arrays/nanpercentile)
  (B) via a from-scratch, manual numpy path that re-derives the columns and
      percentiles independently.
Then asserts A == B and prints the numbers + sanity relationships.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent            # diagnostics/qq
_DIAG = _HERE.parent                               # diagnostics/
_REPO = _DIAG.parent                               # repository root
for _p in (_REPO, _REPO / "src", _DIAG, _HERE):
    sys.path.insert(0, str(_p))

from plot_qq import load_config, get_model_names          # noqa: E402
from plot_qq_ens_members import (                          # noqa: E402
    discover_model_source, filter_chunk, _forecast_arrays, _paired_stats,
)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default=str(_REPO / "configs/ensemble/config_10ff_ens_local_p98obsclim_aifsvsifs.yaml"))
_ap.add_argument("--model", default="fc2")
_ap.add_argument("--orog", default="low")
_ap.add_argument("--day-index", type=int, default=0)
_args = _ap.parse_args()
CONFIG = _args.config
MODEL = _args.model
OROG = _args.orog
DAY_FILE_INDEX = _args.day_index

cfg = load_config(CONFIG)
fc1, fc2 = get_model_names(cfg)
model_name = {"fc1": fc1, "fc2": fc2}[MODEL]
src = discover_model_source(cfg, fc1, fc2, MODEL)
member_cols = src["member_cols"]
print(f"Model {model_name} ({MODEL}); {len(member_cols)} member cols: "
      f"{member_cols[0]} .. {member_cols[-1]}")

# --- read ONE day file, filter identically to the tool ----------------------
fp = src["day_files"][DAY_FILE_INDEX]
raw = pd.read_parquet(fp, columns=src["needed"])
print(f"Read {fp.name}: {len(raw):,} rows")
df = filter_chunk(raw, cfg, member_cols, season=None, orog_type=OROG,
                  lead_time=None, start_date=cfg.get("start_date"),
                  end_date=cfg.get("end_date")).reset_index(drop=True)
print(f"After filter ({OROG}): {len(df):,} rows\n")

obs = df["obs_value"].to_numpy(dtype=np.float32)

# --- (A) tool path ----------------------------------------------------------
ctrl_col = member_cols[0] if member_cols[0].endswith("_0") else None
mem1_col = next((c for c in member_cols if c.endswith("_1")), None)
print(f"tool picked: control={ctrl_col}  member1={mem1_col}  full=ALL {len(member_cols)}\n")

levels = np.array([50, 90, 98, 99, 99.9])
seriesA = {
    "control": _forecast_arrays(df, [ctrl_col])[0],
    "member1": _forecast_arrays(df, [mem1_col])[0],
    "full":    _forecast_arrays(df, member_cols)[0],
}
pctA = {k: np.nanpercentile(v, levels) for k, v in seriesA.items()}

# --- (B) fully independent manual path --------------------------------------
# Re-derive the exact expected column names WITHOUT the tool's helpers.
manual_ctrl_name = f"{MODEL}_member_0"
manual_mem1_name = f"{MODEL}_member_1"
manual_full_names = [f"{MODEL}_member_{i}" for i in range(51)
                     if f"{MODEL}_member_{i}" in df.columns]

assert ctrl_col == manual_ctrl_name, f"CONTROL COLUMN MISMATCH: {ctrl_col} != {manual_ctrl_name}"
assert mem1_col == manual_mem1_name, f"MEMBER1 COLUMN MISMATCH: {mem1_col} != {manual_mem1_name}"
assert sorted(member_cols) == sorted(manual_full_names), "FULL SET MISMATCH"

manual_ctrl = df[manual_ctrl_name].to_numpy(dtype=np.float64)
manual_mem1 = df[manual_mem1_name].to_numpy(dtype=np.float64)
manual_full = df[manual_full_names].to_numpy(dtype=np.float64).ravel()

pctB = {
    "control": np.nanpercentile(manual_ctrl, levels),
    "member1": np.nanpercentile(manual_mem1, levels),
    "full":    np.nanpercentile(manual_full, levels),
}

# --- compare ----------------------------------------------------------------
print(f"{'level':>7} | {'obs':>8} | "
      + " | ".join(f"{k+'(A)':>10} {k+'(B)':>10}" for k in seriesA))
obs_p = np.nanpercentile(obs.astype(np.float64), levels)
ok = True
for i, lv in enumerate(levels):
    row = f"{lv:>7} | {obs_p[i]:>8.3f} | "
    cells = []
    for k in ("control", "member1", "full"):
        a, b = pctA[k][i], pctB[k][i]
        match = np.isclose(a, b, rtol=1e-4, atol=1e-3)
        ok = ok and match
        flag = "" if match else "  <-- MISMATCH"
        cells.append(f"{a:>10.3f} {b:>10.3f}{flag}")
    print(row + " | ".join(cells))

print("\nPaired stats (tool _paired_stats), pooled obs-vs-members:")
for k, cols in [("control", [ctrl_col]), ("member1", [mem1_col]), ("full", member_cols)]:
    mat = df[cols].to_numpy(dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[:, None]
    corr, bias, rmse = _paired_stats(mat, obs)
    # independent manual paired stats
    o = np.tile(obs.astype(np.float64), mat.shape[1])
    f = mat.astype(np.float64).ravel(order="F")
    m = np.isfinite(f) & np.isfinite(o)
    o2, f2 = o[m], f[m]
    mcorr = np.corrcoef(f2, o2)[0, 1]
    mbias = f2.mean() - o2.mean()
    mrmse = np.sqrt(np.mean((f2 - o2) ** 2))
    cm = np.isclose(corr, mcorr, atol=1e-4) and np.isclose(bias, mbias, atol=1e-3) and np.isclose(rmse, mrmse, atol=1e-3)
    ok = ok and cm
    print(f"  {k:>8}: tool rho={corr:.4f} bias={bias:+.3f} rmse={rmse:.3f} | "
          f"manual rho={mcorr:.4f} bias={mbias:+.3f} rmse={mrmse:.3f} "
          f"{'OK' if cm else '<-- MISMATCH'}")

# sanity relationships
print("\nSanity checks:")
print(f"  control != member1 at p99.9 (distinct members)? "
      f"{'YES' if not np.isclose(pctA['control'][4], pctA['member1'][4]) else 'NO (suspicious)'}")
print(f"  full pooled sample size = 51x single? "
      f"{seriesA['full'].size} == 51*{seriesA['control'].size} = {51*seriesA['control'].size}: "
      f"{seriesA['full'].size == 51*seriesA['control'].size}")
print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
