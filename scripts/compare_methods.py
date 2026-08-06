#!/usr/bin/env python3
"""Compare method1 (local) / method2 (mars+stvl) / method3 (quaver_extract) scores.

For each variable and model-pair config, load the per-condition `overall_scores_*.csv`
files from each method's results dir and report the main per-model scores side by side,
so we can confirm the STVL one-date-per-call fix made method3 agree with method1/2.
"""
import glob
import os
import re
import pandas as pd

RESULTS = "results"

# (variable, config-suffix): the three method dirs share this suffix.
CASES = [
    ("2t",   "iekm_vs_ifs"),
    ("2t",   "ens_ifs_vs_aifs"),
    ("10ff", "iekm_vs_ifs"),
    ("10ff", "ens_ifs_vs_aifs"),
]

METHODS = {
    "m1_local":   "method1_local",
    "m2_stvl":    "method2_marsstvl",
    "m3_quaver":  "method3_quaverextract",
}

# Key per-model score columns to show (det vs ens differ; we just show whatever exists).
DET_COLS = ["twMAE_fc1", "twMAE_fc2", "bias_fc1", "bias_fc2",
            "mae_fc1", "rmse_fc1", "n_samples"]
ENS_COLS = ["twCRPS_fc1", "twCRPS_fc2", "twMAE_fc1", "twMAE_fc2",
            "ens_mean_bias_fc1", "ens_mean_bias_fc2", "n_samples"]


def cond_key(fname):
    """overall_scores_2t_0.0C_flat.csv -> '0.0C_flat'"""
    m = re.search(r"overall_scores_[^_]+_(.+)\.csv$", os.path.basename(fname))
    return m.group(1) if m else os.path.basename(fname)


def load_conditions(var, suffix):
    """Return {cond_key: {method_label: row_series}}."""
    out = {}
    for label, mdir in METHODS.items():
        d = f"{RESULTS}/{var}_{mdir}_{suffix}"
        for f in sorted(glob.glob(f"{d}/overall_scores_{var}_*.csv")):
            k = cond_key(f)
            df = pd.read_csv(f)
            if df.empty:
                continue
            out.setdefault(k, {})[label] = df.iloc[0]
    return out


def main():
    for var, suffix in CASES:
        conds = load_conditions(var, suffix)
        print("=" * 92)
        print(f"### {var}  |  {suffix}")
        print("=" * 92)
        if not conds:
            print("  (no results found)\n")
            continue
        # pick columns present in the first available row
        sample = next(iter(next(iter(conds.values())).values()))
        cols = [c for c in (ENS_COLS if "twCRPS_fc1" in sample.index else DET_COLS)
                if c in sample.index]
        for k in sorted(conds):
            rows = conds[k]
            print(f"\n-- {k} --")
            header = f"  {'score':<20}" + "".join(f"{lbl:>14}" for lbl in METHODS)
            header += f"{'max|Δ| m1 vs':>16}"
            print(header)
            for c in cols:
                vals = {lbl: (rows[lbl][c] if lbl in rows and c in rows[lbl].index else None)
                        for lbl in METHODS}
                line = f"  {c:<20}"
                for lbl in METHODS:
                    v = vals[lbl]
                    line += f"{v:>14.5f}" if isinstance(v, (int, float)) else f"{'--':>14}"
                # max abs deviation of m2/m3 from m1
                base = vals.get("m1_local")
                devs = [abs(vals[l] - base) for l in ("m2_stvl", "m3_quaver")
                        if isinstance(vals[l], (int, float)) and isinstance(base, (int, float))]
                line += f"{max(devs):>16.5f}" if devs else f"{'--':>16}"
                print(line)
        print()


if __name__ == "__main__":
    main()
