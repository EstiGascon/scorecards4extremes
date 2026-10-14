#!/usr/bin/env python
"""Exit 0 if a config's merged extraction is complete, 1 otherwise.

Checks that every expected per-forecast-day parquet
(<variable>_<fc1>_vs_<fc2>[_ens]_day<N>.parquet) exists in
extract_points.output_path — following symlinks, so a dangling link counts as
missing. Used by submit_job_quaver.sh (S4E_REQUIRE_EXTRACTED=1) to stop a
scoring-only run from silently re-extracting.

Usage:  python scripts/check_extracted.py <config.yaml>
"""
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def expected_files(config):
    variable = config["variable"]
    rd = config["read_data"]
    is_ensemble = config.get("mode") == "ensemble"
    base = f"{variable}_{rd['forecast_model1']['name']}_vs_{rd['forecast_model2']['name']}" \
        + ("_ens" if is_ensemble else "")
    if config.get("forecast_days"):
        days = sorted(set(config["forecast_days"]))
    else:
        days = sorted({math.ceil(int(s) / 24) for s in config["steps"]})
    out = Path(config["extract_points"]["output_path"])
    out = out if out.is_absolute() else REPO_ROOT / out
    return [out / f"{base}_day{d}.parquet" for d in days]


def main():
    config = yaml.safe_load(open(sys.argv[1]))
    missing = [p for p in expected_files(config) if not p.exists()]
    if missing:
        print(f"Missing {len(missing)} extracted file(s), e.g. {missing[0]}")
        return 1
    print(f"✓ Extraction complete for {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
