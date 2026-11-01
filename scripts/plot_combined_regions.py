#!/usr/bin/env python
"""Draw one scorecard combining NH-extratropics orography columns and a tropics column.

Reads the scores_by_leadtime CSVs already written by two scoring runs that share
one extraction (see configs/ensemble/*_nhtrop_* configs):
  - an NH-extratropics run split by orography (flat / hilly / complex)
  - a tropics run with no orography split
and re-plots every score as a single multi-column heatmap
(NH FLAT | NH HILLY | NH COMPLEX | TROPICS) through plot.run_step9, so styling,
colour limits and significance markers are identical to the normal scorecards.

Usage:
    python scripts/plot_combined_regions.py <nh_config.yaml> <tropics_config.yaml> <output_dir>
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import plot  # noqa: E402


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    p = Path(cfg["save"]["output_directory"])
    return p if p.is_absolute() else REPO_ROOT / p


def _read_by_leadtime(cfg, orog_type):
    """Locate scores_by_leadtime_<var>[_<thr>][_<orog>].csv for one condition."""
    d = _results_dir(cfg)
    var = cfg["variable"]
    pattern = f"scores_by_leadtime_{var}*_{orog_type}.csv" if orog_type else f"scores_by_leadtime_{var}*.csv"
    files = sorted(d.glob(pattern))
    if not orog_type:  # exclude orography-suffixed files when looking for the unsplit one
        orogs = (cfg["filter"].get("orography_ranges") or {}).keys()
        files = [f for f in files if not any(f.stem.endswith(f"_{o}") for o in orogs)]
    if len(files) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {d}, found {[f.name for f in files]}")
    return pd.read_csv(files[0])


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    nh_cfg, tr_cfg = _load(sys.argv[1]), _load(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = [{"orog_type": f"NH\n{o}", "season": None,
                "results": {"by_leadtime": _read_by_leadtime(nh_cfg, o)}}
               for o in nh_cfg["filter"]["orography_type"]]
    columns.append({"orog_type": "Tropics", "season": None,
                    "results": {"by_leadtime": _read_by_leadtime(tr_cfg, None)}})

    plot_cfg = dict(nh_cfg)
    plot_cfg["filter"] = dict(nh_cfg["filter"], area="nh_extratropics_tropics")
    model_names = {"fc1_name": nh_cfg["read_data"]["forecast_model1"]["name"],
                   "fc2_name": nh_cfg["read_data"]["forecast_model2"]["name"]}
    plot.run_step9(plot_cfg, columns, None, out_dir, model_names, season=None)
    print(f"✓ Combined NH-extratropics + tropics scorecards written to {out_dir}")


if __name__ == "__main__":
    main()
