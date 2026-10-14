#!/usr/bin/env python
"""Generate the ocean-coupling ensemble scorecard configs (Control vs Ocean A/B, GT, T).

Each experiment is an 8-member AIFS-family RD ensemble (stream=enfo, type=pf,
members 1-8, NO control member), N320 grid, tp in m. Two archives per model:
    FMA  (init 2026-02-02 .. 2026-04-28)   MJJA (init 2026-05-13 .. 2026-08-23)
(the common dates of the model-2 experiments and the Control; nothing exists
for 2026-04-29 .. 2026-05-12). Model 1 is always the Control.

Per (model, variable) ONE extraction over 20S-90N ('nh_extratropics_tropics'),
in 4 date chunks (2 per period, each with that period's expvers), merged, then
scored twice from the same data:
    *_nhextrop : filter.area nh_extratropics, flat / hilly / complex
    *_tropics  : filter.area tropics, no orography split
and drawn as one NH-columns + tropics-column scorecard (scripts/plot_combined_regions.py).
mn2t reuses the mx2t extraction (identical 6-hourly 2t; only the daily min/max
differs, at scoring time) through symlinked day files.

Writes configs/ensemble/config_<var>_ens_quaver_<thr>_<model>vsctrl_FebAug2026_<region>.yaml,
chunk configs + post-processing scripts under configs/_chunks/, and prints the
watcher submission commands.  Usage:  python scripts/make_ocean_ens_configs.py
"""
import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ENS = REPO / "configs" / "ensemble"
CHUNKS = REPO / "configs" / "_chunks"

CONTROL = {"FMA": "jccs", "MJJA": "jcct"}
MODELS = {  # name: {period: expver}
    "oceanA": {"FMA": "jbyc", "MJJA": "jbuu"},
    "oceanB": {"FMA": "jcea", "MJJA": "jc0v"},
    "GT":     {"FMA": "jc0y", "MJJA": "jcec"},
    "T":      {"FMA": "jc8r", "MJJA": "jc8q"},
}
START, END = "2026-02-02", "2026-08-23"
CHUNK_RANGES = [("FMA", "2026-02-02", "2026-03-16"), ("FMA", "2026-03-17", "2026-04-28"),
                ("MJJA", "2026-05-13", "2026-07-03"), ("MJJA", "2026-07-04", "2026-08-23")]

STEPS_6H = list(range(6, 241, 6))
STEPS_24H = list(range(24, 241, 24))
ORO_RANGES = {"flat": [0, 40], "hilly": [40, 120], "complex": [120, 3000]}
UPPER_QS = ["tw_quantile_score_q099", "tw_quantile_score_q0995", "tw_quantile_score_q0999"]
LOWER_QS = ["tw_quantile_score_q001", "tw_quantile_score_q0005", "tw_quantile_score_q0001"]


def scores(tail_qs):
    return ["twCRPS", "fCRPS", "CRPSS", "Brier", "BSS", "reliability",
            "tw_quantile_score", *tail_qs, "extreme_spread_skill_ratio"]


# variable: settings taken from the user's latest ensemble config of that variable
VARS = {
    "tp24": dict(thr="p99obsclim", steps=STEPS_24H, factor=1000.0, lapse=False, accum=24,
                 coastal=True, valid={"max_valid_precipitation": 800.0},
                 clim=("tp", 99, "above"), scores=scores(UPPER_QS)),
    "10ff": dict(thr="p99obsclim", steps=STEPS_6H, factor=1.0, lapse=False, accum=None,
                 coastal=False, valid={"min_valid_wind_speed": 0.0, "max_valid_wind_speed": 100.0},
                 clim=("10ff", 99, "above"), scores=scores(UPPER_QS)),
    "mx2t": dict(thr="p99obsclim", steps=STEPS_6H, factor=1.0, lapse=True, accum=None,
                 coastal=True, valid={"min_valid_temperature": -60.0, "max_valid_temperature": 60.0},
                 clim=("mx2t", 99, "above"), scores=scores(UPPER_QS)),
    "mn2t": dict(thr="p1obsclim", steps=STEPS_6H, factor=1.0, lapse=True, accum=None,
                 coastal=True, valid={"min_valid_temperature": -60.0, "max_valid_temperature": 60.0},
                 clim=("mn2t", 1, "below"), scores=scores(LOWER_QS)),
}
EXTRACTION_GROUPS = {"tp24": ["tp24"], "10ff": ["10ff"], "mx2t": ["mx2t", "mn2t"]}  # first = extracting var


def model_block(name, expver, factor):
    return {"name": name, "source": "quaver", "unit_conversion_factor": factor,
            "quaver": {"class": "rd", "expver": expver, "stream": "enfo", "type": "pf",
                       "number": "1/to/8"}}


def tag(var, model, region):
    return f"{var}_ens_quaver_{VARS[var]['thr']}_{model}vsctrl_FebAug2026_{region}"


def base_config(var, model, region, extract_dir):
    v = VARS[var]
    pname, pct, event = v["clim"]
    flt = {"lead_times": None, "season": None, "area": region_area(region),
           "orography_type": ["flat", "hilly", "complex"] if region == "nhextrop" else None,
           "orography_ranges": ORO_RANGES,
           "remove_coastal_stations": v["coastal"], "coastal_lsm_threshold": 0.9, **v["valid"]}
    return {
        "variable": var, "start_date": START, "end_date": END,
        "mode": "ensemble", "backend": "quaver_extract", "steps": v["steps"],
        "skip_extraction_if_exists": True, "skip_scoring_if_exists": False,
        "ensemble": {"n_members": 8, "include_control": False},
        "read_data": {
            # identities of the FMA archives; the MJJA ones are set in chunks 2-3
            "forecast_model1": model_block("control", CONTROL["FMA"], v["factor"]),
            "forecast_model2": model_block(model, MODELS[model]["FMA"], v["factor"]),
            "observation_source": "quaver", "quaver_obs": {"sources": ["synop", "hdobs"]}},
        "preprocess": {"wind_speed_from_components": False, "lapse_rate_correction": v["lapse"],
                       "lapse_rate": -0.0065, "precipitation_accumulation_hours": v["accum"]},
        # area None here so filter.area is applied to the shared 20S-90N extraction
        "extract_points": {"output_path": f"./extracted_points/{extract_dir}", "save_format": "pandas",
                           "area": None, "interpolation_method": "nearest"},
        "auxiliary_fields": {
            "model1": {"lsm_path": "/ec/vol/destine/continuous_evaluation/aifs_lsm.grib",
                       "orog_path": "/ec/vol/destine/continuous_evaluation/aifs_orog.grib"},
            "model2": {"lsm_path": "/ec/vol/destine/continuous_evaluation/aifs_lsm.grib",
                       "orog_path": "/ec/vol/destine/continuous_evaluation/aifs_orog.grib"},
            "sdfor_path": "/ec/vol/destine/continuous_evaluation/sdfor_tco1279.grib"},
        "filter": flt,
        "threshold": {"method": "local_obs_climatology", "event_type": event,
                      "local_obs_climatology": {"path": "./obs_clim_local", "parameter": pname,
                                                "percentile": pct, "window_days": 1, "n_years": 20,
                                                "first_year": 2005, "last_year": 2024,
                                                "min_availability_pct": 65}},
        "scores": {"ensemble": v["scores"], "stratify_by": ["lead_time"]},
        "bootstrap": {"enabled": True, "n_samples": 200, "confidence_level": 0.95},
        "save": {"output_directory": f"./results/{tag(var, model, region)}",
                 "save_scores_csv": True, "save_forecast_obs_pairs": False},
        "plot": {"heatmap_style": "smooth", "enabled": True, "dpi": 300, "format": "png"},
    }


def region_area(region):
    return {"nhextrop": "nh_extratropics", "tropics": "tropics"}[region]


HEADER = """\
# ============================================================================
# ENSEMBLE (quaver_extract) — {var} {thr}, {model} vs Control, {region_desc}
# 8 perturbed members (no control member), init 00 UTC, FMA + MJJA 2026 pooled
# ============================================================================
#   model 1 "control": rd/enfo/pf  FMA jccs | MJJA jcct
#   model 2 "{model}": rd/enfo/pf  FMA {fma} | MJJA {mjja}
#   Dates: 2026-02-02..04-28 and 2026-05-13..08-23 (common availability).
#   Extraction is shared with the other region (20S-90N, 4 chunks under
#   configs/_chunks/{chunk_owner}); this config only scores, via filter.area.
#   Generated by scripts/make_ocean_ens_configs.py — edit that, not this file.
# ============================================================================

"""


def write_yaml(path, cfg, header=""):
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))


def main():
    commands = []
    for model, exps in MODELS.items():
        for ex_var, members in EXTRACTION_GROUPS.items():
            extract_dir = f"{ex_var}_ens_quaver_{model}vsctrl_FebAug2026_nhtrop"
            owner = f"config_{tag(ex_var, model, 'nhextrop')}"  # watcher base config
            written = {}
            for var in members:
                var_dir = extract_dir if var == ex_var else extract_dir.replace(ex_var, var, 1)
                for region in ("nhextrop", "tropics"):
                    cfg = base_config(var, model, region, var_dir)
                    name = f"config_{tag(var, model, region)}"
                    hdr = HEADER.format(var=var, thr=VARS[var]["thr"], model=model,
                                        region_desc={"nhextrop": "NH extratropics, flat/hilly/complex",
                                                     "tropics": "Tropics, all stations"}[region],
                                        fma=exps["FMA"], mjja=exps["MJJA"], chunk_owner=owner)
                    write_yaml(ENS / f"{name}.yaml", cfg, hdr)
                    written[(var, region)] = name

                if var != ex_var:  # symlink day files of the shared 2t extraction
                    d = REPO / "extracted_points" / var_dir
                    d.mkdir(parents=True, exist_ok=True)
                    for day in range(1, 11):
                        link = d / f"{var}_control_vs_{model}_ens_day{day}.parquet"
                        target = Path("..") / extract_dir / f"{ex_var}_control_vs_{model}_ens_day{day}.parquet"
                        if not link.is_symlink():
                            link.symlink_to(target)

            # extraction chunks, owned by the NH config of the extracting variable
            nh = base_config(ex_var, model, "nhextrop", extract_dir)
            cdir = CHUNKS / owner
            cdir.mkdir(parents=True, exist_ok=True)
            for i, (period, lo, hi) in enumerate(CHUNK_RANGES):
                ch = copy.deepcopy(nh)
                ch["start_date"], ch["end_date"] = lo, hi
                ch["extraction_only"] = True
                ch["extract_points"]["area"] = "nh_extratropics_tropics"
                ch["extract_points"]["output_path"] = f"./extracted_points/{extract_dir}_chunk{i}"
                ch["read_data"]["forecast_model1"]["quaver"]["expver"] = CONTROL[period]
                ch["read_data"]["forecast_model2"]["quaver"]["expver"] = exps[period]
                write_yaml(cdir / f"{owner}_chunk{i}.yaml", ch)

            # post-processing: combined NH + tropics scorecards for each variable
            lines = ["#!/bin/bash", "#SBATCH --job-name=combined_plot",
                     "#SBATCH --output=combined_plot_%j.out", "#SBATCH --error=combined_plot_%j.err",
                     "#SBATCH --time=01:00:00", "#SBATCH --mem=8G",
                     "#SBATCH --chdir=/home/moeg/scorecards4extremes",
                     'source "/home/moeg/scorecards4extremes/.venv/bin/activate"', "export MPLBACKEND=Agg"]
            for var in members:
                nh_c, tr_c = written[(var, "nhextrop")], written[(var, "tropics")]
                out = f"results/{tag(var, model, 'nhextrop_tropics')}"
                lines.append(f"python -u scripts/plot_combined_regions.py configs/ensemble/{nh_c}.yaml "
                             f"configs/ensemble/{tr_c}.yaml {out}")
            post = cdir / "post_combined_plot.sh"
            post.write_text("\n".join(lines) + "\n")

            extras = [f"configs/ensemble/{written[(ex_var, 'tropics')]}.yaml"]
            for var in members[1:]:
                extras += [f"configs/ensemble/{written[(var, r)]}.yaml" for r in ("nhextrop", "tropics")]
            commands.append((owner, extras, str(post.relative_to(REPO))))

    for owner, extras, post in commands:
        print(owner, "|", " ".join(extras), "|", post)


if __name__ == "__main__":
    main()
