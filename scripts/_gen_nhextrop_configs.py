"""Generate NH-extratropics twins of the Europe full-year configs.

Each twin = exact clone of the Europe source with:
  - extract_points.area          -> "nh_extratropics"
  - extract_points.output_path   -> ./extracted_points/<runname>   (unique per run)
  - save.output_directory        -> ./results/<runname>            (unique per run)
  - (10ff p99 ensemble only) threshold percentile 98 -> 99

Unique per-run extraction dirs guarantee no shared-dir races and no cross-job
dependencies, so every run is fully self-contained for unattended execution.
"""
import copy
import yaml

CFG = "configs"

# (source_config, target_runname, override_percentile_or_None)
JOBS = [
    # ---- deterministic ----
    (f"{CFG}/deterministic/config_mn2t_method3_quaverextract_p1obsclim_ifsvsaifs_fullyear.yaml",
     "mn2t_method3_quaverextract_p1obsclim_ifsvsaifs_fullyear_nhextrop", None),
    (f"{CFG}/deterministic/config_tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear.yaml",
     "tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_nhextrop", None),
    (f"{CFG}/deterministic/config_tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_regrid025.yaml",
     "tp24_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_regrid025_nhextrop", None),
    (f"{CFG}/deterministic/config_10ff_method3_quaverextract_p98obsclim_ifsvsaifs_fullyear_allseasons_withcoastal.yaml",
     "10ff_method3_quaverextract_p98obsclim_ifsvsaifs_fullyear_allseasons_withcoastal_nhextrop", None),
    (f"{CFG}/deterministic/config_10ff_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_allseasons_withcoastal.yaml",
     "10ff_method3_quaverextract_p99obsclim_ifsvsaifs_fullyear_allseasons_withcoastal_nhextrop", None),
    # ---- ensemble ----
    (f"{CFG}/ensemble/config_mn2t_ens_quaver_p1obsclim_aifsvsifs_fullyear.yaml",
     "mn2t_ens_quaver_p1obsclim_aifsvsifs_fullyear_nhextrop", None),
    (f"{CFG}/ensemble/config_tp24_ens_local_p99obsclim_aifsvsifs_native_fullyear.yaml",
     "tp24_ens_local_p99obsclim_aifsvsifs_native_fullyear_nhextrop", None),
    (f"{CFG}/ensemble/config_tp24_ens_local_p99obsclim_aifsvsifs_fullyear.yaml",
     "tp24_ens_local_p99obsclim_aifsvsifs_fullyear_nhextrop", None),
    (f"{CFG}/ensemble/config_10ff_ens_quaver_p98obsclim_aifsvsifs_fullyear_allseasons_withcoastal.yaml",
     "10ff_ens_quaver_p98obsclim_aifsvsifs_fullyear_allseasons_withcoastal_nhextrop", None),
    # 10ff p99 ensemble: derived from the p98 ensemble source, percentile bumped to 99
    (f"{CFG}/ensemble/config_10ff_ens_quaver_p98obsclim_aifsvsifs_fullyear_allseasons_withcoastal.yaml",
     "10ff_ens_quaver_p99obsclim_aifsvsifs_fullyear_allseasons_withcoastal_nhextrop", 99),
]


def target_path(src, runname):
    sub = "deterministic" if "/deterministic/" in src else "ensemble"
    return f"{CFG}/{sub}/config_{runname}.yaml"


for src, runname, pct in JOBS:
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    cfg["extract_points"]["area"] = "nh_extratropics"
    cfg["extract_points"]["output_path"] = f"./extracted_points/{runname}"
    cfg["save"]["output_directory"] = f"./results/{runname}"

    if pct is not None:
        cfg["threshold"]["local_obs_climatology"]["percentile"] = pct

    out = target_path(src, runname)
    with open(out, "w") as f:
        f.write(f"# NH-extratropics twin of {src.split('/')[-1]}\n")
        f.write("# area=nh_extratropics; unique extraction/results dirs; else identical.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    p = cfg["threshold"]["local_obs_climatology"].get("percentile", "?")
    print(f"WROTE {out}")
    print(f"      area={cfg['extract_points']['area']} pct={p} "
          f"extract={cfg['extract_points']['output_path']} results={cfg['save']['output_directory']}")
