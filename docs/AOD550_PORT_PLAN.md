# Plan: Run the scorecards4extremes pipeline on aod550 (using pre-aligned parquet)

> Status: **deferred** — investigation done 2026-07-31, implementation to happen later.

## Context

You already have aligned **aod550** point-parquet (the same format `cams_sample_data/read_cams_data_aod_aeronet.py` produces for aod500). You want to score it with the *rest* of the scorecards4extremes pipeline (`run.py` and its steps), **without** touching `cams_sample_data`.

Key fact established by exploration: **AOD is not a native pipeline variable.** The pipeline only "knows" `2t`, `tp24`, `10ff` as first-class variables (with GRIB/MARS retrieval, unit conversion, QC). But it can score *any* variable that arrives as pre-aligned point-parquet, because the scoring path treats the data as generic `fc1_value` / `fc2_value` / `obs_value` columns. AOD500 already works this way today. So switching to aod550 is **almost entirely configuration** — the code changes are minimal/cosmetic.

## How AOD flows through the pipeline (the mechanism)

- Step 4 loader `filter.py:30` globs `{variable}_{fc1_name}_vs_{fc2_name}_day*.parquet` in `extract_points.output_path`. Your files must be named e.g. `aod550_oper_vs_icki_day1.parquet … _day5.parquet`.
- Step 3 (Metview GRIB extraction) is **skipped** when `skip_extraction_if_exists: true` and the day files already exist (`run.py:726-762`).
- Step 1 (`read_data.run_step1`) **still runs** and *validates that the configured forecast/obs paths exist* (`read_data.py:25-26`, `read_data.py:87-88`). It also sets `fc1_name`/`fc2_name` from the `name:` config keys (`read_data.py:17`). These names must equal the tokens in your parquet filenames (`oper`, `icki`).
- Steps 5–9 (threshold, scores, bootstrap, save, plot) operate on the generic value columns and are variable-agnostic.

## Required parquet layout (what you must have)

Place your files in the directory you'll set as `extract_points.output_path`, named:
`aod550_<fc1name>_vs_<fc2name>_day{1..5}.parquet`
Each file needs the pipeline schema columns (same as the aod500 `_pipeline.parquet`):
`date, step, forecast_day, valid_time, station_id, lat, lon, obs_value, fc1_value, fc2_value, n_obs`.
For a real two-model scorecard, **both `fc1_value` and `fc2_value` must be populated** (single-experiment leaves `fc2_value` NaN, which is not a meaningful comparison).

## Code edits

**None are strictly required to run.** Verified:
- `plot.py` uses `units.get(variable, '')` and `var_display_map.get(variable, variable)` (`plot.py:128,294`) — a missing aod550 entry does **not** crash; it just yields a blank unit and the display label "aod550".
- `filter.py` variable-specific QC only fires for `2t`/`tp24`/`10ff` (`filter.py:433-468`); for aod550 those blocks are inert.
- `utils.compute_steps` only special-cases `tp24`/`tp` (`utils.py:42`); aod550 gets generic step handling.

**Optional (cosmetic, recommended for readable scorecards)** — add an aod550 entry to the display/unit maps in `plot.py` (there are ~5 repeated `units = {...}` dicts and 4 `var_display_map = {...}` dicts, e.g. lines 293, 641, 936, 1216):
- `var_display_map`: `'aod550': 'AOD 550 nm'`
- `units`: `'aod550': ''` (AOD is dimensionless) or `'aod550': 'AOD'`

## Configuration (the actual work) — create `my_aod550.yaml`

Start from `config_example.yaml` and set:

```yaml
variable: "aod550"                 # drives the filename glob
start_date / end_date              # within your parquet's date range
mode: "deterministic"
forecast_days: [1, 2, 3, 4, 5]     # steps 3..120h => 5 days
lead_time_frequency: 3             # AOD steps are 3-hourly (3,6,...,120); use 3 to keep all

skip_extraction_if_exists: true    # <-- makes the pipeline read your parquet, skip Metview

read_data:
  forecast_model1:
    name: "oper"                   # MUST match the token in aod550_oper_vs_icki_day*.parquet
    source: "local_grib"
    local_grib: { path: "<an existing dir>" }   # existence is checked; contents unused (extraction skipped)
  forecast_model2:
    name: "icki"                   # MUST match the second token
    source: "local_grib"
    local_grib: { path: "<an existing dir>" }
  observation_source: "local_gpt"
  local_gpt: { path: "<an existing dir>" }       # existence checked; contents unused

extract_points:
  output_path: "<dir containing your aod550_oper_vs_icki_day*.parquet>"
  save_format: "pandas"

preprocess:
  wind_speed_from_components: false
  lapse_rate_correction: false     # <-- MUST be off: AOD has no elevation lapse; also no orog columns

filter:
  lead_times: [...]                # subset of your steps, or omit
  season: "all"                    # or DJF/MAM/JJA/SON
  orography_type: "all"            # NO terrain stratification (parquet has no 'sdfor' column)
  remove_coastal_stations: false   # parquet has no 'lsm' column (self-skips, but set false)
  remove_outliers: false           # keep false: AOD is right-skewed; ±5σ would strip real extremes

threshold:
  method: "dataset_climatology"    # recommended for AOD (see below)
  event_type: "above"              # AOD extremes = high aerosol loading
  dataset_climatology: { percentile: 95 }   # e.g. p90/p95/p98

scores:
  deterministic: [ ... ]           # pick continuous/threshold scores; avoid unit-specific assumptions
bootstrap: { enabled: true, n_samples: 1000, confidence_level: 0.95 }
save: { output_directory: "results/aod550", save_scores_csv: true }
plot: { enabled: true, create_heatmap: true }
```

### Threshold choice (undecided as of investigation)
- **`dataset_climatology` (recommended)** — percentile of the in-dataset AERONET obs (e.g. p95 for high-AOD extremes) with `event_type: above`. No extra files needed; works out of the box.
- **`fixed`** — a constant AOD cutoff (e.g. `value: 0.4`). Simple; `format_threshold_string` falls through to a generic `f"{value:.1f}"` for aod550 (`utils.py:75-76`).
- **Avoid** `local_obs_climatology` / `station_climatology` — no AOD climatology files exist in `obs_clim_local/`, and `station_climatology` is ECMWF/STVL met-only.

## Be aware of (gotchas)

1. **Dummy-but-existing paths for Step 1.** `read_data` raises `FileNotFoundError` if `local_grib.path`/`local_gpt.path` don't exist, even though extraction is skipped. Point them at any existing directory (the parquet dir is fine).
2. **`name:` tokens must match filenames exactly.** `variable` + `fc1 name` + `fc2 name` reconstruct the glob `aod550_oper_vs_icki_day*.parquet`. A mismatch → "No data file found".
3. **No terrain/coastal stratification.** AOD parquet lacks `sdfor`/`lsm`; orography and coastal filters silently skip. Keep `orography_type: all`.
4. **Lapse-rate off.** Elevation correction is meteorological; leave `lapse_rate_correction: false`.
5. **Don't enable `remove_outliers`.** AOD, like precipitation, is right-skewed; ±5σ removal would delete the extreme events you want to score.
6. **Steps/forecast_days must intersect your parquet's steps** (3-hourly, 3–120h → days 1–5). `lead_time_frequency: 3` keeps every step; `6` would use every other step.
7. **Units are blank by default.** Scorecards will show no unit for aod550 unless you add the optional `plot.py` map entries.
8. **AERONET has no native 550 nm channel** — this is a data-prep concern inside `cams_sample_data` (out of scope here), but be sure your obs `obs_value` is a valid 550 nm quantity (typically interpolated from 500/675 nm), not a raw 500 nm value mislabeled.

## Verification (end-to-end)

1. Sanity-check one parquet: confirm columns and naming
   `python -c "import pandas as pd; d=pd.read_parquet('<out>/aod550_oper_vs_icki_day1.parquet'); print(d.columns.tolist()); print(d[['fc1_value','fc2_value','obs_value']].describe())"`
2. Dry run locally: `python run.py my_aod550.yaml` — watch the log:
   - Step 1 prints `Name: oper` / `Name: icki` and passes path checks.
   - Step 3 prints `✓ Skipped - extracted data already exists`.
   - Step 4 prints `Found N forecast day files (memory-efficient mode)` with non-zero rows.
3. Confirm Steps 5–8 produce a scores CSV in `results/aod550/`, and Step 9 writes a heatmap.
4. Batch: `sbatch submit_job.sh my_aod550.yaml` once the local dry run is clean.
