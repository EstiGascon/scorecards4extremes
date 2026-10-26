# Plan: Run the scorecards4extremes pipeline on the AirNow air-quality data (any species)

> Status: **deferred** — investigation done 2026-07-31, implementation to happen later.
> Companion to `docs/AOD550_PORT_PLAN.md` (same "score pre-aligned point-parquet" mechanism).

## Context

Score the CAMS-vs-AirNow sample data in `cams_sample_data/` with the rest of the
scorecards4extremes pipeline (`run.py` + steps) — for **any** of the AirNow species,
not just PM2.5. The ingestion script `cams_sample_data/read_cams_data_airnow.py`
supports **seven species** via its `PARAMS` registry (`read_cams_data_airnow.py:77-85`):
`pm2.5, pm10, o3, no2, so2, co, no`. PM2.5 is just the one already fully prepared.

The mechanism is identical for every species, and pm2p5 already proves it works:
the ingestion script emits pipeline-format parquet in exactly the layout `filter.py`
globs (`filter.py:30` → `{variable}_{fc1_name}_vs_{fc2_name}_day*.parquet`). The
scoring path (Steps 5–9) is variable-agnostic — it reads generic
`obs_value`/`fc1_value`/`fc2_value` columns. So switching species is **config +
(for species not yet built) one data-prep command**, with only cosmetic code edits.

**Intended outcome:** a reusable recipe producing an extremes scorecard
(oper vs icki over AirNow) — scores CSV + heatmap — for any of the seven species.

## The one subtlety: pipeline `variable:` = the filename token, not the CLI name

The script names files with `var_token(spec)` = `model_var` with `_ml137` stripped
(`read_cams_data_airnow.py:88-90`). The pipeline's `variable:` **must equal that
token** (it drives the glob). These differ from the human names — note especially
**o3 → `go3`**:

| Script `--param` | Pipeline `variable:` (token) | Obs var | Units | event_type | Day-parquet built? |
|---|---|---|---|---|---|
| `pm2.5` | `pm2p5` | PM2.5 | µg/m³ | above | ✅ ready (`aligned_airnow_pm2p5_oper_icki/`) |
| `o3`    | `go3`   | OZONE | ppb   | above | ⚠️ CSVs only (`aligned_airnow_o3_oper_icki/`) — must build parquet |
| `pm10`  | `pm10`  | PM10  | µg/m³ | above | ❌ must build |
| `no2`   | `no2`   | NO2   | ppb   | above | ❌ must build |
| `so2`   | `so2`   | SO2   | ppb   | above | ❌ must build |
| `co`    | `co`    | CO    | ppm   | above | ❌ must build |
| `no`    | `no`    | NO    | ppb   | above | ❌ must build |

Forecast tokens are `oper_airnow` / `icki_airnow` (`tag = "{expid1}_vs_{expid2}"`,
`read_cams_data_airnow.py:399,569`), so a full filename is e.g.
`go3_oper_airnow_vs_icki_airnow_day1.parquet`. Model values are converted to obs
units in-script (`to_obs_units`, `:93-97`), so fc/obs are directly comparable.

## Step A — Build day-parquet for the chosen species (skip for pm2p5)

Only pm2p5 is pre-built. For any other species, run the ingestion script in
**range mode** first — it writes `{tok}_oper_airnow_vs_icki_airnow_day{1..5}.parquet`
into `--out-dir` (`read_cams_data_airnow.py:459-468`):

```bash
python cams_sample_data/read_cams_data_airnow.py \
  --param o3 \                                        # pm2.5 | pm10 | o3 | no2 | so2 | co | no
  --start-date 20250101 --end-date 20251231 \
  --expid1 oper_airnow --expid2 icki_airnow \
  --out-dir cams_sample_data/aligned_airnow_o3_oper_icki \
  --no-daily-csv                                       # parquet only; drop for CSVs too
```

Notes:
- Prerequisite: the model netCDFs in `cams_sample_data/oper_airnow/` and
  `icki_airnow/` must contain that species' `model_var` (e.g. `go3_ml137`,
  `no2_ml137`, …). pm2p5 and go3 are confirmed present (their aligned data exists);
  verify the others don't error with "missing variable" on a single date first
  (drop `--start-date/--end-date` to run one `--base-date`).
- Missing model/obs dates are skipped, not fatal (`:445-450`).
- Resulting schema matches the pipeline exactly:
  `date, step, forecast_day, valid_time, station_id, lat, lon, obs_value, fc1_value, fc2_value, n_obs`
  (verified on pm2p5: 0 % NaN in both fc columns, ~661 stations, steps 3-hourly
  3–120 h, `forecast_day = ceil(step/24)` → days 1–5, dates 20250101–20251231).

## Step B — Code edits (cosmetic; nice labels)

`pm2p5`/`go3`/etc. are all safe generic variables end-to-end (confirmed by code
trace): no variable-specific QC fires (`filter.py:280,433,441,456` only match
`2t`/`tp24`/`10ff`), `compute_steps`/`format_threshold_string` fall through
generically (`utils.py:42,75`), `preprocess.run_step2` is a no-op
(`preprocess.py:27-44`), and `plot.py` uses `.get(variable, default)` so a missing
entry never crashes. **No code edits are required to run.**

For readable scorecard labels, add entries for **all seven tokens** to the inline
maps in `plot.py` (add to every occurrence):
- `units` dicts (`plot.py:127, 313, 434, 647, 696, 940`):
  `'pm2p5':'µg/m³', 'pm10':'µg/m³', 'go3':'ppb', 'no2':'ppb', 'so2':'ppb', 'no':'ppb', 'co':'ppm'`
- `var_display_map` dicts (`plot.py:293, 641, 936, 1216`):
  `'pm2p5':'PM2.5', 'pm10':'PM10', 'go3':'O₃', 'no2':'NO₂', 'so2':'SO₂', 'no':'NO', 'co':'CO'`

## Step C — Configuration — create one `my_<token>.yaml` per species

Start from `config_example.yaml`. Only the **bold** fields change per species;
everything else is shared. Example shown for o3/`go3`:

```yaml
variable: "go3"                    # <-- the TOKEN from the table (pm2p5|pm10|go3|no2|so2|co|no)
start_date: "2025-01-01"           # short test window first; widen to 2025-12-31 for full run
end_date:   "2025-01-31"
mode: "deterministic"
forecast_days: [1, 2, 3, 4, 5]     # steps 3..120h => 5 days
lead_time_frequency: 3             # AirNow steps are 3-hourly; 3 keeps every step (6 = every other)

skip_extraction_if_exists: true    # <-- reads the existing day-parquet, skips Metview (run.py:726-756)
skip_scoring_if_exists: false

read_data:
  forecast_model1:
    name: "oper_airnow"            # MUST match token in <var>_oper_airnow_vs_icki_airnow_day*.parquet
    source: "local_grib"
    local_grib: { path: "./cams_sample_data/aligned_airnow_o3_oper_icki" }   # existence checked; contents unused
  forecast_model2:
    name: "icki_airnow"            # MUST match the second token
    source: "local_grib"
    local_grib: { path: "./cams_sample_data/aligned_airnow_o3_oper_icki" }
  observation_source: "local_gpt"
  local_gpt: { path: "./cams_sample_data/aligned_airnow_o3_oper_icki" }      # existence checked; contents unused

preprocess:
  wind_speed_from_components: false
  lapse_rate_correction: false     # MUST be off: air-quality species have no elevation lapse
  precipitation_accumulation_hours: null

extract_points:
  output_path: "./cams_sample_data/aligned_airnow_o3_oper_icki"   # the dir holding the day*.parquet
  save_format: "pandas"

filter:
  lead_times: null                 # null = all steps
  season: null                     # or ["DJF","MAM","JJA","SON"]
  # orography_type: OMIT            # no sdfor column -> no terrain stratification
  remove_coastal_stations: false   # no lsm column; keep false so nothing is dropped
  remove_outliers: false           # MUST be false: pollutants are right-skewed; 5σ cut strips real extremes

threshold:                         # OPEN CHOICE (left configurable):
  method: "dataset_climatology"
  event_type: "above"              # air-quality extremes = high concentration (same for all 7 species)
  dataset_climatology:
    percentile: 95                 # p95 = more events / tighter CIs; p99 = rarer, more "extreme"

scores:
  deterministic: ["ETS","PSS","POD","FAR","twMAE","twRMSE","bias","mae","rmse","correlation"]
  stratify_by: ["lead_time"]

bootstrap: { enabled: true, n_samples: 200, confidence_level: 0.95 }   # 1000 for publication runs
save: { output_directory: "./results/go3_airnow", save_scores_csv: true, save_forecast_obs_pairs: false }
plot: { enabled: true, create_heatmap: true, heatmap_style: "smooth", dpi: 150, format: "png" }
```

Per-species diffs vs the template above: `variable`, the three `path`/`output_path`
dirs (point at that species' `aligned_airnow_<param>_oper_icki` dir), and
`save.output_directory`. Everything else is identical.

### Threshold — left open (per species)
- **`dataset_climatology` p95** (recommended first run) — 95th pct of in-dataset obs, `event_type: above`. More events → tighter bootstrap CIs. No external files. (pm2p5 reference: p95 ≈ 19, p99 ≈ 50 µg/m³.)
- **`dataset_climatology` p99** — rarer, more genuinely extreme; fewer samples per station.
- **`fixed`** (e.g. a WHO/EPA-style µg/m³ or ppb cutoff) — physically interpretable, but `format_threshold_string` gives an *unlabeled* numeric suffix for these tokens (`utils.py:75-76`).
- **Avoid** `local_obs_climatology` / `station_climatology` — no AirNow climatology files exist; those paths are met-only.

## Gotchas

1. **`variable:` is the token, not the CLI name** — especially **o3 → `go3`**. A mismatch → "No data file found" (`filter.py:30`).
2. **Dummy-but-existing paths for Step 1.** `read_data` raises `FileNotFoundError` if `local_grib.path`/`local_gpt.path` don't exist even though extraction is skipped (`read_data.py:25-26, 87-88`). Point all three at the species' parquet directory.
3. **`name:` tokens must match filenames exactly** — `oper_airnow` / `icki_airnow` (not bare `oper`/`icki`).
4. **`output_path` must be the directory that holds the day-parquet** so `skip_extraction_if_exists` finds them (`run.py:739-744`).
5. **Lapse-rate off / preprocess no-op / no terrain stratification.** Parquet has no `sdfor`/`lsm`; omit `orography_type` and set `remove_coastal_stations: false`. Both filters self-skip but this keeps the log clean.
6. **Don't enable `remove_outliers`.** All these pollutants are right-skewed; the 5σ cut (`filter.py:420`) would delete the extremes you want to score.
7. **Steps/forecast_days must intersect the parquet's steps** (3-hourly, 3–120 h → days 1–5). `lead_time_frequency: 3` keeps every step.
8. **Units differ across species** (µg/m³ vs ppb vs ppm) — the Step B plot labels encode this; don't hard-code one unit.
9. **A species other than pm2p5/go3 may lack its `model_var` in the netCDFs.** Test one base date before a full range build (Step A note).

## Verification (end-to-end, per species — example uses go3)

1. Confirm day-parquet exists and schema is right (after Step A):
   `python -c "import pandas as pd; d=pd.read_parquet('cams_sample_data/aligned_airnow_o3_oper_icki/go3_oper_airnow_vs_icki_airnow_day1.parquet'); print(d.columns.tolist()); print(d[['fc1_value','fc2_value','obs_value']].describe())"`
2. Local dry run: `python run.py my_go3.yaml` — watch the log:
   - Step 1 prints `Name: oper_airnow` / `Name: icki_airnow` and passes path checks.
   - Step 3 prints skipped / "extracted data already exists".
   - Step 4 prints found N forecast-day files with non-zero rows.
3. Confirm Steps 5–8 write a scores CSV in `results/go3_airnow/`, and Step 9 writes a heatmap PNG labelled "O₃ (ppb)" (from the Step B edits).
4. Widen `start_date`/`end_date` to full 2025, then batch: `sbatch submit_job.sh my_go3.yaml`.
5. Repeat A–C for each additional species (`pm10`, `no2`, `so2`, `co`, `no`) — only `variable`, the dirs, `save.output_directory`, and (optionally) the threshold change.
