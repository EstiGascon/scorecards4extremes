# Scorecards4Extremes — Quick Start

Get your first scorecard running in 5 steps.

---

## Prerequisites

- Python ≥ 3.10 with the project virtual environment set up (see [INSTALL.md](../INSTALL.md))
- GRIB forecast files for two models on local disk
- `.gpt` (Geopoints) observation files on local disk (see [§10 of the User Guide](USER_GUIDE.md#10-observation-data-format-gpt--geopoints) for format details)
- **Ensemble runs** additionally require ≥ 64 GB RAM; use a batch/HPC system if your laptop has less

---

## Step 1 — Pick a config template

Choose the closest template for your use case:

| Template | Variable | Threshold | Mode |
|----------|----------|-----------|------|
| `configs/deterministic/config_2t_local_p99obsclim.yaml` | 2 m temperature | p99 warm | deterministic |
| `configs/deterministic/config_2t_local_p1obsclim.yaml` | 2 m temperature | p1 cold | deterministic |
| `configs/deterministic/config_tp24_local_p99obsclim.yaml` | 24 h precipitation | p99 heavy | deterministic |
| `configs/deterministic/config_10ff_local_p99obsclim.yaml` | 10 m wind speed | p99 strong | deterministic |
| `configs/ensemble/config_2t_ens_local_p99obsclim.yaml` | 2 m temperature | p99 warm | ensemble |
| `configs/ensemble/config_tp24_ens_local_p99obsclim.yaml` | 24 h precipitation | p99 heavy | ensemble |

For a clean external-user starting point with no ECMWF dependencies, use:

```bash
cp config_example.yaml my_run.yaml
```

Or copy an existing experiment template:

```bash
cp configs/deterministic/config_tp24_local_p99obsclim.yaml my_run.yaml
```

---

## Step 2 — Edit 5 key fields

Open `my_run.yaml` and change only these:

```yaml
start_date: "2024-01-01"          # ← your verification period start
end_date:   "2024-12-31"          # ← your verification period end

read_data:
  forecast_model1:
    name: "my_model"               # ← short label (used in plot titles)
    local_grib:
      path: "/data/forecasts/model1" # ← path to model 1 GRIB files

  forecast_model2:
    name: "reference"
    local_grib:
      path: "/data/forecasts/model2" # ← path to model 2 GRIB files

  local_gpt:
    path: "/data/observations"       # ← path to .gpt observation files

extract_points:
  output_path: "./extracted_points/my_run"  # ← where to cache data

save:
  output_directory: "./results/my_run"      # ← where to save results
```

Everything else can stay at the template defaults for a first run.

---

## Step 3 — Run (interactive, small test)

For a quick sanity-check over a short date range:

```bash
cd /path/to/scorecards4extremes
source .venv/bin/activate
python run.py my_run.yaml
```

This runs all 9 pipeline steps sequentially in the current terminal.

---

## Step 4 — Run (batch, full period)

For the full verification period, submit as a SLURM job:

```bash
sbatch submit_job.sh my_run.yaml
```

Monitor progress:

```bash
squeue -u $USER
tail -f scorecards_<JOBID>.out
```

---

## Step 5 — View results

Output files appear in `save.output_directory`:

```
./results/my_run/
├── panel_heatmap_DJF_flat.png     ← 4-panel scorecard (flat terrain, winter)
├── panel_heatmap_DJF_complex.png  ← 4-panel scorecard (complex terrain, winter)
├── panel_heatmap_JJA_flat.png
├── ...
└── scores_DJF_flat.csv            ← raw score numbers
```

Green cells = model 1 better than model 2. Red cells = model 2 better.
Dotted cells = statistically significant at 95%.

---

## Common next steps

### Re-run only the plots (no re-extraction or re-scoring)

Set both skip flags in the config and rerun:

```yaml
skip_extraction_if_exists: true
skip_scoring_if_exists:    true
```

Then change the `plot:` section and run again:

```bash
.venv/bin/python run.py my_run.yaml
```

### Run diagnostic Q-Q plots

```bash
.venv/bin/python diagnostics/plot_qq_extremes.py \
  --config my_run.yaml \
  --day 3 --season DJF --orog complex
```

### Run detailed diagnostics (11-plot set)

```bash
sbatch submit_diagnose.sh my_run.yaml --day 3 --season DJF --orog complex
```

### Switch to ensemble mode

Change these fields in the config:

```yaml
mode: "ensemble"
steps: [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

scores:
  ensemble:
    - "twCRPS"
    - "fCRPS"
    - "Brier"
    - "BSS"
    - "tw_quantile_score"
    - "extreme_spread_skill_ratio"
```

For a full ensemble run with 250 dates × 50 members, expect **multiple 48h HPC jobs** due to wall-time limits (see `docs/COMPUTING.md`).

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Job fails immediately | Missing GRIB or GPT files | Check paths in `read_data` section |
| `No stations found` | Season/orography filter too strict | Try `season: null` and `orography_type: null` first |
| Heatmap all grey (NaN) | Too few events at threshold | Lower the percentile or use `dataset_climatology` |
| Memory error in interactive | > 8 GB needed | Use `sbatch submit_job.sh` instead |
| Job killed before finishing | Ensemble needs > 48h | See `docs/COMPUTING.md` for restart strategy |
