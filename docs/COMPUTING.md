# Scorecards4Extremes — Computational Requirements

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [SLURM Job Configuration](#2-slurm-job-configuration)
3. [Runtime Estimates](#3-runtime-estimates)
4. [Storage Requirements](#4-storage-requirements)
5. [Incremental Restart Strategy](#5-incremental-restart-strategy)
6. [Parallelism and Worker Limits](#6-parallelism-and-worker-limits)
7. [Bootstrap Cost](#7-bootstrap-cost)
8. [ECMWF HPC Environment](#8-ecmwf-hpc-environment)

---

## 1. Hardware Requirements

### Minimum (interactive testing, short date ranges)

| Resource | Minimum |
|----------|---------|
| RAM | 8 GB |
| CPUs | 1 |
| Storage | 2 GB |

Interactive sessions on ECMWF HPC are limited to 8 GB. This is sufficient for:
- Deterministic runs over ≤ 2 months
- Diagnostics (`diagnostics/diagnose_extremes.py`, `diagnostics/plot_qq_extremes.py`)

### Recommended (full verification period, deterministic)

| Resource | Recommended |
|----------|-------------|
| RAM | 64 GB |
| CPUs | 4–8 |
| Walltime | 18–24 h |

### Recommended (full verification period, ensemble)

| Resource | Recommended |
|----------|-------------|
| RAM | 128 GB |
| CPUs | 12 |
| Walltime | 48 h (may require multiple runs) |

---

## 2. SLURM Job Configuration

The main submission script `submit_job.sh` currently requests:

```bash
#SBATCH --partition=nf      # ECMWF-specific partition name — change to your cluster's partition
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
```

> **External users**: replace `--partition=nf` with the appropriate partition
> name for your own HPC system (e.g. `--partition=compute`, `--partition=long`,
> etc.). The other resource requests are generic SLURM and do not need changing.

Adjust these as needed:

```bash
# Lighter deterministic run:
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00

# Heavy ensemble run (maximum config):
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
```

---

## 3. Runtime Estimates

### Extraction step (Step 3)

The extraction step is the dominant cost for ensemble runs. Timing is measured per date (one forecast initialisation date).

#### Deterministic extraction

| Variable | Steps | Time per date | 250 dates |
|----------|-------|--------------|-----------|
| 2t | daily (10 steps) | ~1–2 min | ~5–8 h |
| tp24 | daily (10 steps) | ~1–2 min | ~5–8 h |
| 10ff | daily (10 steps) | ~1–2 min | ~5–8 h |

Deterministic runs **complete comfortably within a single 18–24 h job**.

#### Ensemble extraction

Ensemble extraction is substantially slower due to the large number of member columns.

| Variable | Steps | Members | Time per date | 250 dates |
|----------|-------|---------|--------------|-----------|
| 2t | 6-hourly (40 steps) | 50 | ~25–30 min | ~100–125 h |
| tp24 | 6-hourly (40 steps) | 50 | ~25–30 min | ~100–125 h |
| 10ff | 6-hourly (40 steps) | 50 | ~25–30 min | ~100–125 h |
| 2t | daily (10 steps) | 50 | ~7 min | ~29 h |
| tp24 | daily (10 steps) | 50 | ~7 min | ~29 h |

**For 6-hourly ensemble runs, budget 3–4 SLURM jobs of 48 h each** (total ~6–9 days elapsed, thanks to the restart mechanism).

**For daily steps, one 48 h job is sufficient.**

### Scoring step (Step 6)

Scoring is fast relative to extraction:

| Mode | Score set | 250 dates | Notes |
|------|-----------|-----------|-------|
| Deterministic | All scores | ~10–30 min | |
| Ensemble | twCRPS + Brier + BSS + twQS | ~1–2 h | Per bootstrap sample |

### Bootstrap step (Step 7)

See [Section 7](#7-bootstrap-cost).

---

## 4. Storage Requirements

### Extracted points (parquet files)

Parquet files store float32 values. Approximate sizes:

| Mode | Members | Steps | Dates | Size per date | Total |
|------|---------|-------|-------|--------------|-------|
| Deterministic | 1 per model | 10 | 250 | ~5 MB | ~1.3 GB |
| Ensemble | 50 per model | 10 (daily) | 250 | ~20 MB | ~5 GB |
| Ensemble | 50 per model | 40 (6-hourly) | 250 | ~50 MB | ~12 GB |

`_tmp/` per-date files are removed once merged into per-day parquets.

### Results (CSV + PNG)

Small: typically < 100 MB per experiment.

### GRIB inputs

Read-only. Not created by the tool.

---

## 5. Incremental Restart Strategy

Long ensemble extraction jobs will be killed at the 48 h wall-time limit before completing all dates. The pipeline handles this gracefully through two mechanisms.

### Per-date `_tmp/` cache (extraction)

During ensemble extraction, each completed date is written to:

```
{extract_points.output_path}/_tmp/YYYY-MM-DD.parquet
```

On the next submission, already-present dates are detected and **skipped automatically** without re-reading GRIB files. The job resumes from where it left off.

No config changes are needed between submissions. Simply resubmit:

```bash
sbatch submit_job.sh my_run.yaml
```

### Score CSV skip (scoring)

If score CSVs already exist and `skip_scoring_if_exists: true`, the entire scoring step is bypassed. Use this to replot or adjust bootstrap without re-running all scoring:

```yaml
skip_extraction_if_exists: true
skip_scoring_if_exists:    true
```

### Recommended workflow for a large ensemble run

1. First submission: extraction starts from date 1.
2. Job killed at 48 h (e.g. after ~100/249 dates).
3. Second submission: extraction resumes from date 101.
4. ... repeat until all dates extracted.
5. Final submission: `skip_extraction_if_exists: true`; scoring and plotting complete.

---

## 6. Parallelism and Worker Limits

### CPUs and workers

The ensemble scoring step (`ens_scores.py`) uses Python's `ProcessPoolExecutor` with a **hard cap of 4 workers**, regardless of the number of CPUs available. This prevents memory exhaustion when many members are loaded simultaneously.

The extraction step (`extract_points_ensemble.py`) uses sequential per-date processing (no parallelism within the step), but multiple steps within a date are processed in parallel.

### Memory per worker

Each worker holds one date's worth of ensemble data in memory:
- 2 models × 50 members × 40 steps × N_stations (typically 10,000–30,000 for Europe)
- float32: ~4 bytes per value
- Typical: ~300 MB per worker at 40 steps, 50 members, 20,000 stations
- With 4 workers: ~1.2 GB for workers, plus overhead → 4–8 GB for scoring

The 128 GB allocation is driven primarily by the extraction step.

---

## 7. Bootstrap Cost

Bootstrap cost scales linearly with `n_samples`.

| n_samples | Deterministic (~10,000 stations) | Ensemble (~10,000 stations) |
|-----------|----------------------------------|----------------------------|
| 100 | ~2 min | ~15 min |
| 200 | ~4 min | ~30 min |
| 500 | ~10 min | ~75 min |
| 1000 | ~20 min | ~150 min |

### Recommendations

| Use case | Recommendation |
|----------|---------------|
| Quick sanity-check | `n_samples: 100` |
| Standard ensemble run | `n_samples: 200` |
| Deterministic run | `n_samples: 1000` |
| Publication / final results | `n_samples: 1000` |

---

## 8. ECMWF HPC Environment

### Required modules

The SLURM job script loads:

```bash
module load ecmwf-toolbox/new
```

This provides access to `eccodes`, `metview`, and related ECMWF libraries. Python uses the project virtual environment:

```bash
.venv/bin/python
```

Do **not** use the system `python3` — it lacks the required packages.

### TMPDIR configuration

For jobs that generate large temporary files (e.g. ensemble extraction), ensure `TMPDIR` points to a high-capacity scratch space:

```bash
export TMPDIR=/scratch/$(whoami)
```

The `submit_job.sh` script already sets this.

### Checking resource usage after a job

```bash
sacct -j <JOBID> --format=JobID,MaxRSS,ReqMem,Elapsed,State
```

Use this to tune memory and walltime for subsequent submissions.

### Useful commands

```bash
# Submit
sbatch submit_job.sh my_run.yaml

# Monitor queue
squeue -u $USER

# Live output
tail -f scorecards_<JOBID>.out

# Cancel
scancel <JOBID>

# Check completed job resources
sacct -j <JOBID> --format=JobID,MaxRSS,ReqMem,Elapsed,State
```
