# Batch Job Submission

> For the full submission reference see [USER_GUIDE.md § 13](USER_GUIDE.md#13-batch-submission-on-ecmwf-hpc) and [COMPUTING.md](COMPUTING.md).

## Quick start

```bash
sbatch scripts/submit_job.sh configs/deterministic/config_tp24_local_p99obsclim.yaml
```

Monitor progress:

```bash
squeue -u $USER
tail -f scorecards_*.out
```

## Current resource allocation (`scripts/submit_job.sh`)

| Resource | Value |
|----------|-------|
| Memory | 128 GB |
| CPUs | 12 |
| Walltime | 48 h |

For lightweight deterministic runs (≤ 3 months, single variable) 64 GB / 4 CPU / 18 h is sufficient. Edit the `#SBATCH` headers in `scripts/submit_job.sh` accordingly.

## Interactive memory limit

Interactive sessions on ECMWF HPC are limited to 8 GB. This is enough for:
- Running diagnostics (`diagnostics/diagnose_extremes.py`, `diagnostics/plot_qq_extremes.py`)
- Deterministic runs over ≤ 2 months

For full verification periods always use `sbatch`.

## Incremental restart

If a job is killed before completion:
- **Extraction**: per-date progress is cached in `_tmp/`. Resubmit and already-extracted dates are skipped automatically.
- **Scoring**: set `skip_scoring_if_exists: true` in the config to skip if score CSVs already exist.

## Checking memory usage after a job

```bash
sacct -j <JOB_ID> --format=JobID,MaxRSS,ReqMem,Elapsed
```
