#!/usr/bin/env python
"""Split a full-year extraction config into N date-range chunks for parallel
SLURM extraction, then (later) merge the results back together.

Rationale: quaver_extract.run_step3 processes dates strictly serially within
one job (per-date MARS/STVL retrieval). A single ~8-9 month job can exceed the
48h wall-clock limit. Splitting the date range into N independent chunks —
each writing to its own extraction directory — lets them run as N parallel
SLURM jobs, cutting wall-clock time by ~N (subject to instantaneous cluster
availability / MARS server rate limits).

Usage:
    # 1) Generate N chunk configs + submit them
    python scripts/chunk_extraction.py split <base_config.yaml> --n-chunks 4 --submit

    # 2) After all chunk jobs complete, merge the per-day parquet files into
    #    the base config's extract_points.output_path
    python scripts/chunk_extraction.py merge <base_config.yaml> --n-chunks 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNK_CONFIG_DIR = REPO_ROOT / "configs" / "_chunks"


def _load(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def _chunk_date_ranges(start_date, end_date, n_chunks):
    dates = pd.date_range(start_date, end_date, freq="24h")
    n = len(dates)
    chunk_size = -(-n // n_chunks)  # ceil division
    ranges = []
    for i in range(n_chunks):
        lo = i * chunk_size
        hi = min((i + 1) * chunk_size, n)
        if lo >= n:
            break
        ranges.append((dates[lo].strftime("%Y-%m-%d"), dates[hi - 1].strftime("%Y-%m-%d")))
    return ranges


def cmd_split(args):
    base_config_path = Path(args.config)
    base_name = base_config_path.stem
    config = _load(base_config_path)

    start_date = args.resume_from or config["start_date"]
    ranges = _chunk_date_ranges(start_date, config["end_date"], args.n_chunks)
    print(f"Chunking period {start_date} -> {config['end_date']} "
          f"(base config spans {config['start_date']} -> {config['end_date']}): "
          f"{len(ranges)} chunk(s)")

    base_output_path = config["extract_points"]["output_path"]

    chunk_dir = CHUNK_CONFIG_DIR / base_name
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_config_paths = []
    for i, (chunk_start, chunk_end) in enumerate(ranges):
        chunk_config = dict(config)
        chunk_config["start_date"] = chunk_start
        chunk_config["end_date"] = chunk_end
        chunk_config["extraction_only"] = True
        chunk_config["extract_points"] = dict(config["extract_points"])
        chunk_config["extract_points"]["output_path"] = f"{base_output_path}_chunk{i}"

        chunk_config_path = chunk_dir / f"{base_name}_chunk{i}.yaml"
        with open(chunk_config_path, "w") as f:
            yaml.safe_dump(chunk_config, f, sort_keys=False)
        chunk_config_paths.append(chunk_config_path)
        print(f"  chunk{i}: {chunk_start} -> {chunk_end}  ->  {chunk_config_path}")

    if args.submit:
        job_ids = []
        for chunk_config_path in chunk_config_paths:
            result = subprocess.run(
                ["sbatch", str(REPO_ROOT / "scripts" / "submit_job_quaver.sh"), str(chunk_config_path)],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            )
            print(result.stdout.strip())
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
        jobs_file = chunk_dir / "job_ids.txt"
        jobs_file.write_text("\n".join(job_ids) + "\n")
        print(f"\nSubmitted {len(job_ids)} job(s): {', '.join(job_ids)}")
        print(f"Job IDs saved to {jobs_file}")


def cmd_merge(args):
    base_config_path = Path(args.config)
    base_name = base_config_path.stem
    config = _load(base_config_path)

    variable = config["variable"]
    fc1_name = config["read_data"]["forecast_model1"]["name"]
    fc2_name = config["read_data"]["forecast_model2"]["name"]
    is_ensemble = config.get("mode") == "ensemble"
    filename_base = f"{variable}_{fc1_name}_vs_{fc2_name}" + ("_ens" if is_ensemble else "")

    base_output_path = Path(config["extract_points"]["output_path"])
    base_output_path.mkdir(parents=True, exist_ok=True)

    # Leftover per-date cache in the base dir itself, e.g. from an earlier
    # un-chunked run that was interrupted (killed/timed out) before it could
    # merge. Not deleted by this script — merge is additive/idempotent.
    base_tmp_dir = base_output_path / "_tmp"

    chunk_dirs = [Path(f"{config['extract_points']['output_path']}_chunk{i}")
                  for i in range(args.n_chunks)]
    chunk_dirs = [d for d in chunk_dirs if d.exists()]
    if not chunk_dirs and not base_tmp_dir.exists():
        print(f"No chunk directories or base _tmp cache found for {base_name} — nothing to merge.")
        return

    forecast_days = sorted({
        int(p.stem.split("_day")[-1])
        for d in chunk_dirs
        for p in d.glob(f"{filename_base}_day*.parquet")
    } | ({
        int(p.stem.split("_day")[-1])
        for p in base_tmp_dir.glob("*_day*.parquet")
    } if base_tmp_dir.exists() else set()))
    if not forecast_days:
        print("No per-day parquet files found in any chunk directory or base _tmp cache.")
        return

    for day in forecast_days:
        parts = [d / f"{filename_base}_day{day}.parquet" for d in chunk_dirs]
        parts = [p for p in parts if p.exists()]
        if base_tmp_dir.exists():
            parts += sorted(base_tmp_dir.glob(f"*_day{day}.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        final_file = base_output_path / f"{filename_base}_day{day}.parquet"
        df.to_parquet(final_file, compression="snappy", index=False)
        print(f"  merged day{day}: {len(parts)} source file(s) -> {len(df):,} rows -> {final_file}")

    print(f"\n✓ Merge complete: {len(forecast_days)} forecast-day file(s) written to {base_output_path}")
    print("  (chunk dirs and base _tmp cache left in place — remove manually once verified)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="Generate + optionally submit N chunk configs")
    p_split.add_argument("config", help="Base full-year config yaml")
    p_split.add_argument("--n-chunks", type=int, default=4)
    p_split.add_argument("--submit", action="store_true", help="sbatch each chunk config immediately")
    p_split.add_argument("--resume-from", default=None,
                          help="Only chunk the remaining range [resume-from, end_date] "
                               "(use when the base output_path already has a partial "
                               "_tmp cache from an earlier interrupted run)")
    p_split.set_defaults(func=cmd_split)

    p_merge = sub.add_parser("merge", help="Merge completed chunk extraction outputs into the base output_path")
    p_merge.add_argument("config", help="Base full-year config yaml")
    p_merge.add_argument("--n-chunks", type=int, default=4)
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
