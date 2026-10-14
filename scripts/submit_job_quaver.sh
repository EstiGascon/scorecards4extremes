#!/bin/bash
#SBATCH --job-name=scorecards_quaver
#SBATCH --output=scorecards_%j.out
#SBATCH --error=scorecards_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --chdir=/home/moeg/scorecards4extremes

# ==============================================================================
# Batch submission for scorecards4extremes — QUAVER / VTB extraction backend
# ==============================================================================
# Same as submit_job.sh but ALSO puts vtb (the ECMWF verification toolbox) on
# PYTHONPATH so backend: quaver_extract works.
#
# Usage:
#   sbatch submit_job_quaver.sh configs/deterministic/config_tp24_method3_quaverextract_iekm_vs_ifs.yaml
# ==============================================================================

echo "=========================================="
echo "Scorecards4Extremes — QUAVER Batch Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

CONFIG=${1:?'Usage: sbatch submit_job_quaver.sh <config_file.yaml>'}

# Minimal setup (avoid verbose output that might buffer)
export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p $TMPDIR
export METVIEW_PYTHON_START_TIMEOUT=120

# ecmwf-toolbox env (metview binary + eccodes), set explicitly to avoid module
# conflicts on compute nodes (same block as submit_job.sh).
_TB="/usr/local/apps/ecmwf-toolbox/2026.04.0.0/GNU/8.5"
export PATH="${_TB}/bin:$PATH"
export ECMWF_TOOLBOX_DIR="${_TB}"
export ECCODES_DIR="${_TB}"
export ECCODES_PYTHON_USE_FINDLIBS="1"
export FINDLIBS_DISABLE_PACKAGE="yes"
export MAGICS_DIR="${_TB}"
export MAGPLUS_HOME="${_TB}"
export MAGPLUS_DEV="OFF"
export MAGPLUS_INFO="OFF"
export METVIEW_DIR="${_TB}"
export FDB5_DIR="${_TB}"
export ODC_DIR="${_TB}"

# Project virtualenv (python 3.12): provides metview bindings, scores, xarray,
# pandas, pyyaml, plus the vtb runtime deps installed into it
# (python-magic, jsonschema, psycopg[binary]).
source "/home/moeg/scorecards4extremes/.venv/bin/activate"

# vtb (VTB / quaver toolbox) — python 3.12 build, matches the venv interpreter.
export PYTHONPATH="/usr/local/apps/vtb/1.3.3/lib/python3.12/site-packages:${PYTHONPATH}"

# Optional: merge chunked extraction outputs before scoring. Set by the watcher
# (scripts/watch_and_score.sh) via --export=ALL,S4E_MERGE_CHUNKS=<n_chunks>.
# The merge (a pd.concat of large ensemble parquets) runs HERE, inside the 128G
# scoring job, instead of in the 1G watcher where it was OOM-killed. If the
# merge fails we must NOT fall through to run.py, which would silently
# re-extract everything from scratch.
if [[ -n "${S4E_MERGE_CHUNKS:-}" ]]; then
    echo "=========================================="
    echo "Merging ${S4E_MERGE_CHUNKS} chunk outputs before scoring"
    echo "=========================================="
    python -u scripts/chunk_extraction.py merge "$CONFIG" --n-chunks "$S4E_MERGE_CHUNKS"
    merge_rc=$?
    if [[ $merge_rc -ne 0 ]]; then
        echo "MERGE FAILED (exit $merge_rc) — aborting before scoring to avoid re-extraction"
        exit $merge_rc
    fi
    echo ""
fi

# Scoring-only runs over an existing extraction (see watch_and_score.sh extra
# configs): refuse to start if the extracted day files are missing, otherwise
# run.py would silently re-extract everything. Exits before the retry logic.
if [[ "${S4E_REQUIRE_EXTRACTED:-0}" == "1" ]]; then
    python -u scripts/check_extracted.py "$CONFIG" || {
        echo "EXTRACTED DATA MISSING — not running scoring (would re-extract)"; exit 3; }
fi

echo ""
echo "=========================================="
echo "Running: python run.py $CONFIG"
echo "=========================================="
echo ""

# Run (unbuffered Python output)
python -u run.py "$CONFIG"

exit_code=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

# Auto-resubmit on failure: long quaver/vtb runs can die from a native
# (Metview/eccodes) segfault after many hours. Per-date extraction is
# already cached to extracted_points/<...>/_tmp, so a resubmission just
# resumes from the last completed date instead of restarting. Bounded
# retries avoid an infinite loop on a genuinely broken config.
S4E_RETRY_COUNT=${S4E_RETRY_COUNT:-0}
S4E_MAX_RETRIES=${S4E_MAX_RETRIES:-3}
if [[ $exit_code -ne 0 && $S4E_RETRY_COUNT -lt $S4E_MAX_RETRIES ]]; then
    next_retry=$((S4E_RETRY_COUNT + 1))
    echo "Non-zero exit — auto-resubmitting (retry ${next_retry}/${S4E_MAX_RETRIES})..."
    sbatch --export=ALL,S4E_RETRY_COUNT=${next_retry} \
        /home/moeg/scorecards4extremes/scripts/submit_job_quaver.sh "$CONFIG"
fi

exit $exit_code
