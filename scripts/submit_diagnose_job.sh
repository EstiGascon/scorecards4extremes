#!/bin/bash
# Usage: ./submit_diagnose_job.sh --config <yaml> --day N --threshold-pct P [--season S] [--orog O]
#
# Creates a unique args file and a one-off sbatch script with the path
# hardcoded, so multiple jobs can be submitted back-to-back without race
# conditions on shared files.

set -e

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 --config <yaml> --day N --threshold-pct P [--season S] [--orog O]"
    exit 1
fi

BASEDIR="$(cd "$(dirname "$0")/.." && pwd)"
ARGS="$*"

# Unique identifier based on timestamp + random suffix
UID_TAG="$(date +%Y%m%d_%H%M%S)_$$"
ARGS_FILE="${BASEDIR}/.diagnose_args_${UID_TAG}"
JOB_SCRIPT="${BASEDIR}/.submit_diagnose_${UID_TAG}.sh"

# Write args to unique file
echo "$ARGS" > "$ARGS_FILE"
echo "Args written to: $ARGS_FILE"

# Generate a self-contained sbatch script with the path hardcoded
cat > "$JOB_SCRIPT" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=diagnose_extremes
#SBATCH --output=${BASEDIR}/diagnose_%j.out
#SBATCH --error=${BASEDIR}/diagnose_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=${BASEDIR}

echo "=========================================="
echo "Diagnose Extremes Batch Job"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$HOSTNAME"
echo "Start time: \$(date)"
echo "Args file: ${ARGS_FILE}"

DIAG_ARGS=\$(cat "${ARGS_FILE}")
echo "Args: \$DIAG_ARGS"
echo ""

export TMPDIR=/ec/res4/scratch/\$USER/tmp
mkdir -p \$TMPDIR
export METVIEW_PYTHON_START_TIMEOUT=30

module load ecmwf-toolbox/new
module load python3

echo ""
echo "Running: python3 diagnostics/diagnose_extremes.py \$DIAG_ARGS"
echo ""

python3 -u diagnostics/diagnose_extremes.py \$DIAG_ARGS
exit_code=\$?

# Clean up per-job files once done
rm -f "${ARGS_FILE}" "${JOB_SCRIPT}"

echo ""
echo "=========================================="
echo "Job completed with exit code: \$exit_code"
echo "End time: \$(date)"
echo "=========================================="
exit \$exit_code
SBATCH_EOF

chmod +x "$JOB_SCRIPT"

# Submit
JOB_ID=$(sbatch "$JOB_SCRIPT" | grep -o '[0-9]*')
echo "Submitted job $JOB_ID for: $ARGS"
