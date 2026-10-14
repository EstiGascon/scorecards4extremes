#!/bin/bash
#SBATCH --job-name=watch_score
#SBATCH --output=watch_%j.out
#SBATCH --error=watch_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/moeg/scorecards4extremes

# ==============================================================================
# Watches a set of parallel chunk-extraction jobs (see scripts/chunk_extraction.py)
# and, once they all finish, automatically merges the chunk outputs and submits
# the base config's scoring/plotting job — no manual intervention needed.
#
# Any chunk that did NOT complete cleanly (e.g. cancelled by SLURM TIME_LIMIT)
# is automatically resubmitted (it resumes from its own per-date _tmp cache),
# and the watcher waits again before merging.
#
# Usage:
#   sbatch scripts/watch_and_score.sh <base_config.yaml> <n_chunks> [extra_config.yaml ...]
#
# Optional extra configs are further scoring runs over the SAME merged extraction
# (e.g. a different filter.area / orography split). Each is submitted with
# --dependency=afterany:<merge+score job> and S4E_REQUIRE_EXTRACTED=1, so it only
# scores and never re-extracts. If S4E_POST_SCRIPT is set (path to a batch script),
# it is submitted after all scoring jobs finish (e.g. a combined-region plot).
# ==============================================================================

set -uo pipefail

CONFIG=${1:?"Usage: sbatch watch_and_score.sh <base_config.yaml> <n_chunks>"}
N_CHUNKS=${2:?"Usage: sbatch watch_and_score.sh <base_config.yaml> <n_chunks>"}
EXTRA_CONFIGS=("${@:3}")

REPO_ROOT="/home/moeg/scorecards4extremes"
source "$REPO_ROOT/.venv/bin/activate"

BASE_NAME=$(basename "$CONFIG" .yaml)
CHUNK_DIR="$REPO_ROOT/configs/_chunks/$BASE_NAME"
JOBIDS_FILE="$CHUNK_DIR/job_ids.txt"

echo "$(date): watching chunk jobs for $CONFIG (n_chunks=$N_CHUNKS)"
cat "$JOBIDS_FILE"

wait_for_jobs () {
    # $1: name of a bash array variable holding job IDs to wait for
    local -n ids=$1
    while true; do
        local active=0
        for jid in "${ids[@]}"; do
            if squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; then
                active=1
            fi
        done
        if [[ $active -eq 0 ]]; then
            break
        fi
        echo "$(date): still waiting on: ${ids[*]}"
        sleep 300
    done
}

mapfile -t JOB_IDS < "$JOBIDS_FILE"
wait_for_jobs JOB_IDS

echo "$(date): all tracked chunk jobs left the queue — checking completion status"

chunk_output_dir () {
    # $1: chunk index -> absolute extraction output dir for that chunk
    local cfg="$CHUNK_DIR/${BASE_NAME}_chunk${1}.yaml"
    local p
    p=$(grep -E '^[[:space:]]*output_path:' "$cfg" | head -1 \
        | sed -E 's/^[[:space:]]*output_path:[[:space:]]*//')
    p=${p#\"}; p=${p%\"}; p=${p#\'}; p=${p%\'}; p=${p#./}
    case "$p" in
        /*) echo "$p" ;;
        *)  echo "$REPO_ROOT/$p" ;;
    esac
}

chunk_done () {
    # $1: chunk index -> succeeds if extraction finished: run_step3 writes the
    # per-day parquet(s) and removes the per-date _tmp cache on success. This
    # checks the actual artifact rather than a log string, so it is correct no
    # matter which job (original or an auto-resubmitted one) produced it.
    local dir
    dir=$(chunk_output_dir "$1")
    [[ ! -d "$dir/_tmp" ]] && compgen -G "$dir/"'*_day*.parquet' > /dev/null
}

orphan_job_for_chunk () {
    # $1: chunk index -> prints the id of a still-queued scorecards_quaver job
    # already (re)running this chunk config — i.e. an auto-resubmit spawned by
    # submit_job_quaver.sh after a segfault — or empty if none has started yet.
    local chunk_base="${BASE_NAME}_chunk${1}.yaml"
    local jid out
    for jid in $(squeue -u "$USER" -h -n scorecards_quaver -o '%i' 2>/dev/null); do
        out="$REPO_ROOT/scorecards_${jid}.out"
        if [[ -f "$out" ]] && grep -q "run.py .*${chunk_base}" "$out"; then
            echo "$jid"; return
        fi
    done
    echo ""
}

RETRY_ROUND=0
while true; do
    NEED_RETRY=0
    for i in $(seq 0 $((N_CHUNKS - 1))); do
        if chunk_done "$i"; then
            echo "  chunk$i: OK"
            continue
        fi
        NEED_RETRY=1
        # Prefer adopting an auto-resubmit already working this chunk over
        # launching our own copy, so two jobs never write the same output dir.
        adopt=$(orphan_job_for_chunk "$i")
        if [[ -n "$adopt" ]]; then
            echo "  chunk$i: incomplete — adopting in-flight job $adopt"
            JOB_IDS[$i]=$adopt
        else
            CHUNK_CONFIG="$CHUNK_DIR/${BASE_NAME}_chunk${i}.yaml"
            echo "  chunk$i: incomplete — resubmitting $CHUNK_CONFIG"
            # S4E_MAX_RETRIES=0: the watcher is the sole retry authority, so a
            # watcher-launched retry must not spawn its own untracked orphan.
            OUT=$(sbatch --export=ALL,S4E_MAX_RETRIES=0 \
                "$REPO_ROOT/scripts/submit_job_quaver.sh" "$CHUNK_CONFIG")
            echo "    $OUT"
            JOB_IDS[$i]=$(echo "$OUT" | awk '{print $NF}')
        fi
    done

    if [[ $NEED_RETRY -eq 0 ]]; then
        break
    fi

    RETRY_ROUND=$((RETRY_ROUND + 1))
    if [[ $RETRY_ROUND -gt 5 ]]; then
        echo "$(date): giving up after 5 retry rounds — check logs manually"
        exit 1
    fi

    printf '%s\n' "${JOB_IDS[@]}" > "$JOBIDS_FILE"
    wait_for_jobs JOB_IDS
done

echo "$(date): all chunks completed cleanly — submitting merge+scoring job"
# The merge (large pd.concat of ensemble parquets) and scoring run together in
# ONE 128G job: submit_job_quaver.sh honours S4E_MERGE_CHUNKS by merging first,
# then aborts before run.py if the merge fails. Doing the merge inline here in
# the 1G watcher OOM-killed it, leaving no merged files, which made run.py
# silently re-extract everything from scratch.
JID=$(sbatch --parsable --export=ALL,S4E_MERGE_CHUNKS="$N_CHUNKS" \
    "$REPO_ROOT/scripts/submit_job_quaver.sh" "$CONFIG")
echo "Submitted batch job $JID"

ALL_IDS="$JID"
for extra in "${EXTRA_CONFIGS[@]}"; do
    eid=$(sbatch --parsable --dependency=afterany:"$JID" \
        --export=ALL,S4E_REQUIRE_EXTRACTED=1 \
        "$REPO_ROOT/scripts/submit_job_quaver.sh" "$extra")
    echo "Submitted extra scoring job $eid for $extra (after $JID)"
    ALL_IDS="$ALL_IDS:$eid"
done
if [[ -n "${S4E_POST_SCRIPT:-}" ]]; then
    pid=$(sbatch --parsable --dependency=afterany:"$ALL_IDS" "$S4E_POST_SCRIPT")
    echo "Submitted post-processing job $pid ($S4E_POST_SCRIPT) after $ALL_IDS"
fi

echo "$(date): watcher done"
