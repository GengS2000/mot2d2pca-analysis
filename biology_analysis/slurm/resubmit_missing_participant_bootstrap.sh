#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT=${1:-"$ROOT/results/participant_bootstrap_30"}
STAGE=${STAGE:-stage1}
PARTITION=${PARTITION:-hebhcnormal02}
ACCOUNT=${ACCOUNT:-}
MEM_PER_CPU=${MEM_PER_CPU:-3G}
MAX_CONCURRENT=${MAX_CONCURRENT:-48}
N_BOOTSTRAP=${N_BOOTSTRAP:-30}
PYTHON_BIN=${PYTHON_BIN:-python}
RSCRIPT=${RSCRIPT:-/public/software/VersionHub/R/4.4.2/Rscript}

if [[ "$STAGE" == "stage1" ]]; then
  MISSING="$OUTPUT_ROOT/aggregate/missing_stage1_task_ids.txt"
  SCRIPT=slurm/participant_bootstrap_stage1.slurm
  TIME=${TIME:-06:00:00}
elif [[ "$STAGE" == "stage2" ]]; then
  MISSING="$OUTPUT_ROOT/aggregate/missing_stage2_task_ids.txt"
  SCRIPT=slurm/participant_bootstrap_stage2.slurm
  TIME=${TIME:-12:00:00}
else
  echo "STAGE must be stage1 or stage2" >&2; exit 2
fi

[[ -f "$MISSING" ]] || { echo "Missing-task file not found: $MISSING" >&2; exit 1; }
N_MISSING=$(grep -cve '^[[:space:]]*$' "$MISSING" || true)
((N_MISSING > 0)) || { echo "No missing tasks."; exit 0; }
COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=1 --chdir="$ROOT" --partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && COMMON+=(--account="$ACCOUNT")
EXPORTS="ALL,BIOLOGY_ROOT=$ROOT,BOOTSTRAP_OUTPUT_ROOT=$OUTPUT_ROOT,PYTHON_BIN=$PYTHON_BIN,RSCRIPT=$RSCRIPT,N_BOOTSTRAP=$N_BOOTSTRAP"
ARRAY_SPEC=""
if [[ "$STAGE" == "stage1" ]]; then
  IDS=$(paste -sd, "$MISSING")
  ARRAY_SPEC="${IDS}%${MAX_CONCURRENT}"
else
  RETRY_IDS="$OUTPUT_ROOT/aggregate/retry_stage2_task_ids.txt"
  cp "$MISSING" "$RETRY_IDS"
  EXPORTS+=",STAGE2_TASK_ID_FILE=$RETRY_IDS"
  ARRAY_SPEC="0-$((N_MISSING - 1))%${MAX_CONCURRENT}"
fi
JOB=$(sbatch --parsable "${COMMON[@]}" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" \
  --export="$EXPORTS" --array="$ARRAY_SPEC" \
  --output="$OUTPUT_ROOT/logs/slurm/retry_${STAGE}_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/slurm/retry_${STAGE}_%A_%a.err" "$SCRIPT")
echo "Resubmitted $N_MISSING $STAGE missing task IDs as job $JOB"
