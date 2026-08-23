#!/bin/bash
#SBATCH --job-name=mot2d2_stage_a
#SBATCH --output=outputs/logs/stage_a_%A_%a.out
#SBATCH --error=outputs/logs/stage_a_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --array=0-0

set -euo pipefail
# Slurm executes a spool copy of this file, so BASH_SOURCE points into
# /opt/.../slurmd rather than the repository.  The submitter exports the real
# suite path explicitly.
: "${ROBUSTNESS_SUITE_ROOT:?ROBUSTNESS_SUITE_ROOT must be exported by the submitter}"
ROOT="$ROBUSTNESS_SUITE_ROOT"
cd "$ROOT"
: "${ROBUSTNESS_OUTPUT_ROOT:?ROBUSTNESS_OUTPUT_ROOT must be exported by the submitter}"
mkdir -p "$ROBUSTNESS_OUTPUT_ROOT/logs" "$ROBUSTNESS_OUTPUT_ROOT/stage_a/tasks"
source ~/miniconda3/bin/activate
conda activate micro3dimp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# The scheduler limits this account to 200 submitted array elements.  Spread
# the complete manifest across those workers: worker i runs i, i+count, ... .
TOTAL_TASKS=$(( $(wc -l < manifests/stage_a.csv) - 1 ))
FAILED=0
TASK_RETRIES=${TASK_RETRIES:-2}
if [[ -n "${TASK_ID_FILE:-}" ]]; then
  mapfile -t SELECTED_TASK_IDS < "$TASK_ID_FILE"
  TOTAL_TASKS=${#SELECTED_TASK_IDS[@]}
else
  SELECTED_TASK_IDS=()
fi
for ((INDEX=SLURM_ARRAY_TASK_ID; INDEX<TOTAL_TASKS; INDEX+=SLURM_ARRAY_TASK_COUNT)); do
  if [[ ${#SELECTED_TASK_IDS[@]} -gt 0 ]]; then TASK_ID=${SELECTED_TASK_IDS[$INDEX]}; else TASK_ID=$INDEX; fi
  echo "[$(date '+%F %T')] worker=${SLURM_ARRAY_TASK_ID} task=${TASK_ID}/${TOTAL_TASKS}"
  SUCCESS=0
  for ((ATTEMPT=1; ATTEMPT<=TASK_RETRIES; ATTEMPT++)); do
    echo "[$(date '+%F %T')] task=${TASK_ID} attempt=${ATTEMPT}/${TASK_RETRIES}"
    if python -u scripts/run_stage_a_task.py --manifest manifests/stage_a.csv --task-id "$TASK_ID"; then
      SUCCESS=1
      break
    fi
  done
  if (( SUCCESS == 0 )); then
    FAILED=$((FAILED + 1))
    echo "[$(date '+%F %T')] task=${TASK_ID} failed after ${TASK_RETRIES} attempts; continuing" >&2
  fi
done
# Every manifest task has a status JSON even on failure.  Aggregation must run
# to produce the complete failure report instead of being stranded by afterok.
echo "[$(date '+%F %T')] worker=${SLURM_ARRAY_TASK_ID} finished; failed_tasks=${FAILED}"
exit 0
