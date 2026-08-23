#!/bin/bash
#SBATCH --job-name=mot2d2_robustness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
set -euo pipefail
: "${ROBUSTNESS_SUITE_ROOT:?missing suite root}"; : "${ROBUSTNESS_OUTPUT_ROOT:?missing output root}"; : "${STAGE:?missing stage}"
ROOT="$ROBUSTNESS_SUITE_ROOT"; cd "$ROOT"
source ~/miniconda3/bin/activate; conda activate micro3dimp
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$ROBUSTNESS_OUTPUT_ROOT/$STAGE/tasks" "$ROBUSTNESS_OUTPUT_ROOT/logs"
TOTAL=$(( $(wc -l < "manifests/${STAGE}.csv") - 1 )); RETRIES=${TASK_RETRIES:-2}
for ((INDEX=SLURM_ARRAY_TASK_ID; INDEX<TOTAL; INDEX+=SLURM_ARRAY_TASK_COUNT)); do
  OK=0
  for ((A=1; A<=RETRIES; A++)); do
    echo "[$(date '+%F %T')] stage=$STAGE worker=$SLURM_ARRAY_TASK_ID task=$INDEX attempt=$A/$RETRIES"
    if python -u "scripts/run_${STAGE}_task.py" --manifest "manifests/${STAGE}.csv" --task-id "$INDEX"; then OK=1; break; fi
  done
  if ((OK==0)); then echo "task=$INDEX failed after retries" >&2; fi
done
# Failed logical tasks have status JSON and are reported by aggregation.
exit 0
