#!/bin/bash
#SBATCH --job-name=comp_ctf
#SBATCH --partition=hebhctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
# 197 scheduler jobs cover 750 logical benchmark tasks.  This keeps the whole
# prepare + array + merge workflow below AssocGrpSubmitJobsLimit=200.
#SBATCH --array=0-196%197
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
BENCHMARK_ENV="${BENCHMARK_ENV:-micro3dimp}"
N_REPEATS="${N_REPEATS:-50}"
N_ARRAY_WORKERS="${N_ARRAY_WORKERS:-197}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/competitor_results}"
cd "${PROJECT_DIR}"
mkdir -p logs "${RESULT_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${BENCHMARK_ENV}" ]]; then
  set +u
  source "${HOME}/miniconda3/bin/activate"
  conda activate "${BENCHMARK_ENV}"
  set -u
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

FRACS=(0.5 0.6 0.7 0.8 0.9)
SIMS=(sim1 sim2 sim3)
TOTAL_LOGICAL_TASKS=$((${#SIMS[@]} * ${#FRACS[@]} * N_REPEATS))

for ((IDX=SLURM_ARRAY_TASK_ID; IDX<TOTAL_LOGICAL_TASKS; IDX+=N_ARRAY_WORKERS)); do
  REPEAT=$((IDX % N_REPEATS))
  BLOCK=$((IDX / N_REPEATS))
  FRAC_INDEX=$((BLOCK % ${#FRACS[@]}))
  SIM_INDEX=$((BLOCK / ${#FRACS[@]}))
  SIM="${SIMS[$SIM_INDEX]}"
  FRAC="${FRACS[$FRAC_INDEX]}"
  TRUTH="artifacts/competitor_truths/${SIM}.nc"
  ALGO_SEED=20260322
  if [[ "${SIM}" == "sim2" ]]; then ALGO_SEED=0; fi

  echo "[logical task] index=${IDX} simulation=${SIM} frac=${FRAC} repeat=${REPEAT}"
  python competitor_task.py \
    --simulation "${SIM}" --method ctf --frac "${FRAC}" --repeat "${REPEAT}" \
    --truth-path "${TRUTH}" --outdir "${RESULT_ROOT}" \
    --sim1-seed-base 20260322 --sim1-mask-eligible-mode positive_only \
    --mask-seed-base 20260322 --algo-seed "${ALGO_SEED}" \
    --ctf-rank 3 --ctf-initializations 5 --ctf-max-iterations 50
done
