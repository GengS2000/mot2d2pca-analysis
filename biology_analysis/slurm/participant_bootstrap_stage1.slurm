#!/bin/bash
#SBATCH --job-name=p2d2_boot_imp
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=06:00:00
#SBATCH --array=0-59%48

set -euo pipefail
: "${BIOLOGY_ROOT:?BIOLOGY_ROOT must be exported}"
: "${BOOTSTRAP_OUTPUT_ROOT:?BOOTSTRAP_OUTPUT_ROOT must be exported}"
: "${PYTHON_BIN:=python}"
: "${N_BOOTSTRAP:=30}"
: "${BOOTSTRAP_SEED:=20260716}"

cd "$BIOLOGY_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" participant_resampling_slurm/resample_impute_worker.py \
  --project-dir "$BIOLOGY_ROOT" \
  --output-root "$BOOTSTRAP_OUTPUT_ROOT" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$BOOTSTRAP_SEED" \
  --resume
