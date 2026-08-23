#!/bin/bash
#SBATCH --job-name=merge_comp
#SBATCH --partition=hebhctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
BENCHMARK_ENV="${BENCHMARK_ENV:-micro3dimp}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/competitor_results}"
cd "${PROJECT_DIR}"
mkdir -p logs
source "${HOME}/miniconda3/bin/activate"
conda activate "${BENCHMARK_ENV}"
python merge_competitor_results.py --root "${RESULT_ROOT}" --n-repeats 50 --fracs 0.5 0.6 0.7 0.8 0.9

