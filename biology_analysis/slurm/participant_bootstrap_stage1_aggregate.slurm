#!/bin/bash
#SBATCH --job-name=p2d2_boot_imp_agg
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=01:00:00

set -euo pipefail
: "${BIOLOGY_ROOT:?BIOLOGY_ROOT must be exported}"
: "${BOOTSTRAP_OUTPUT_ROOT:?BOOTSTRAP_OUTPUT_ROOT must be exported}"
: "${PYTHON_BIN:=python}"
: "${N_BOOTSTRAP:=30}"
cd "$BIOLOGY_ROOT"
"$PYTHON_BIN" participant_resampling_slurm/aggregate_imputations.py \
  --output-root "$BOOTSTRAP_OUTPUT_ROOT" --n-bootstrap "$N_BOOTSTRAP"
