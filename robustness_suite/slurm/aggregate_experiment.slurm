#!/bin/bash
#SBATCH --job-name=mot2d2_robustness_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
set -euo pipefail
: "${ROBUSTNESS_SUITE_ROOT:?missing suite root}"; : "${ROBUSTNESS_OUTPUT_ROOT:?missing output root}"; : "${STAGE:?missing stage}"
cd "$ROBUSTNESS_SUITE_ROOT"; source ~/miniconda3/bin/activate; conda activate micro3dimp
export PYTHONPATH="$ROBUSTNESS_SUITE_ROOT/src:${PYTHONPATH:-}"
python scripts/aggregate_stage.py --stage "$STAGE"
