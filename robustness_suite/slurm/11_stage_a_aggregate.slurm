#!/bin/bash
#SBATCH --job-name=mot2d2_stage_a_agg
#SBATCH --output=outputs/logs/stage_a_aggregate_%j.out
#SBATCH --error=outputs/logs/stage_a_aggregate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00

set -euo pipefail
: "${ROBUSTNESS_SUITE_ROOT:?ROBUSTNESS_SUITE_ROOT must be exported by the submitter}"
ROOT="$ROBUSTNESS_SUITE_ROOT"
cd "$ROOT"
: "${ROBUSTNESS_OUTPUT_ROOT:?ROBUSTNESS_OUTPUT_ROOT must be exported by the submitter}"
mkdir -p "$ROBUSTNESS_OUTPUT_ROOT/logs" "$ROBUSTNESS_OUTPUT_ROOT/aggregate"
source ~/miniconda3/bin/activate
conda activate micro3dimp
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
python scripts/aggregate_stage_a.py
