#!/bin/bash
#SBATCH --job-name=truth_merge
#SBATCH --output=server_runs/logs/%x_%j.out
#SBATCH --error=server_runs/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=hebhctest

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
source ~/miniconda3/bin/activate
conda activate q2-2024
python merge_truth_diagnostics.py \
  --root server_runs/truth_sweep \
  --output server_runs/truth_sweep/truth_candidates_ranked.csv

