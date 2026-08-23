#!/bin/bash
#SBATCH --job-name=sim12_merge
#SBATCH --output=server_runs/logs/%x_%j.out
#SBATCH --error=server_runs/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=hebhctest

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
source ~/miniconda3/bin/activate
conda activate q2-2024

python sim1_compare/sim1_methodcompare.py \
  --outdir server_runs/final/sim1 --merge-only --methods mean
python sim2-compare/ctf_reconstruct_compare.py \
  --outdir server_runs/final/sim2 --merge_only --methods penalized_2d2_pca

