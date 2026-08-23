#!/bin/bash
#SBATCH -J run_farmm
#SBATCH -p hebhcnormal02
#SBATCH -N 1
#SBATCH -n 8
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs artifacts/diabimmune_tempted_p2d2pca

python run_realdata_tempted_p2d2pca_comparison.py \
  --tax_path kraken_results.tsv \
  --meta_path 20200619_farmm_metadata.tsv \
  --outdir artifacts/FARMM_full \
  --eps 1e-3 \
  --min_pos_cells 10 \
  --min_participants 5 \
  --impute_mode eps_only \
  --imputed_input_mode grid_filled \
  --read_depth 100000 \
  --tempted_rank 3 \
  --n_repeats 20 \
  --quiet

