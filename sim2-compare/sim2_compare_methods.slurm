#!/bin/bash
#SBATCH --job-name=sim2_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=45G
#SBATCH --partition=hebhctest

set -euo pipefail
mkdir -p logs

source ~/miniconda3/bin/activate
conda activate q2-2024

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python ctf_reconstruct_compare.py \
  --tax_path hmp2_mgx_taxonomy.tsv \
  --meta_path metagenomics.xlsx \
  --outdir results_sim2_compare \
  --n_repeats 50 \
  --fracs 0.5,0.6,0.7,0.8,0.9 \
  --mask_seed 20260322 \
  --methods \
    penalized_2d2_pca \
    ctf \
    mean \
    nearest_positive \
    temporal_knn \
  --eps 1e-3 \
  --d 5 \
  --q 3 \
  --lam 0.03 \
  --sigma 4.0 \
  --T0 30 \
  --T 30 \
  --ctf_method gemelli \
  --temporal-knn-k 3 \
  --temporal-knn-sigma 4
