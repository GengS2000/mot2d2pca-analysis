#!/bin/bash
#SBATCH --job-name=ctf_demo
#SBATCH --partition=hebhctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs artifacts/method_compare_demo

source ~/miniconda3/bin/activate
conda activate micro3dimp

which python
python --version
hostname

echo "[START] $(date)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python sim3_compare.py \
  --tax_path hmp2_mgx_taxonomy.tsv \
  --meta_path metagenomics.xlsx \
  --outdir artifacts/method_compare_demo \
  --truth_path artifacts/method_compare_demo/truth_ctf.nc \
  --fracs 0.5 \
  --n_repeats 1 \
  --mask_seed 123 \
  --algo_seed 0 \
  --eps 1e-3 \
  --min_pos_cells 10 \
  --min_participants 5 \
  --methods penalized_2d2_pca,mean \
  --d 1 \
  --q 1 \
  --lam 0.1 \
  --sigma 4.0 \
  --T0 5 \
  --T 5 \
  --ctf_rank 3 \
  --ctf_method fallback \
  --ctf_seed 0 \
  --temporal_knn_k 3 \
  --temporal_knn_sigma 4.0 \
  --spline_min_points 4 \
  --equal_fill_value 0.0 \
  --dbn_dense_alpha 1.0 \
  --dbn_sparse_alpha 0.05 \
  --dbn_sparse_n_iter 30 \
  --glv_alpha 0.5

echo "[END] $(date)"
