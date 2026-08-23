#!/bin/bash
#SBATCH --partition=hebhctest
mkdir -p logs
#SBATCH --job-name=sim_two
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=45G
#SBATCH --partition=hebhctest

set -euo pipefail

# Create the log directory


# ====== Environment: adjust to the local cluster ======
source ~/miniconda3/bin/activate
conda activate q2-2024

# Print run environment information
which python
python --version
hostname

echo "[START] $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"

# Cap threading-library thread counts to avoid oversubscription
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Run the task
python ctf_reconstruct_compare.py \
  --tax_path hmp2_mgx_taxonomy.tsv \
  --meta_path metagenomics.xlsx \
  --outdir artifacts/method_compare \
  --truth_path artifacts/method_compare/truth_ctf.nc \
  --fracs 0.5 \
  --n_repeats 1 \
  --mask_seed 123 \
  --algo_seed 0 \
  --eps 1e-3 \
  --min_pos_cells 10 \
  --min_participants 5 \
  --methods penalized_2d2_pca,ctf,mean,nearest_positive,temporal_knn \
  --d 5 \
  --q 3 \
  --lam 0.03 \
  --sigma 4.0 \
  --T0 30 \
  --T 30 \
  --ctf_rank 3 \
  --ctf_method gemelli \
  --ctf_seed 0 \
  --temporal_knn_k 3 \
  --temporal_knn_sigma 4.0 \
  --spline_min_points 4 \
  --equal_fill_value 0.0 \
  --dbn_dense_alpha 1.0 \
  --dbn_sparse_alpha 0.05 \
  --dbn_sparse_n_iter 150 \
  --glv_alpha 0.5
echo "[END] $(date)"
