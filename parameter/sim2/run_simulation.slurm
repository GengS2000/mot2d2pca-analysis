#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=parameter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=45G
#SBATCH --partition=hebhctest

set -euo pipefail

# Create the log directory
mkdir -p logs

# ====== Environment: adjust to the local cluster ======
source ~/miniconda3/bin/activate
conda activate micro3dimp

python tune_penalized_2d2_pca.py \
  --artifacts-dir artifacts/focused_pm1_run \
  --r-grid 1,2,3 \
  --lam-grid 0.1,0.25,0.5,0.75 \
  --sigma-grid 4,6,8 \
  --coarse-fracs 0.2,0.5,0.8 \
  --n-repeats-coarse 6 \
  --n-repeats-fine 12 \
  --top-k 4 \
  --verbose


# Job array example for multiple frac values:
# 1) set array in header, e.g. #SBATCH --array=0-4
# 2) define FRACS=(0.1 0.3 0.5 0.7 0.9)
# 3) use FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}
# 4) pass unique outdir per frac to avoid overwrite, e.g. ${BASE_OUT}/frac_${FRAC}
