#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=DA_sim
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
conda activate gemelli

# Example:
# sbatch simulation5/run_simulation.slurm /path/to/ctf_imp.nc /path/to/outdir 0.3 100 20260326
OUTDIR=artifacts/simulation3_datasave

SEED=${5:-20260326}

mkdir -p "${OUTDIR}" logs
 
python run_realdata_tempted_p2d2pca_comparison.py
 # Job array example for multiple frac values:
# 1) set array in header, e.g. #SBATCH --array=0-4
# 2) define FRACS=(0.1 0.3 0.5 0.7 0.9)
# 3) use FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}
# 4) pass unique outdir per frac to avoid overwrite, e.g. ${BASE_OUT}/frac_${FRAC}
