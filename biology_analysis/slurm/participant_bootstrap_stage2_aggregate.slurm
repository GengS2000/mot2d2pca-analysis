#!/bin/bash
#SBATCH --job-name=p2d2_boot_da_agg
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=04:00:00

set -euo pipefail
: "${BIOLOGY_ROOT:?BIOLOGY_ROOT must be exported}"
: "${BOOTSTRAP_OUTPUT_ROOT:?BOOTSTRAP_OUTPUT_ROOT must be exported}"
: "${RSCRIPT:=/public/software/VersionHub/R/4.4.2/Rscript}"
: "${N_BOOTSTRAP:=30}"
: "${STAGE2_METHODS:=linda,maaslin2}"
: "${AGGREGATE_SUBDIR:=aggregate}"
cd "$BIOLOGY_ROOT"
"$RSCRIPT" participant_resampling_slurm/aggregate_stage2.R \
  --output-root "$BOOTSTRAP_OUTPUT_ROOT" --n-bootstrap "$N_BOOTSTRAP" \
  --methods "$STAGE2_METHODS" --aggregate-subdir "$AGGREGATE_SUBDIR"
