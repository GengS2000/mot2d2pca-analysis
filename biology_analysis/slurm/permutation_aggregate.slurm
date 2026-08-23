#!/bin/bash
#SBATCH --job-name=mot_perm_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00

set -euo pipefail
: "${BIOLOGY_ROOT:?BIOLOGY_ROOT must be exported by the submitter}"
: "${PERM_OUTPUT_ROOT:?PERM_OUTPUT_ROOT must be exported by the submitter}"
: "${RSCRIPT:=/public/software/VersionHub/R/4.4.2/Rscript}"
: "${N_PERM:=1000}"
: "${PERM_SEED:=20260715}"
: "${PERM_METHODS:=linda,maaslin2}"
: "${AGGREGATE_SUBDIR:=aggregate}"

cd "$BIOLOGY_ROOT"
mkdir -p "$PERM_OUTPUT_ROOT/logs/slurm" "$PERM_OUTPUT_ROOT/aggregate"
"$RSCRIPT" permutation_slurm/aggregate_permutations.R \
  --project-dir "$BIOLOGY_ROOT" \
  --out-dir "$PERM_OUTPUT_ROOT" \
  --n-perm "$N_PERM" \
  --seed "$PERM_SEED" \
  --cohorts ECAM,DIABIMMUNE \
  --methods "$PERM_METHODS" \
  --aggregate-subdir "$AGGREGATE_SUBDIR"
