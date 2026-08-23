#!/bin/bash
set -euo pipefail

# The merge job starts only after the CTF array finishes successfully.
PREP_JOB=$(sbatch --parsable slurm/prepare_competitor_truths.slurm)
CTF_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" slurm/competitor_ctf_array.slurm)
MERGE_JOB=$(sbatch --parsable --dependency="afterok:${CTF_JOB}" slurm/merge_competitor_results.slurm)

echo "prepare=${PREP_JOB}"
echo "ctf=${CTF_JOB}"
echo "merge=${MERGE_JOB}"
