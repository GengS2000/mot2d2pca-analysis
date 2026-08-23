#!/bin/bash
set -euo pipefail

# Default: rebuild the three deterministic truth tensors, then run only CTF.
# If exact truth tensors from the previous benchmark have already been placed
# at artifacts/competitor_truths/{sim1,sim2,sim3}.nc, use SKIP_TRUTH_PREP=1.
SKIP_TRUTH_PREP="${SKIP_TRUTH_PREP:-0}"
PARTITION="${PARTITION:-hebhcnormal02}"

if [[ "${SKIP_TRUTH_PREP}" == "1" ]]; then
  CTF_JOB=$(sbatch --parsable --partition="${PARTITION}" slurm/competitor_ctf_array.slurm)
  PREP_JOB="reused-existing-truth"
else
  PREP_JOB=$(sbatch --parsable --partition="${PARTITION}" slurm/prepare_competitor_truths.slurm)
  CTF_JOB=$(sbatch --parsable --partition="${PARTITION}" --dependency="afterok:${PREP_JOB}" slurm/competitor_ctf_array.slurm)
fi

MERGE_JOB=$(sbatch --parsable --partition="${PARTITION}" --dependency="afterok:${CTF_JOB}" slurm/merge_ctf_results.slurm)

echo "partition=${PARTITION}"
echo "prepare=${PREP_JOB}"
echo "ctf=${CTF_JOB}"
echo "merge=${MERGE_JOB}"
