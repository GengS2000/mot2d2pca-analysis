#!/bin/bash
set -euo pipefail
mkdir -p server_runs/logs
sweep_job="$(sbatch --parsable tune_truths_q2.slurm)"
merge_job="$(sbatch --parsable --dependency="afterok:${sweep_job}" merge_truth_sweep_q2.slurm)"
echo "Sim1 truth sweep job: ${sweep_job} (24 tasks, max 24 concurrent)"
echo "diagnostic merge job: ${merge_job} (afterok:${sweep_job})"
