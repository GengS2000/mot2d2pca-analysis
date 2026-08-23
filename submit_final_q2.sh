#!/bin/bash
set -euo pipefail
: "${SIM1_TRUTH_PATH:?Export SIM1_TRUTH_PATH before submission}"
: "${SIM2_TRUTH_PATH:?Export SIM2_TRUTH_PATH before submission}"
test -f "$SIM1_TRUTH_PATH"
test -f "$SIM2_TRUTH_PATH"
mkdir -p server_runs/logs

array_job="$(sbatch --parsable \
  --export="ALL,SIM1_TRUTH_PATH=${SIM1_TRUTH_PATH},SIM2_TRUTH_PATH=${SIM2_TRUTH_PATH},SIM1_D=${SIM1_D:-5},SIM1_Q=${SIM1_Q:-3},SIM1_LAM=${SIM1_LAM:-0.03},SIM1_SIGMA=${SIM1_SIGMA:-4},SIM2_D=${SIM2_D:-5},SIM2_Q=${SIM2_Q:-3},SIM2_LAM=${SIM2_LAM:-0.03},SIM2_SIGMA=${SIM2_SIGMA:-4}" \
  run_sim1_sim2_q2.slurm)"
merge_job="$(sbatch --parsable --dependency="afterok:${array_job}" merge_sim1_sim2_q2.slurm)"
echo "joint Sim1+Sim2 array: ${array_job} (500 shards, max 180 concurrent)"
echo "final merge job: ${merge_job} (afterok:${array_job})"

