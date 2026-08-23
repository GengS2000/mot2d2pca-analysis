#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper:
#   bash slurm/submit_known_mask_pipeline.sh
#
# Optional overrides:
#   OUTDIR=outputs_known_mask REPEATS=20 PYTHON_BIN=/path/to/python bash slurm/submit_known_mask_pipeline.sh

exp_job_id="$(sbatch --parsable slurm/submit_known_mask_experiments.sbatch)"
echo "Submitted known-mask experiment array: ${exp_job_id}"

analysis_job_id="$(sbatch --parsable --dependency=afterok:${exp_job_id} slurm/submit_known_mask_analysis.sbatch)"
echo "Submitted dependent analysis job: ${analysis_job_id}"
