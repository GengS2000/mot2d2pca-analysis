#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source ~/miniconda3/bin/activate
conda activate micro3dimp
OUTPUT_CONFIG=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('output_dir', 'results'))")
if [[ "$OUTPUT_CONFIG" = /* ]]; then OUTPUT_ROOT="$OUTPUT_CONFIG"; else OUTPUT_ROOT="$ROOT/$OUTPUT_CONFIG"; fi
export ROBUSTNESS_OUTPUT_ROOT="$OUTPUT_ROOT"
python scripts/verify_completeness.py --stage stage_a
IDS=$(paste -sd, "$OUTPUT_ROOT/stage_a/failed_or_missing_task_ids.txt")
if [[ -z "$IDS" ]]; then echo "No tasks to resubmit."; exit 0; fi
N_MISSING=$(wc -l < "$OUTPUT_ROOT/stage_a/failed_or_missing_task_ids.txt")
MAX=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['array_concurrency'])")
WORKERS=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('max_array_tasks', 200))")
PARTITION=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['partition'])")
ACCOUNT=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('account', ''))")
CPUS=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['cpus_per_task'])")
MEM=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['mem'])")
WALLTIME=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml'))['time'])")
TASK_RETRIES=$(python -c "import yaml; print(yaml.safe_load(open('configs/slurm_resources.yaml')).get('task_retries', 2))")
SBATCH_COMMON=(--partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" --time="$WALLTIME" --chdir="$ROOT")
if [[ -n "$ACCOUNT" ]]; then SBATCH_COMMON+=(--account="$ACCOUNT"); fi
if (( WORKERS > N_MISSING )); then WORKERS=$N_MISSING; fi
if (( MAX > WORKERS )); then MAX=$WORKERS; fi
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/stage_a/tasks" "$OUTPUT_ROOT/aggregate"
EXPORTS="ALL,ROBUSTNESS_SUITE_ROOT=$ROOT,ROBUSTNESS_OUTPUT_ROOT=$OUTPUT_ROOT,TASK_RETRIES=$TASK_RETRIES,TASK_ID_FILE=$OUTPUT_ROOT/stage_a/failed_or_missing_task_ids.txt"
JOB=$(sbatch --parsable "${SBATCH_COMMON[@]}" --export="$EXPORTS" --output="$OUTPUT_ROOT/logs/stage_a_retry_%A_%a.out" --error="$OUTPUT_ROOT/logs/stage_a_retry_%A_%a.err" --array="0-$((WORKERS - 1))%${MAX}" slurm/10_stage_a_array.slurm)
AGG=$(sbatch --parsable "${SBATCH_COMMON[@]}" --export="ALL,ROBUSTNESS_SUITE_ROOT=$ROOT,ROBUSTNESS_OUTPUT_ROOT=$OUTPUT_ROOT" --output="$OUTPUT_ROOT/logs/stage_a_aggregate_%j.out" --error="$OUTPUT_ROOT/logs/stage_a_aggregate_%j.err" --dependency="afterok:${JOB}" slurm/11_stage_a_aggregate.slurm)
echo "Resubmitted ${N_MISSING} task(s) over ${WORKERS} worker(s): ${JOB}; aggregation: ${AGG}"
