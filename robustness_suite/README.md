# MOT-(2D)^2 PCA robustness suite

This is an independent, manifest-driven Slurm workflow for parameter selection.
It does not modify `mot2d2pca`, `parameter`, or `downstream`.

## What is implemented

Stage A uses an outer test mask plus a separate inner validation mask.  Array
tasks can see only the training and validation data; the outer test mask is
stored but is not used in tuning.  Each task evaluates one
`(outer_repeat, fold, parameter)` combination and writes its own JSON result.
`aggregate_stage_a.py` applies the prespecified failure/zero-fill filters,
composite ranking, and one-standard-error rule, then writes the locked
parameters used by later experiments.

The default dataset is the existing CTF pseudo-truth
`parameter/sim2/ctf_imp_new.nc`.  Change its path or add datasets in
`configs/datasets.yaml` before submitting on the server.

## Server use

From the project root on the Slurm server (after copying this repository):

```bash
export PROJECT_ROOT="$PWD"
cd "$PROJECT_ROOT/robustness_suite"
# Edit configs/slurm_resources.yaml and configs/datasets.yaml once.
bash slurm/submit_stage_a.sh
```

The submitter creates the prepared tensor, masks, and manifest locally as
short setup steps, then submits the array and dependent aggregation job.  It
prints both Slurm job IDs.  To submit only missing/failed tasks after a run:

```bash
python scripts/verify_completeness.py --stage stage_a
bash slurm/resubmit_missing_stage_a.sh
```

`slurm/10_stage_a_array.slurm` intentionally requests one CPU per array task
and pins BLAS/OpenMP to one thread.  Adjust partition, account, environment,
memory, walltime, and the maximum concurrent array count in
`configs/slurm_resources.yaml` for your cluster.  The submitters read those
settings (plus `partition` and `account`) from that YAML file and override the
safe defaults embedded in the Slurm files.

`max_array_tasks` is the number of submitted Slurm array workers. Each worker
processes a strided subset of manifest rows sequentially, so set it no higher
than the account's submitted-job quota. With the configured limit of 200,
3,600 manifest rows are handled by 199 workers while one slot is reserved for
aggregation (about 18–19 rows per worker).
The submitter reserves one slot for aggregation and reduces the worker count
when the user already has other queued jobs.

## Expected outputs

By default, `results/stage_a/tasks/task_*.json` contains independent task results;
`results/aggregate/` contains `selected_params.json`, the one-SE set, task
metrics, failure records, and rankings.  The complete workflow does not save
large imputed tensors.

## Stage B and C

After Stage A aggregation has written `selected_params.json`, submit the next
two independent experiments in order:

```bash
bash slurm/submit_stage_b.sh
# after Stage B finishes
bash slurm/submit_stage_c.sh
```

Each B/C submitter first refreshes the configured prepared tensor through
`prepare_data.py`; it does not rely on a cache left by a prior Stage A run.

Stage B uses new deterministic masks for random entry, low-abundance-biased,
missed-time-point, missed-time-window, and subject-dropout mechanisms at 0.2, 0.5, and 0.8
missingness. Stage C compares full, temporal/mask/low-rank ablations and
one-direction baselines on new masks. `no_taxa_penalty` is not included because
the current formal implementation has no independent taxa-penalty component.
