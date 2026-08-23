# ECAM / DIABIMMUNE participant bootstrap (30 replicates) on the cluster

## Experiment definition

This workflow uses a participant-block nonparametric bootstrap: each cohort is resampled
with replacement along the participant dimension, drawing as many participants as the
original cohort holds, and the complete `taxon x time-bin` longitudinal block of a drawn
participant moves with it. A participant drawn more than once receives a unique
bootstrap ID, so the participant random intercept in LinDA/MaAsLin2 does not merge the
duplicate copies.

This differs from the label-permutation experiment in `permutation_slurm/`:

- the permutation test holds the data fixed and shuffles participant labels, to check
  calibration under the null;
- this bootstrap holds each participant's label and full longitudinal trajectory
  together, resamples participants and re-imputes, to check how stable imputation and
  differential inference are with respect to cohort composition.

Both cohorts start from their existing NetCDF files.
`run_realdata_tempted_p2d2pca_comparison.py` exists only under `ECAM_data/`;
`DIABIMMUNE_data/` has no file of that name. DIABIMMUNE nevertheless has a complete
`tensor_base_raw.nc`, `replace_mask.nc` and original configuration, so that entry script
does not need to be copied or recreated.

## Scheduling design

Stage 1: `2 cohorts x 30 bootstrap = 60` array tasks, each rerunning the MOT-(2D)2 PCA
imputation.

Stage 2: `2 cohorts x 30 bootstrap x 2 engines = 120` logical differential-inference
tasks. To stay within the Slurm submitted-task limit, each array element runs 2 logical
tasks sequentially by default, so only 60 array elements are submitted.

Resources are set explicitly:

```text
partition       = hebhcnormal02
cpus-per-task   = 1
mem-per-cpu     = 3G
```

The default concurrency budget is 60, minus the roughly 12 tasks already present on the
cluster, so Stage 1 and Stage 2 each add at most 48 concurrent tasks by default. The two
stages are never submitted concurrently: Stage 1 goes first, and Stage 2 is submitted
separately once Stage 1's aggregation has completed and released the array-element
quota. Stage 1's 60 tasks usually need about two waves; Stage 2's 60 array elements also
take about two waves, but each element contains two sequential analyses. Slurm backfills
as soon as any element finishes.

If "60 running at once" is meant to allow this experiment 60 new tasks rather than 60
including the existing 12, set `EXISTING_TASKS=0`, which raises the maximum concurrency
of both stages to 60. Do not raise the per-model core count at the same time; this
workflow fixes each model at 1 CPU.

## Files

- `participant_resampling_slurm/resample_impute_worker.py`: participant bootstrap plus
  MOT-(2D)2 PCA imputation;
- `participant_resampling_slurm/aggregate_imputations.py`: Stage 1 completeness and
  imputation-stability aggregation;
- `participant_resampling_slurm/stage2_inference_worker.R`: compact LinDA/MaAsLin2
  worker;
- `participant_resampling_slurm/aggregate_stage2.R`: aggregation of selection
  frequencies, effect intervals and cross-engine concordance;
- `slurm/participant_bootstrap_stage*.slurm`: the four Slurm job templates;
- `slurm/submit_participant_bootstrap_30.sh`: submits the full dependency chain;
- `slurm/resubmit_missing_participant_bootstrap.sh`: reruns by missing task ID.

## Submission on the cluster

Sync the whole `biology_analysis` directory to the cluster and enter the project root:

```bash
cd /public/home/USER/biology_analysis
chmod +x slurm/*.sh
```

If Python or R are not in the default locations, point at the real paths. Also set
`R_LIBS_USER` when the R packages live in a personal library:

```bash
export PYTHON_BIN=/path/to/python
export RSCRIPT=/path/to/R/4.4.2/Rscript
export R_LIBS_USER=/path/to/R/library
```

Submit Stage 1 first. Stage 2 is deliberately not submitted immediately, to avoid
hitting the account's array submit-job limit:

```bash
bash slurm/submit_participant_bootstrap_30.sh \
  "$PWD/results/participant_bootstrap_30"
```

Once the Stage 1 aggregation reports completeness, submit Stage 2:

```bash
bash slurm/submit_participant_bootstrap_stage2.sh \
  "$PWD/results/participant_bootstrap_30"
```

`submit_participant_bootstrap_stage2.sh` first checks `complete: true` in
`aggregate/stage1_aggregate_status.json`, so it cannot start the differential inference
while Stage 1 output is missing.

Under the assumption of "at most about 60 tasks in total", submitting directly gives 48
concurrent experiment tasks. To let this experiment use the full 60:

```bash
EXISTING_TASKS=0 TOTAL_TASK_BUDGET=60 \
bash slurm/submit_participant_bootstrap_30.sh \
  "$PWD/results/participant_bootstrap_30"
```

The budget calculation can also be bypassed by setting both limits explicitly:

```bash
STAGE1_MAX_CONCURRENT=60 STAGE2_MAX_CONCURRENT=60 \
bash slurm/submit_participant_bootstrap_30.sh \
  "$PWD/results/participant_bootstrap_30"
```

The script runs the Python and R preflight checks before submitting. Both of these must
appear:

```text
PYTHON_PREFLIGHT_OK
STAGE2_PREFLIGHT_OK
```

Stage 2 requires `MicrobiomeStat`, `Maaslin2` and `ncdf4`. The Stage 1 Python
environment requires `numpy`, `pandas`, `xarray` and the dependencies of this project's
`sim3_compare.py`.

## Output structure

```text
results/participant_bootstrap_30/
  stage1/<COHORT>/bootstrap_XXX/
    tensor_base_raw.nc
    tensor_p2d2pca_imputed.nc
    replace_mask.nc
    participant_order.csv
    subject_groups_long.csv
    bootstrap_draws.csv
    metrics.json
    SUCCESS.json
  stage2/results/<COHORT>/<ENGINE>/bootstrap_XXX.rds
  stage2/status/<COHORT>/<ENGINE>/bootstrap_XXX.rds
  stage2/logs/<COHORT>/<ENGINE>/bootstrap_XXX.log
  aggregate/
```

The large Stage 2 MaAsLin2 `features/` and `fits/` directories and the per-task
intermediates exist only in a temporary `work/`, and are deleted once the compact RDS
passes validation, to keep disk usage bounded.

## Main aggregated results

Stage 1:

- `stage1_completeness_report.csv`: whether each of the 60 tasks is complete;
- `missing_stage1_task_ids.txt`: array IDs that can be resubmitted directly;
- `imputation_run_metrics.csv`: per-draw coverage, replacement fraction, imputation
  change and runtime;
- `imputation_metric_summary.csv`: per-cohort mean, SD, median and 95% bootstrap
  interval;
- `taxon_imputation_stability.csv.gz`: per-taxon imputation stability across bootstrap
  replicates;
- `stage1_aggregate_status.json`: overall completion status.

The Stage 1 worker enforces two hard checks: every cell to be imputed must be finite,
and the maximum absolute difference at non-replaced cells before and after imputation
must not exceed `1e-10`.

Stage 2:

- `stage2_completeness_report.csv`, `missing_stage2_task_ids.txt`;
- `bootstrap_inference_run_summary.csv`: significant counts and raw p-value fractions per
  bootstrap replicate;
- `bootstrap_inference_stability.csv.gz`: per-taxon q<0.05 / q<0.10 selection frequency,
  effect-direction stability, median effect and 95% bootstrap interval;
- `imputed_vs_raw_bootstrap_stability.csv.gz`: selection-frequency and effect differences
  between the imputed and raw data;
- `engine_concordance_summary.csv`: Spearman correlation of effects, sign-agreement rate
  and selection-frequency correlation between LinDA and MaAsLin2;
- `stage2_aggregate_status.csv`: final completion status.

The definition of a "stable finding" should be written into the analysis plan in
advance, for example requiring all of: at least 24/30 successful replicates,
`selection_frequency_q05 >= 0.8`, `sign_stability >= 0.9`, and reporting the bootstrap
interval of the effect. Conclusions should not rest on a single q value or on a
threshold adjusted after the fact.

## Monitoring and reruns

```bash
squeue -u "$USER"
tail -f results/participant_bootstrap_30/logs/slurm/stage1_JOBID_0.out
cat results/participant_bootstrap_30/aggregate/stage1_aggregate_status.json
cat results/participant_bootstrap_30/aggregate/stage2_aggregate_status.csv
```

Stage 1 rerun:

```bash
STAGE=stage1 MAX_CONCURRENT=48 \
bash slurm/resubmit_missing_participant_bootstrap.sh \
  "$PWD/results/participant_bootstrap_30"
```

Stage 2 rerun:

```bash
STAGE=stage2 MAX_CONCURRENT=48 \
bash slurm/resubmit_missing_participant_bootstrap.sh \
  "$PWD/results/participant_bootstrap_30"
```

After the reruns finish, resubmit the corresponding aggregate `.slurm`, or run the
aggregation directly on a login node where short login-node jobs are permitted:

```bash
python participant_resampling_slurm/aggregate_imputations.py \
  --output-root "$PWD/results/participant_bootstrap_30" --n-bootstrap 30

/path/to/R/4.4.2/Rscript \
  participant_resampling_slurm/aggregate_stage2.R \
  --output-root "$PWD/results/participant_bootstrap_30" --n-bootstrap 30
```
