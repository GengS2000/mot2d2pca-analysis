# glmmPQL permutation and participant bootstrap (reusing existing cluster results)

This workflow adds glmmPQL inference only. It does not regenerate the permutation
labels and does not rerun participant-bootstrap Stage 1 or the MOT-(2D)2 PCA imputation.

## Directories reused on the cluster

- Permutation: `results/permutation_1000`
  - reads the existing `labels/<COHORT>/perm_XXXX.csv.gz`;
  - appends `results|status|logs/<COHORT>/glmmpql/`;
  - writes the new aggregate to `aggregate_glmmpql/` without overwriting the old
    `aggregate/`.
- Bootstrap: `results/participant_bootstrap_30`
  - reads the existing `stage1/<COHORT>/bootstrap_XXX/`;
  - appends `stage2/results|status|logs/<COHORT>/glmmpql/`;
  - writes the new aggregate to `aggregate_glmmpql/` without overwriting the old
    `aggregate/`.

The model is identical to the glmmPQL fit in `realdata_glmmPQL_ZIGMM_runner.R`:

```text
y ~ subject_group + bin_c
random = ~ 1 | participant
family = gaussian(identity)
MASS::glmmPQL(..., niter = 10)
```

The response is the log10 abundance stored in the NetCDF files. The sample set,
participant order, label levels and taxon mapping are still constructed by the original
per-cohort NetCDF scripts. A failed single-taxon fit is retained with
`fit_status != ok` and NA p/q values, and does not discard the whole shard.

## Synchronization and permissions

After syncing this update into the project directory on the cluster:

```bash
cd /path/to/penalized_2d2_pca/biology_analysis
chmod +x slurm/*.sh
```

If the R packages live in a personal library:

```bash
export R_LIBS_USER=/path/to/your/R/library
export RSCRIPT=/path/to/R/4.4.2/Rscript
```

The submit scripts check for `ncdf4`, `MASS` and `nlme`.

## 1. Permutation smoke test

Runs IDs 0-1 against the original permutation root. Successful results are picked up
directly by the `resume=true` of the full submission:

```bash
N_PERM=1 CHUNK_SIZE=2 MAX_CONCURRENT=1 \
AGGREGATE_SUBDIR=aggregate_glmmpql_smoke \
bash slurm/submit_permutation_glmmpql_1000.sh \
  "$PWD/results/permutation_1000"
```

Once the array and the aggregate job finish, check:

```bash
cat results/permutation_1000/aggregate_glmmpql_smoke/completeness_report.csv
cat results/permutation_1000/aggregate_glmmpql_smoke/glmmpql_fit_diagnostics.csv
```

There should be 4 compact results: 2 permutation IDs x 2 cohorts.

## 2. Full submission of 1,000 permutations

With the default `CHUNK_SIZE=2` this is 501 array elements plus 1 aggregate job, at most
200 running concurrently:

```bash
bash slurm/submit_permutation_glmmpql_1000.sh \
  "$PWD/results/permutation_1000"
```

If the account already holds jobs, subtract them explicitly:

```bash
EXISTING_SUBMITTED_TASKS=120 EXISTING_RUNNING_TASKS=30 \
bash slurm/submit_permutation_glmmpql_1000.sh \
  "$PWD/results/permutation_1000"
```

The script keeps the new array plus aggregate within the remaining 1,000-job submission
quota and caps the added concurrency at `200 - EXISTING_RUNNING_TASKS`.

Monitoring:

```bash
squeue -u "$USER"
find results/permutation_1000/results -path '*/glmmpql/perm_*.rds' | wc -l
cat results/permutation_1000/aggregate_glmmpql/completeness_report.csv
```

The final count should be 2,002 glmmPQL compact RDS files.

Rerun missing IDs:

```bash
bash slurm/resubmit_missing_permutations_glmmpql.sh \
  "$PWD/results/permutation_1000"
```

## 3. Bootstrap Stage 2 smoke test

This command reads only `bootstrap_001` for the two cohorts and does not run Stage 1:

```bash
N_BOOTSTRAP=1 MAX_CONCURRENT=2 \
AGGREGATE_SUBDIR=aggregate_glmmpql_smoke \
bash slurm/submit_participant_bootstrap_glmmpql_30.sh \
  "$PWD/results/participant_bootstrap_30"
```

Check:

```bash
cat results/participant_bootstrap_30/aggregate_glmmpql_smoke/stage2_aggregate_status.csv
cat results/participant_bootstrap_30/aggregate_glmmpql_smoke/glmmpql_fit_diagnostics.csv
```

## 4. Full submission of the 30 bootstrap glmmPQL replicates

60 array elements in total: 2 cohorts x 30 replicates. The default concurrency cap of 60
is below 200:

```bash
bash slurm/submit_participant_bootstrap_glmmpql_30.sh \
  "$PWD/results/participant_bootstrap_30"
```

Monitoring:

```bash
squeue -u "$USER"
find results/participant_bootstrap_30/stage2/results -path '*/glmmpql/bootstrap_*.rds' | wc -l
cat results/participant_bootstrap_30/aggregate_glmmpql/stage2_aggregate_status.csv
```

The final count should be 60 glmmPQL compact RDS files.

Rerun missing tasks:

```bash
bash slurm/resubmit_missing_participant_bootstrap_glmmpql.sh \
  "$PWD/results/participant_bootstrap_30"
```

## 5. Joint three-engine aggregation (optional)

Once glmmPQL has completed, a three-engine summary can be produced on a compute node
without overwriting the earlier results.

Permutation:

```bash
"$RSCRIPT" permutation_slurm/aggregate_permutations.R \
  --project-dir "$PWD" \
  --out-dir "$PWD/results/permutation_1000" \
  --n-perm 1000 --seed 20260715 \
  --cohorts ECAM,DIABIMMUNE \
  --methods linda,maaslin2,glmmpql \
  --aggregate-subdir aggregate_3methods
```

Bootstrap:

```bash
"$RSCRIPT" participant_resampling_slurm/aggregate_stage2.R \
  --output-root "$PWD/results/participant_bootstrap_30" \
  --n-bootstrap 30 \
  --methods linda,maaslin2,glmmpql \
  --aggregate-subdir aggregate_3methods
```

Do not run the full permutation job (up to 200 concurrent) and the bootstrap job (up to
60 concurrent) at the same time; submitting them sequentially keeps total concurrency
within 200.
