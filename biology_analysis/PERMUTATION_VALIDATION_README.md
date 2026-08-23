# ECAM / DIABIMMUNE small-scale permutation validation

Script: `permutation_validation_small.R`

## Default design

- The existing `base_raw` and `p2d2pca_imputed` tensors are held fixed; nothing is
  re-imputed.
- Labels are permuted at the participant level, preserving the original group sizes.
- The same label permutation is used for observed/completed and for LinDA/MaAsLin2.
- Default targets:
  - ECAM: `delivery_group` (Vaginal 24, Cesarean 18);
  - DIABIMMUNE: `abx_group` (unexposed 19, exposed 21).
- `permutation_id = 0` is the true-label baseline; `1...B` are null permutations.
- The default `B=5` is only for checking the workflow and is not used for any conclusion
  in the manuscript.

## Recommended strategy from the R console

Set the working directory to the project root first:

```r
setwd("/path/to/penalized_2d2_pca/biology_analysis")
source("permutation_validation_small.R")
print_small_permutation_strategy()
```

Confirm that the R console is using the same package environment as the earlier DA
analyses:

```r
.libPaths()
sapply(c("ncdf4", "MicrobiomeStat", "Maaslin2"), requireNamespace, quietly = TRUE)
preflight_packages(c("linda", "maaslin2"))
```

If LinDA is installed under the name `LinDA` or `linda` rather than `MicrobiomeStat`,
`preflight_packages()` recognizes that automatically.

### Step 1: test ECAM + LinDA only

```r
pilot1 <- run_small_permutation_validation(
  n_perm = 2,
  cohorts = "ECAM",
  methods = "linda",
  cores = 1,
  seed = 20260715,
  out_dir = "artifacts/permutation_validation_smoke"
)

pilot1$run_manifest
pilot1$run_summary
```

Proceed only once every `run_manifest$status` is `completed`. The script does not retry
failed tasks automatically; error details are saved in each permutation's `logs/`
directory.

### Step 2: small pilot over two cohorts x two DA engines

```r
pilot2 <- run_small_permutation_validation(
  n_perm = 5,
  cohorts = c("ECAM", "DIABIMMUNE"),
  methods = c("linda", "maaslin2"),
  cores = 1,
  seed = 20260715,
  out_dir = "artifacts/permutation_validation_small"
)

pilot2$run_manifest
pilot2$null_calibration
pilot2$paired_inflation
```

By default the script passes the current R console's `.libPaths()` to the child
processes. To override it explicitly:

```r
lib_path = paste(.libPaths(), collapse = ",")
```

### Step 3: scale up to a 20-100 permutation technical pilot

Confirm the following before scaling up:

1. the group sizes in every label file match the original data exactly;
2. observed and completed always use the same `permutation_seed`;
3. neither engine fails systematically;
4. every successful task returns 480 ECAM taxa or 839 DIABIMMUNE taxa;
5. `resume=TRUE` continues from existing results without rerunning completed tasks.

```r
pilot100 <- run_small_permutation_validation(
  n_perm = 100,
  cohorts = c("ECAM", "DIABIMMUNE"),
  methods = c("linda", "maaslin2"),
  cores = 2,
  seed = 20260715,
  out_dir = "artifacts/permutation_validation_100",
  resume = TRUE
)
```

## Saved output

The output root contains:

- `run_config.csv`: configuration of this run;
- `run_manifest.csv`: exit status, elapsed time, log and result path for each
  cohort/permutation/method;
- `label_permutation_manifest.csv`: seeds, group sizes and label-file paths;
- `permutation_taxon_results.csv`: per-taxon effect, p and q for every successful task;
- `permutation_run_summary.csv`: nominal type-I error and BH false-discovery count per
  permutation;
- `paired_inflation_completed_vs_observed.csv`: paired `Delta` and `RR` of completed
  relative to observed;
- `null_calibration_summary.csv`: cross-permutation means, intervals, `Pr(any q<alpha)`
  and mean false-discovery count;
- `empirical_pilot.csv`: workflow check table of taxon-specific empirical p/q.

Each `cohort/perm_XXXX/` directory additionally keeps:

- the participant label table actually used;
- the raw LinDA and MaAsLin2 output;
- a separate log.

## Interpretation limits

`B=5/20/100/1000` can be used to check the workflow and to assess overall
false-positive inflation, but the `empirical_pilot.csv` produced by a small run must not
be treated as a final significance result. Under `(1 + exceedances)/(B + 1)`, the 480
ECAM taxa and 839 DIABIMMUNE taxa demand far more than 1,000 permutations for adequate
BH resolution. The permutation count and storage scheme for production empirical p/q
values should be decided separately, after the small pilot passes.
