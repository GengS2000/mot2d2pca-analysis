# Semi-synthetic longitudinal profile zero-recovery experiments

This project evaluates `penalized_2d2_pca` on real longitudinal microbiome profiles using semi-synthetic masking/downsampling. The input table `hmp2_mgx_taxonomy.tsv` is treated as a MetaPhlAn-style relative-abundance/profile table, not a raw count table. All reports and variables therefore use `relative_abundance`, `profile`, or `log10(profile + eps)` wording.

## Inputs

- `hmp2_mgx_taxonomy.tsv`: taxa-by-sample relative-abundance/profile table. Sample columns may end in `_profile`.
- `metagenomics.xlsx`: metadata with `External ID`, `Participant ID`, `data_type`, and `week_num`.
- Only rows with `data_type == "metagenomics"` are used.
- Week bins are `0`, `1-4`, `5-8`, ..., `33-36`.
- Duplicate participant-bin samples are averaged.
- Missing participant-bin grid cells are filled with profile value `0`, then transformed to `log10(profile + eps)`.

## Code Layout

- `src/preprocessing.py`: profile/metadata alignment, taxon filtering, tensor construction, preprocessing diagnostics.
- `src/penalized_2d2_pca.py`: mask-aware penalized 2D2-PCA core.
- `src/imputation_wrappers.py`: unified imputation dispatcher and baselines.
- `src/masking.py`: semi-synthetic masking mechanisms and zero detectors.
- `src/metrics.py`: numeric, distribution, trajectory, and correlation/network metrics.
- `src/plotting.py`: matplotlib-only PNG plotting helpers.
- `src/experiment_runner.py`: shared CLI/config utilities.
- `scripts/run_experiment1_zero_recovery.py`: zero recovery benchmark.
- `scripts/run_experiment2_distribution_preservation.py`: nonzero distribution preservation.
- `scripts/run_experiment3_trajectory_preservation.py`: longitudinal trajectory preservation.
- `scripts/run_experiment4_dynamic_correlation.py`: static and lagged taxon-taxon correlation/network recovery.

## Scientific Questions

Experiment 1 asks whether artificial sampling zeros inserted into real nonzero profile entries can be recovered numerically under different masking mechanisms.

Experiment 2 asks whether imputation preserves taxon-specific nonzero profile distributions rather than simply replacing zeros with means or creating artificial spikes.

Experiment 3 asks whether imputation preserves participant-taxon longitudinal trajectories, including lag-1 autocorrelation and second-difference smoothness.

Experiment 4 asks whether imputation improves static and lagged taxon-taxon correlation/network structure, extending static correlation checks toward longitudinal microbial dynamics.

## Main Metrics

- `mae`, `rmse`, `median_absolute_error`: absolute numeric recovery error on artificial masked nonzero entries.
- `pearson`, `spearman`: association between imputed and original true log-profile values.
- `wasserstein`: distributional distance between imputed and original values.
- `bias`: mean imputed minus true value.
- `zero_detection_sensitivity`, `precision`, `specificity`, `f1`: end-to-end detector quality. Oracle mode does not interpret these as detector performance.
- `trajectory_rmse`: original vs imputed participant-taxon trajectory error.
- `lag1_error_imputed`: lag-1 autocorrelation recovery error.
- `second_difference_mae_imputed`: trajectory curvature/smoothness recovery error.
- `js_full`: taxon-wise Jensen-Shannon divergence between original and imputed distributions.
- `lagged_pearson`, `lagged_spearman`, `lagged_frobenius`: lagged correlation matrix recovery.
- `topk_precision`, `jaccard`, `hub_jaccard`: dynamic network edge and hub recovery.

## Interpretation

If `penalized_2d2_pca` improves RMSE, Pearson/Spearman, trajectory RMSE, and lagged correlation recovery over baselines, it supports the method as a better longitudinal microbiome zero-recovery model.

If Wasserstein distance and JS divergence do not increase relative to baselines, it suggests the method is not obviously damaging nonzero profile distributions.

If second-difference error decreases, it suggests the method restores smoother longitudinal trajectory structure.

If lagged network overlap improves, it suggests the method helps recover dynamic taxon-taxon relationships.

## Run Commands

Run from the project root. If your system Python is missing dependencies, use a virtual environment built from the requirements listed above; otherwise `python` is fine.

### Minimal Smoke Test

These commands use a small participant/taxon subset, one repeat, two replacement modes, and low PCA iterations.

```powershell
python .\scripts\run_experiment1_zero_recovery.py --outdir outputs\smoke --max-participants 8 --max-taxa 12 --min-pos-cells 2 --min-participants 2 --fracs 0.10 --repeats 1 --mechanisms random_entry,low_abundance_biased --replacement-modes oracle_mask,detector_longitudinal_neighbor --methods penalized_2d2_pca,mean,no_imputation --rank-d 2 --rank-q 2 --T0 1 --T 1 --overwrite
python .\scripts\run_experiment2_distribution_preservation.py --outdir outputs\smoke --max-participants 8 --max-taxa 12 --min-pos-cells 2 --min-participants 2 --fracs 0.10 --repeats 1 --mechanisms random_entry --replacement-modes oracle_mask,detector_longitudinal_neighbor --methods penalized_2d2_pca,mean,no_imputation --rank-d 2 --rank-q 2 --T0 1 --T 1 --overwrite
python .\scripts\run_experiment3_trajectory_preservation.py --outdir outputs\smoke --max-participants 8 --max-taxa 12 --min-pos-cells 2 --min-participants 2 --fracs 0.10 --repeats 1 --mechanisms random_entry --replacement-modes oracle_mask,detector_longitudinal_neighbor --methods penalized_2d2_pca,mean,no_imputation --rank-d 2 --rank-q 2 --T0 1 --T 1 --overwrite
python .\scripts\run_experiment4_dynamic_correlation.py --outdir outputs\smoke --max-participants 8 --max-taxa 12 --min-pos-cells 2 --min-participants 2 --fracs 0.10 --repeats 1 --mechanisms random_entry --replacement-modes oracle_mask,detector_longitudinal_neighbor --methods penalized_2d2_pca,mean,no_imputation --rank-d 2 --rank-q 2 --T0 1 --T 1 --top-variable-taxa 8 --overwrite
```

### Full Experiment Commands

```powershell
python .\scripts\run_experiment1_zero_recovery.py --tax-path "hmp2_mgx_taxonomy.tsv" --meta-path "metagenomics.xlsx" --outdir outputs --eps 1e-3 --min-pos-cells 10 --min-participants 5 --rank-d 10 --rank-q 10 --lam 0.7 --sigma 8 --T0 30 --T 30 --fracs 0.05,0.10,0.20,0.30 --repeats 10 --mechanisms random_entry,low_depth_sample,time_point,subject_specific_dropout,consecutive_time_window_dropout,low_abundance_biased --replacement-modes oracle_mask,detector_all_zero,detector_longitudinal_neighbor --methods penalized_2d2_pca,mean,median,last,weighted_time_average,temporal_knn,spline,no_imputation --overwrite
python .\scripts\run_experiment2_distribution_preservation.py --tax-path "hmp2_mgx_taxonomy.tsv" --meta-path "metagenomics.xlsx" --outdir outputs --eps 1e-3 --min-pos-cells 10 --min-participants 5 --rank-d 10 --rank-q 10 --lam 0.7 --sigma 8 --T0 30 --T 30 --repeats 10 --overwrite
python .\scripts\run_experiment3_trajectory_preservation.py --tax-path "hmp2_mgx_taxonomy.tsv" --meta-path "metagenomics.xlsx" --outdir outputs --eps 1e-3 --min-pos-cells 10 --min-participants 5 --rank-d 10 --rank-q 10 --lam 0.7 --sigma 8 --T0 30 --T 30 --repeats 10 --overwrite
python .\scripts\run_experiment4_dynamic_correlation.py --tax-path "hmp2_mgx_taxonomy.tsv" --meta-path "metagenomics.xlsx" --outdir outputs --eps 1e-3 --min-pos-cells 10 --min-participants 5 --rank-d 10 --rank-q 10 --lam 0.7 --sigma 8 --T0 30 --T 30 --repeats 10 --top-variable-taxa 40 --overwrite
```

## Outputs

- `outputs/preprocessing_summary.json`
- `outputs/preprocessing_summary.csv`
- `outputs/experiment1_zero_recovery/metrics_long.csv`
- `outputs/experiment1_zero_recovery/metrics_summary.csv`
- `outputs/experiment2_distribution/distribution_metrics_long.csv`
- `outputs/experiment2_distribution/distribution_metrics_summary.csv`
- `outputs/experiment2_distribution/raw_nonzero_da_log_distribution_summary.csv`
- `outputs/experiment2_distribution/raw_nonzero_da_log_distribution_quantiles.csv`
- `outputs/experiment2_distribution/raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv`
- `outputs/experiment2_distribution/raw_nonzero_vs_imputed_at_masked_distribution_summary.csv`
- `outputs/experiment2_distribution/raw_nonzero_vs_imputed_at_masked_distribution_quantiles.csv`
- `outputs/experiment3_trajectory/trajectory_metrics_long.csv`
- `outputs/experiment3_trajectory/trajectory_metrics_summary.csv`
- `outputs/experiment3_trajectory/subject_level_metrics.csv`
- `outputs/experiment3_trajectory/taxon_clustering_metrics.csv`
- `outputs/experiment4_dynamic_correlation/static_correlation_metrics.csv`
- `outputs/experiment4_dynamic_correlation/lagged_correlation_metrics.csv`
- `outputs/experiment4_dynamic_correlation/network_edge_overlap.csv`
- `outputs/experiment4_dynamic_correlation/hub_taxa_overlap.csv`
- Each experiment also writes `config.json` and `failed_runs.csv`.
- Experiment 2 defaults to CSV-only output for batch/cluster runs; add `--save-figures` to write optional PNG figures under its `figures/` directory when matplotlib is available.
- Other experiments write PNG figures under each experiment's `figures/` directory when matplotlib is available.
