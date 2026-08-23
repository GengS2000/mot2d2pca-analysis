# MOT-(2D)² PCA — analysis code

Source code for the figures, tables, and statistics reported in the MOT-(2D)² PCA
manuscript: a mask-aware, temporally regularized method for reconstructing sparse
longitudinal microbiome tensors.

The method itself is packaged and released separately:
**https://github.com/GengS2000/mot2d2pca**

This repository holds the simulation studies, real-data analyses, cluster job
scripts, and figure/table generators. Input data and computed results are not
included (see [Data](#data)).

---

## Figures and main-text tables

| Item | Analysis code | Figure/table code |
| --- | --- | --- |
| **Fig. 1** — workflow | — | artwork only |
| **Fig. 2A–D** — Simulation 1/2/3 reconstruction | `sim1_compare/`, `sim2-compare/`, `sim3_compare/`, `competitor_methods.py`, `competitor_task.py`, `merge_competitor_results.py`, root `*.slurm` and `submit_*.sh` | `figures/fig2_sim123/plot_figure1_sim123_abcd.py` |
| **Fig. 3A–G** — parameter sensitivity and ablation | `robustness_suite/` (stages A/B/C), `parameter/` | `robustness_suite/scripts/plot_figure3_parameter_sensitivity_portrait.py`, `prepare_figure3_data.py` |
| **Fig. 4A–E** — differential-abundance simulation | `downstream/DA_simulation/`, `downstream/DA_simulation/TUNE/` | `downstream/DA_simulation/TUNE/simulation3_effect_tune/revised_main_figures/fig4/make_figure4_revised.py` |
| **Fig. 5A–I** — FARMM / TEMPTED | `downstream/dimensionality_reduction/FARMM_data/run_realdata_tempted_p2d2pca_comparison.py` | `downstream/dimensionality_reduction/FARMM_data/mainresults/make_figure.py` |
| **Fig. 6A–E** — HMP2 non-zero recovery | `non-zero_recover/scripts/`, `non-zero_recover/src/` | `non-zero_recover_new/figures_final/make_figures.py` |
| **Fig. 7A–E** — DIABIMMUNE / ECAM inference | `biology_analysis/DIABIMMUNE_data/`, `biology_analysis/ECAM_data/`, `biology_analysis/permutation_slurm/`, `biology_analysis/participant_resampling_slurm/`, `biology_analysis/glmmpql_longitudinal_helpers.R` | `downstream/DA_simulation/TUNE/simulation3_effect_tune/revised_main_figures/fig7/make_figure7_revised.py` |
| **Table 1, Table 2** — HMP2 activity resolution | `downstream/dimensionality_reduction/run_realdata_pca_p2d2pca_comparison.py` (builds the completed tensors) | `biology_analysis/hmp2_activity_resolution/analyze_activity_resolution.py` |

## Supplementary figures

| Item | Code | Name emitted by the script |
| --- | --- | --- |
| **S1A/B/C** | `robustness_suite/scripts/analyze_unified_results.py`, `plot_parameter_sensitivity_ablation.py` | `fig_a_parameter_selection`, `fig_b_parameter_sensitivity`, `fig_c_ablation_forest` |
| **S2, S3, S4** | `downstream/DA_simulation/TUNE/simulation3_effect_tune/make_publication_DA_imputation_figures_additive_linda.R` | `supplementary_full_metric_curves`, `supplementary_delta_heatmaps`, `supplementary_confusion_counts` |
| **S5** | `.../revised_main_figures/fig4/make_supp_grid_improvement_figure.py` | `Figure_S5_grid_cell_improvement_fractions` |
| **S6** | `non-zero_recover_new/figures_final/make_figures.py` | `FigureS1_inference_unit` |
| **S7** | same script | `FigureS3_mechanism_panorama` |
| **S8** | same script | `FigureS2_metric_forest` |
| **S9** | `supplementary/figures/make_figure_s9_antibiotic.py` | `Figure_S9_antibiotic_exposure_null` |

> **Note on naming.** `make_figures.py` labels its outputs S1/S2/S3 internally, from an
> earlier numbering. In the published numbering, internal **S2 → S8** and internal
> **S3 → S7** — the two are swapped relative to what the filenames suggest.

## Supplementary tables

| Item | Code |
| --- | --- |
| S1a/S1b, S2a/S2b | `figures/fig2_sim123/make_supplementary_tables_sim123.py` |
| S3, S4, S5 | `robustness_suite/scripts/aggregate_stage.py`, `aggregate_stage_a.py`, `analyze_unified_results.py` |
| S6–S12, Data S1 | `downstream/dimensionality_reduction/FARMM_data/` (run script and `mainresults/make_figure.py`) |
| S13, S14, S15 | `non-zero_recover/scripts/analyze_repeat_level_inference.py`, `recompute_experiment3_repeat_level_stats.py` |
| S16–S25 | `biology_analysis/permutation_slurm/`, `biology_analysis/participant_resampling_slurm/`, `biology_analysis/realdata_glmmPQL_ZIGMM_runner.R`, `biology_analysis/prepare_supplementary_data_glmmpql.py` |
| S26 | `biology_analysis/hmp2_activity_resolution/analyze_activity_resolution.py` |

---

## The shared core library

`sim3_compare.py` (~123 KB) holds the shared implementation — `penalized_2d2_pca()`,
`ctf_complete_tensor_from_log_da()`, `cp_als_observed_only()`, `build_base_tensor()`,
`rclr_transform_tensor()`. It is vendored into each analysis directory rather than
imported from a single location, so several copies are present:

| Copy | Difference from the reference version |
| --- | --- |
| `non-zero_recover/`, `downstream/DA_simulation/TUNE/` | reference version |
| `biology_analysis/ECAM_data/`, `downstream/DA_simulation/`, `downstream/dimensionality_reduction/` | netCDF backend selection only (`scipy` vs `netcdf4`) |
| `downstream/dimensionality_reduction/FARMM_data/` | adds `PS` (pre-study) time-bin handling, required by the FARMM design: study days 0–15 plus the pre-study sample |

`penalized_2d2_pca()` is byte-identical across all copies; the differences are confined
to file I/O and to the FARMM-specific time-bin parser.

Despite the name, `sim3_compare/sim3_compare.py` (169 KB) is a different file: the
Simulation-3 benchmark driver, not the shared library. A standalone extraction of the
method also lives at `non-zero_recover/src/penalized_2d2_pca.py`.

---

## Practical notes

1. **Absolute paths.** The figure scripts open `D:\penalized_2d2\penalized_2d2_pca\...`
   directly (for example `make_figure.py:22`, `make_figure7_revised.py:23`,
   `make_figure4_revised.py:48`). Edit these before running elsewhere.

2. **`non-zero_recover/xarray.py`** is a 99-line minimal `DataArray` fallback. Because it
   sits at the package root it shadows the real `xarray` package whenever that directory
   is on `sys.path`, and `sim3_compare.py` does `import xarray as xr`. Which
   implementation is used therefore depends on the working directory.

3. **Comparator scope.** The reported comparators are CTF, Mean, Nearest Positive
   Observation, and Temporal-KNN; `competitor_methods.py` implements CTF and the shared
   mask invariants, and the three trajectory fills live in
   `sim1_compare/sim1_methodcompare.py`. See `COMPETITOR_BASELINES_README.md` for the
   methods that were considered and not benchmarked. Likewise, the reported
   differential-abundance engines are LinDA, MaAsLin2, ZIGMM, and glmmPQL.

4. **Figure S9.** `biology_analysis/figure_main_infant_gut/supp_make_figures.py` draws panels
   S1–S7 itself and delegates S9 to `supplementary/figures/make_figure_s9_antibiotic.py`, which
   reads `supplementary/tables/source_csv/Figure_S9_antibiotic_three_engine_source.csv`. That
   source table ships with the supplementary package and is not tracked here; point `SOURCE` at
   it before running.

5. **Tuning provenance for Tables 1–2.** The HMP2 tensors used for the activity-resolution
   analysis came from a label-aware exploratory tuning run; see
   `downstream/dimensionality_reduction/output_full_tuning/tuning_design_notes.md` and
   `final_d16q32_eps_T30_selected_results/final_tuning_report.md`. That result is
   reconstruction-conditional and is reported as exploratory.

---

## Data

Input data are not included. The analyses require:

- **Simulations 1–3** — pseudo-truth tensors built by the `*_truth_*.slurm` scripts.
- **FARMM** — `kraken_results.tsv`, `20200619_farmm_metadata.tsv`.
- **DIABIMMUNE, ECAM** — 16S OTU tables and metadata from the source cohorts.
- **HMP2 / IBDMDB** — `hmp2_mgx_taxonomy.tsv`, `metagenomics.xlsx`.

Python dependencies for the comparator benchmark: `requirements_competitors.txt`.
R dependencies for the differential-abundance engines:
`downstream/DA_simulation/TUNE/install_r_deps.R`.
