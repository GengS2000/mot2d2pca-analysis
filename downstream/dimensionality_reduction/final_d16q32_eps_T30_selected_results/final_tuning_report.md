# Parameter-tuning run: impute+PCA vs direct PCA

## Frozen parameters
- impute_mode: eps_only
- d: 16
- q: 32
- lam: 0.0
- sigma: 2.0
- T0: 30
- T: 30
- pca_n_components: 5
- permanova_n_pcs: 2
- permanova_permutations: 999
- classification_n_repeats_manual: 20
- test_size: 0.2
- random_state: 2026
- thread_note: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 was required to avoid PCA BLAS hang in this environment.

## Matched-subject PERMANOVA
| contrast      |   n_subjects | groups                     |   direct_F |   imputed_F |   delta_F |   direct_p |   imputed_p |
|:--------------|-------------:|:---------------------------|-----------:|------------:|----------:|-----------:|------------:|
| ibd_vs_nonibd |          130 | {"IBD": 103, "nonIBD": 27} |   10.1011  |   11.8147   |  1.71364  |      0.001 |       0.001 |
| cd_vs_uc      |          103 | {"CD": 65, "UC": 38}       |    1.26808 |    0.914381 | -0.353696 |      0.308 |       0.405 |

## IBD vs nonIBD classification metrics: mean over 20 StratifiedShuffleSplit repeats
| classifier          | metric            |   direct_pca |   p2d2pca_imputed_then_pca |   delta_imputed_minus_direct |
|:--------------------|:------------------|-------------:|---------------------------:|-----------------------------:|
| logistic_regression | AUPRC             |     0.908155 |                   0.914887 |                   0.00673269 |
| logistic_regression | AUROC             |     0.729524 |                   0.732381 |                   0.00285714 |
| logistic_regression | F1                |     0.773561 |                   0.797082 |                   0.0235218  |
| logistic_regression | accuracy          |     0.676923 |                   0.709615 |                   0.0326923  |
| logistic_regression | balanced_accuracy |     0.651429 |                   0.694524 |                   0.0430952  |
| random_forest       | AUPRC             |     0.918758 |                   0.941278 |                   0.0225196  |
| random_forest       | AUROC             |     0.72881  |                   0.787381 |                   0.0585714  |
| random_forest       | F1                |     0.888161 |                   0.892181 |                   0.00402068 |
| random_forest       | accuracy          |     0.809615 |                   0.817308 |                   0.00769231 |
| random_forest       | balanced_accuracy |     0.592619 |                   0.612619 |                   0.02       |

## IBD vs nonIBD PC loading test summary
| pipeline                 | component   |   n_group0 |   n_group1 |   effect_size_group1_minus_group0 |   mannwhitney_p |   mannwhitney_fdr_bh |   welch_t_p |   welch_t_fdr_bh |
|:-------------------------|:------------|-----------:|-----------:|----------------------------------:|----------------:|---------------------:|------------:|-----------------:|
| p2d2pca_imputed_then_pca | PC1         |         27 |        103 |                          0.724377 |     0.000407201 |          0.001018    | 0.000288807 |       0.00144403 |
| p2d2pca_imputed_then_pca | PC2         |         27 |        103 |                          0.760983 |     0.000155473 |          0.000777367 | 0.00471435  |       0.0117859  |

## Conclusions
- On matched IBD vs nonIBD, the PERMANOVA pseudo-F of impute+PCA exceeds that of direct PCA.
- For classification, the mean AUROC, balanced accuracy, accuracy, F1 and AUPRC of both logistic regression and random forest exceed those of direct PCA.
- CD vs UC is not the target contrast of this run; under the frozen parameters its PERMANOVA pseudo-F is lower than direct PCA.
