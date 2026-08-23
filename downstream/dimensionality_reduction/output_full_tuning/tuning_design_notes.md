# Tuning design notes

This tuning is label-aware and exploratory.
It optimizes IBD vs nonIBD separation.
It should not be reported as unbiased predictive validation unless validated in nested split or external cohort.
CD vs UC is not used as the tuning objective.

## Score definition (tune_score_mode=ibd_nonibd_label_aware)
score = 2.0*delta_logistic_AUROC + 1.5*delta_logistic_balanced_accuracy + 1.0*delta_rf_AUROC + 0.5*delta_permanova_F
All deltas are imputed minus direct on the SAME IBD-vs-nonIBD matched subject set.
Failed/too-few-subject combos receive score = -1e9 (excluded from the top).
A small -0.5 penalty applies when fewer than 6 matched subjects remain.
PERMANOVA p-values are reported but NOT used for ranking (permutation/sample-size sensitive).

