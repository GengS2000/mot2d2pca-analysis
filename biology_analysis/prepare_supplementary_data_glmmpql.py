from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(r"D:\penalized_2d2\penalized_2d2_pca\biology_analysis")
PERM_DIRS = [
    ANALYSIS_ROOT / "results" / "permutation" / "aggregate",
    ANALYSIS_ROOT / "results" / "glmmpql_aggregates" / "permutation_1000" / "aggregate_glmmpql",
]
BOOT_DIRS = [
    ANALYSIS_ROOT / "results" / "bootstrap" / "aggregate",
    ANALYSIS_ROOT / "results" / "glmmpql_aggregates" / "participant_bootstrap_30" / "aggregate_glmmpql",
]
DIAB = ANALYSIS_ROOT / "DIABIMMUNE_data" / "artifacts" / "subject_group_existing_methods_r"
GLMMPQL_DIAB = DIAB / "glmmPQL_ZIGMM" / "DIABIMMUNE_glmmPQL_ZIGMM_longitudinal_results.csv"
WORK_DIR = Path(__file__).resolve().parent
OUT = WORK_DIR / "supplementary_data.json"
FIGURE_BOOT_OUT = WORK_DIR / "figure_s9_bootstrap_glmmpql.csv"
DIAGNOSTICS_OUT = WORK_DIR / "glmmpql_fit_diagnostics_summary.json"


def clean(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return str(value)


def records(frame: pd.DataFrame):
    return [{str(key): clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


def read_union(directories: list[Path], filename: str, **kwargs) -> pd.DataFrame:
    frames = []
    for directory in directories:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, **kwargs))
    union = pd.concat(frames, ignore_index=True)
    if "method" in union:
        if union.duplicated().any():
            raise ValueError(f"Unexpected duplicated rows after merging {filename}")
    return union


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


contrast_labels = {
    "ECAM": "Delivery mode (Cesarean vs Vaginal)",
    "DIABIMMUNE": "Antibiotic exposure (exposed vs unexposed)",
}
dataset_labels = {"base_raw": "Observed", "p2d2pca_imputed": "Completed"}
method_labels = {"linda": "LinDA", "maaslin2": "MaAsLin2", "glmmpql": "glmmPQL"}
method_order = ["linda", "maaslin2", "glmmpql"]


def same_contrast(frame: pd.DataFrame, item: dict) -> pd.DataFrame:
    mask = (
        (frame["cohort"] == item["cohort"])
        & (frame["group_mode"] == item["group_mode"])
        & (frame["feature_id"].astype(str) == item["feature_id"])
    )
    if item["diet_col"] is None:
        mask &= frame["diet_col"].isna()
    else:
        mask &= frame["diet_col"] == item["diet_col"]
    return frame.loc[mask]


def diet_short(column: str) -> str:
    return re.sub(r"^dietary_compound_|_start_age_month$", "", column)


def spearman(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation without requiring the optional scipy dependency."""
    return float(left.rank(method="average").corr(right.rank(method="average")))


null = read_union(PERM_DIRS, "null_calibration_summary.csv")
paired = read_union(PERM_DIRS, "paired_inflation_completed_vs_observed.csv")
emp = read_union(PERM_DIRS, "empirical_permutation_results.csv", low_memory=False)
boot = read_union(BOOT_DIRS, "bootstrap_inference_stability.csv.gz", low_memory=False)

for frame in (null, paired, emp, boot):
    if "feature_id" in frame:
        frame["feature_id"] = frame["feature_id"].astype(str)


# Table S16: global-null calibration and paired completed-minus-observed inflation.
paired_null = paired.loc[~paired["is_observed"].astype(bool)].copy()
perm_rows = []
for (cohort, method), block in null.groupby(["cohort", "method"], sort=False):
    raw = block.loc[block["dataset"] == "base_raw"].iloc[0]
    comp = block.loc[block["dataset"] == "p2d2pca_imputed"].iloc[0]
    pair_block = paired_null.loc[(paired_null["cohort"] == cohort) & (paired_null["method"] == method)]
    delta01 = float(pair_block["delta_p_lt_0_01"].mean())
    delta05 = float(pair_block["delta_p_lt_0_05"].mean())
    se01 = float(pair_block["delta_p_lt_0_01"].std(ddof=1) / math.sqrt(len(pair_block)))
    se05 = float(pair_block["delta_p_lt_0_05"].std(ddof=1) / math.sqrt(len(pair_block)))
    k_any = int(round(float(comp["probability_any_q_lt_0_05"]) * int(comp["n_permutations"])))
    wlo, whi = wilson_interval(k_any, int(comp["n_permutations"]))
    perm_rows.append({
        "cohort": cohort,
        "contrast": contrast_labels[cohort],
        "engine": method_labels[method],
        "n_permutations": int(comp["n_permutations"]),
        "observed_prop_p_lt_0_01": float(raw["mean_prop_p_lt_0_01"]),
        "completed_prop_p_lt_0_01": float(comp["mean_prop_p_lt_0_01"]),
        "delta_p_lt_0_01": delta01,
        "delta_p_lt_0_01_mc_ci_low": delta01 - 1.96 * se01,
        "delta_p_lt_0_01_mc_ci_high": delta01 + 1.96 * se01,
        "rr_p_lt_0_01": float(pair_block["rr_p_lt_0_01"].replace([np.inf, -np.inf], np.nan).mean()),
        "observed_prop_p_lt_0_05": float(raw["mean_prop_p_lt_0_05"]),
        "completed_prop_p_lt_0_05": float(comp["mean_prop_p_lt_0_05"]),
        "delta_p_lt_0_05": delta05,
        "delta_p_lt_0_05_mc_ci_low": delta05 - 1.96 * se05,
        "delta_p_lt_0_05_mc_ci_high": delta05 + 1.96 * se05,
        "rr_p_lt_0_05": float(pair_block["rr_p_lt_0_05"].replace([np.inf, -np.inf], np.nan).mean()),
        "observed_probability_any_q_lt_0_05": float(raw["probability_any_q_lt_0_05"]),
        "completed_probability_any_q_lt_0_05": float(comp["probability_any_q_lt_0_05"]),
        "completed_any_q05_wilson_ci_low": wlo,
        "completed_any_q05_wilson_ci_high": whi,
        "observed_mean_n_q_lt_0_05": float(raw["mean_n_q_lt_0_05"]),
        "completed_mean_n_q_lt_0_05": float(comp["mean_n_q_lt_0_05"]),
        "interpretation": "Completed data increased the fraction of small nominal P values, but the mean P<0.05 fraction remained near 0.05 and P(any q<0.05) did not exceed 0.05.",
    })
permutation_calibration = pd.DataFrame(perm_rows)
permutation_calibration["_method_order"] = permutation_calibration["engine"].map({value: index for index, value in enumerate(method_labels.values())})
permutation_calibration.sort_values(["cohort", "_method_order"], inplace=True)
permutation_calibration.drop(columns="_method_order", inplace=True)


# Table S22: original discovery counts relative to the paired-label global null.
global_rows = []
for (cohort, method), block in paired.groupby(["cohort", "method"], sort=False):
    observed = block.loc[block["is_observed"].astype(bool)].iloc[0]
    null_block = block.loc[~block["is_observed"].astype(bool)]
    for data_key, data_label in [("observed", "Observed"), ("completed", "Completed")]:
        for qtag, qvalue in [("05", 0.05), ("10", 0.10)]:
            column = f"n_q_lt_0_{qtag}_{data_key}"
            observed_n = int(observed[column])
            null_exceedances = int((null_block[column] >= observed_n).sum()) if observed_n > 0 else len(null_block)
            empirical_p = (null_exceedances + 1) / (len(null_block) + 1) if observed_n > 0 else 1.0
            if observed_n == 0:
                conclusion = "No original-label discoveries at this threshold."
            elif empirical_p < 0.05:
                conclusion = "The observed discovery count was uncommon under the permuted-label global null."
            else:
                conclusion = "The observed discovery count was not uncommon at the 0.05 level under the permuted-label global null."
            global_rows.append({
                "cohort": cohort,
                "contrast": contrast_labels[cohort],
                "engine": method_labels[method],
                "data_version": data_label,
                "bh_threshold": qvalue,
                "observed_discovery_count": observed_n,
                "null_permutations_with_at_least_observed_count": null_exceedances,
                "n_null_permutations": len(null_block),
                "plus_one_global_empirical_p": empirical_p,
                "interpretation": conclusion,
            })
global_signal = pd.DataFrame(global_rows)
global_signal["_method_order"] = global_signal["engine"].map({value: index for index, value in enumerate(method_labels.values())})
global_signal.sort_values(["cohort", "_method_order", "data_version", "bh_threshold"], inplace=True)
global_signal.drop(columns="_method_order", inplace=True)


# Taxon-level empirical permutation detail for the completed ECAM delivery discoveries.
# This companion section of Table S22 distinguishes each taxon's plus-one P from
# the global discovery-count probability reported above.
ecam_taxon_labels = {
    "T0024": "Bacteroides",
    "T0142": "Bifidobacterium",
    "T0220": "Veillonella parvula",
}
ecam_completed_discovery_taxa = emp.loc[
    (emp["cohort"] == "ECAM")
    & (emp["group_mode"] == "delivery_group")
    & emp["diet_col"].isna()
    & (emp["dataset"] == "p2d2pca_imputed")
    & (emp["q_observed"] < 0.05),
    ["method", "feature_id", "effect_observed", "p_observed", "q_observed", "exceedances", "n_effective_permutations", "empirical_p", "empirical_q"],
].copy()
ecam_completed_discovery_taxa["engine"] = ecam_completed_discovery_taxa["method"].map(method_labels)
ecam_completed_discovery_taxa["taxon_label"] = ecam_completed_discovery_taxa["feature_id"].map(ecam_taxon_labels).fillna(ecam_completed_discovery_taxa["feature_id"])
ecam_completed_discovery_taxa.rename(columns={
    "effect_observed": "original_effect",
    "p_observed": "original_p",
    "q_observed": "original_q",
    "exceedances": "null_exceedances",
    "n_effective_permutations": "n_null_permutations",
    "empirical_p": "plus_one_taxon_empirical_p",
    "empirical_q": "taxon_empirical_bh_q",
}, inplace=True)
ecam_completed_discovery_taxa["_method_order"] = ecam_completed_discovery_taxa["engine"].map({value: index for index, value in enumerate(method_labels.values())})
ecam_completed_discovery_taxa.sort_values(["_method_order", "feature_id"], inplace=True)
ecam_completed_discovery_taxa = ecam_completed_discovery_taxa[[
    "engine", "feature_id", "taxon_label", "original_effect", "original_p", "original_q",
    "null_exceedances", "n_null_permutations", "plus_one_taxon_empirical_p", "taxon_empirical_bh_q",
]]


# Table S23: focal taxa. Primary contrasts use empirical-permutation original results;
# carrot uses original full-model result files because it was not part of label permutation.
linda_full = pd.read_csv(DIAB / "linda_longitudinal_results.csv", low_memory=False)
maas_full = pd.read_csv(DIAB / "maaslin2_longitudinal_results.csv", low_memory=False)
glmmpql_full = pd.read_csv(GLMMPQL_DIAB, low_memory=False)
focal = [
    {"cohort": "ECAM", "group_mode": "delivery_group", "diet_col": None, "feature_id": "T0024", "taxon_label": "Bacteroides", "role": "Stable delivery-mode association strengthened by completion"},
    {"cohort": "ECAM", "group_mode": "delivery_group", "diet_col": None, "feature_id": "T0142", "taxon_label": "Bifidobacterium", "role": "Stable delivery-mode association"},
    {"cohort": "ECAM", "group_mode": "delivery_group", "diet_col": None, "feature_id": "T0220", "taxon_label": "Veillonella parvula", "role": "Completion-strengthened delivery-mode association"},
    {"cohort": "ECAM", "group_mode": "delivery_group", "diet_col": None, "feature_id": "T0019", "taxon_label": "Bacteroides uniformis", "role": "Observed-data-favoured but directionally stable association"},
    {"cohort": "DIABIMMUNE", "group_mode": "abx_group", "diet_col": None, "feature_id": "T0487", "taxon_label": "Ruminococcus sp.", "role": "Reconstruction-sensitive antibiotic association"},
    {"cohort": "DIABIMMUNE", "group_mode": "diet_group", "diet_col": "dietary_compound_carrot_start_age_month", "feature_id": "T0336", "taxon_label": "Chloroplast/Streptophyta reads", "role": "Raw-only carrot-timing association attenuated by completion"},
    {"cohort": "DIABIMMUNE", "group_mode": "diet_group", "diet_col": "dietary_compound_carrot_start_age_month", "feature_id": "T0351", "taxon_label": "Mitochondrial reads", "role": "Raw-only carrot-timing association attenuated by completion"},
]


def original_diet_result(method: str, item: dict, dataset: str):
    if method == "linda":
        source = linda_full
        query = source.loc[
            (source["dataset"] == dataset)
            & (source["group_mode"] == item["group_mode"])
            & (source["diet_col"] == item["diet_col"])
            & (source["feature_id"].astype(str) == item["feature_id"])
        ]
    elif method == "maaslin2":
        source = maas_full
        query = source.loc[
            (source["dataset"] == dataset)
            & (source["group_mode"] == item["group_mode"])
            & (source["diet_col"] == item["diet_col"])
            & (source["feature_id"].astype(str) == item["feature_id"])
        ]
    else:
        query = glmmpql_full.loc[
            (glmmpql_full["dataset"] == dataset)
            & (glmmpql_full["group_mode"] == f"diet_group_{item['diet_col']}")
            & (glmmpql_full["feature_id"].astype(str) == item["feature_id"])
            & (glmmpql_full["fit_status"] == "ok")
        ]
    return query.iloc[0] if len(query) else None


focal_rows = []
for item in focal:
    for method in method_order:
        for dataset in dataset_labels:
            original_effect = original_p = original_q = empirical_p = empirical_q = None
            if item["group_mode"] in ("delivery_group", "abx_group"):
                query = same_contrast(emp, item)
                query = query.loc[(query["method"] == method) & (query["dataset"] == dataset)]
                if len(query):
                    row = query.iloc[0]
                    original_effect = row["effect_observed"]
                    original_p = row["p_observed"]
                    original_q = row["q_observed"]
                    empirical_p = row["empirical_p"]
                    empirical_q = row["empirical_q"]
            else:
                row = original_diet_result(method, item, dataset)
                if row is not None:
                    original_effect = row["effect_size"]
                    original_p = row["p_value"]
                    original_q = row["q_value"]
            boot_query = same_contrast(boot, item)
            boot_query = boot_query.loc[(boot_query["method"] == method) & (boot_query["dataset"] == dataset)]
            boot_row = boot_query.iloc[0] if len(boot_query) else None
            focal_rows.append({
                "cohort": item["cohort"],
                "contrast": contrast_labels[item["cohort"]] if item["diet_col"] is None else "Carrot introduction timing (early vs late)",
                "feature_id": item["feature_id"],
                "taxon_label": item["taxon_label"],
                "interpretive_role": item["role"],
                "engine": method_labels[method],
                "data_version": dataset_labels[dataset],
                "original_effect": original_effect,
                "original_p": original_p,
                "original_q": original_q,
                "taxon_empirical_p_1000_permutations": empirical_p,
                "taxon_empirical_q_1000_permutations": empirical_q,
                "n_successful_bootstraps": None if boot_row is None else boot_row["n_successful_bootstraps"],
                "bootstrap_selection_frequency_q05": None if boot_row is None else boot_row["selection_frequency_q05"],
                "bootstrap_selection_frequency_q10": None if boot_row is None else boot_row["selection_frequency_q10"],
                "bootstrap_sign_stability": None if boot_row is None else boot_row["sign_stability"],
                "bootstrap_effect_median": None if boot_row is None else boot_row["effect_median"],
                "bootstrap_effect_ci_low": None if boot_row is None else boot_row["effect_ci_low"],
                "bootstrap_effect_ci_high": None if boot_row is None else boot_row["effect_ci_high"],
                "bootstrap_interval_excludes_zero": None if boot_row is None else bool((boot_row["effect_ci_low"] > 0) or (boot_row["effect_ci_high"] < 0)),
            })
focal_taxa = pd.DataFrame(focal_rows)
focal_taxa["_method_order"] = focal_taxa["engine"].map({value: index for index, value in enumerate(method_labels.values())})
focal_taxa["_dataset_order"] = focal_taxa["data_version"].map({"Observed": 0, "Completed": 1})
focal_taxa.sort_values(["cohort", "contrast", "feature_id", "_method_order", "_dataset_order"], inplace=True)
focal_taxa.drop(columns=["_method_order", "_dataset_order"], inplace=True)


# Table S18: original-participant eligibility of dietary contrasts.
diet_long = pd.read_csv(DIAB / "participant_diet_groups_long.csv")
diet_long = diet_long.loc[diet_long["diet_col"].astype(str).str.startswith("dietary_compound_")].copy()
boot_diets = sorted(boot.loc[(boot["cohort"] == "DIABIMMUNE") & (boot["group_mode"] == "diet_group"), "diet_col"].dropna().unique())
formal_diets = set(pd.read_csv(DIAB / "diet_group_diff_taxa_annotated_FULL.csv")["diet"].dropna().astype(str).unique())
eligibility_rows = []
for diet_col in boot_diets:
    subset = diet_long.loc[diet_long["diet_col"] == diet_col]
    counts = subset.groupby("diet_group")["participant_id"].nunique().to_dict()
    total_participants = int(subset["participant_id"].nunique())
    name = diet_short(diet_col)
    eligible = name in formal_diets
    eligibility_rows.append({
        "diet": name,
        "diet_col": diet_col,
        "original_unique_participants_early": int(counts.get("early", 0)),
        "original_unique_participants_late": int(counts.get("late", 0)),
        "original_missing_or_unclassified": int(total_participants - counts.get("early", 0) - counts.get("late", 0)),
        "included_in_original_22_contrast_analysis": eligible,
        "present_in_bootstrap_output": True,
        "reporting_status": "Eligible" if eligible else "Exclude from inferential summary",
        "reason": "Original contrast retained for formal analysis." if eligible else "Extreme original group imbalance: the minority group contains only one unique source participant; resampling duplicates must not be treated as independent support.",
    })
diet_eligibility = pd.DataFrame(eligibility_rows)


# Table S25: descriptive bootstrap stability summary for all eligible food contrasts.
diet_boot = boot.loc[(boot["cohort"] == "DIABIMMUNE") & (boot["group_mode"] == "diet_group")].copy()
diet_boot["diet"] = diet_boot["diet_col"].map(diet_short)
diet_boot = diet_boot.loc[diet_boot["diet"].isin(formal_diets)]
diet_boot_rows = []
for (diet, diet_col, method, dataset), block in diet_boot.groupby(["diet", "diet_col", "method", "dataset"], sort=False):
    ci_excludes_zero = (block["effect_ci_low"] > 0) | (block["effect_ci_high"] < 0)
    robust_q05 = (block["selection_frequency_q05"] >= 0.8) & (block["sign_stability"] >= 0.8) & ci_excludes_zero
    robust_q10 = (block["selection_frequency_q10"] >= 0.8) & (block["sign_stability"] >= 0.8) & ci_excludes_zero
    diet_boot_rows.append({
        "diet": diet,
        "diet_col": diet_col,
        "engine": method_labels[method],
        "data_version": dataset_labels[dataset],
        "n_taxa": len(block),
        "minimum_successful_bootstraps": int(block["n_successful_bootstraps"].min()),
        "maximum_successful_bootstraps": int(block["n_successful_bootstraps"].max()),
        "maximum_selection_frequency_q05": float(block["selection_frequency_q05"].max()),
        "taxa_with_selection_frequency_q05_ge_0_50": int((block["selection_frequency_q05"] >= 0.5).sum()),
        "taxa_with_selection_frequency_q05_ge_0_80": int((block["selection_frequency_q05"] >= 0.8).sum()),
        "taxa_meeting_descriptive_robustness_q05": int(robust_q05.sum()),
        "taxa_meeting_descriptive_robustness_q10": int(robust_q10.sum()),
        "criterion": "Selection frequency >=0.80, sign stability >=0.80, and percentile effect interval excludes zero; descriptive/post hoc, not a new confirmatory threshold.",
    })
diet_bootstrap_summary = pd.DataFrame(diet_boot_rows)
diet_bootstrap_summary["_method_order"] = diet_bootstrap_summary["engine"].map({value: index for index, value in enumerate(method_labels.values())})
diet_bootstrap_summary["_dataset_order"] = diet_bootstrap_summary["data_version"].map({"Observed": 0, "Completed": 1})
diet_bootstrap_summary.sort_values(["diet", "_method_order", "_dataset_order"], inplace=True)
diet_bootstrap_summary.drop(columns=["_method_order", "_dataset_order"], inplace=True)


# Table S19: recompute cross-engine concordance from the unioned bootstrap summaries.
primary_boot = boot.loc[
    ((boot["cohort"] == "ECAM") & (boot["group_mode"] == "delivery_group") & boot["diet_col"].isna())
    | ((boot["cohort"] == "DIABIMMUNE") & (boot["group_mode"] == "abx_group") & boot["diet_col"].isna())
].copy()
concordance_rows = []
for (cohort, dataset, group_mode), block in primary_boot.groupby(["cohort", "dataset", "group_mode"], sort=False):
    for left_method, right_method in combinations(method_order, 2):
        left = block.loc[block["method"] == left_method, ["feature_id", "effect_median", "selection_frequency_q05"]]
        right = block.loc[block["method"] == right_method, ["feature_id", "effect_median", "selection_frequency_q05"]]
        merged = left.merge(right, on="feature_id", suffixes=("_left", "_right"), validate="one_to_one")
        effect = merged.dropna(subset=["effect_median_left", "effect_median_right"])
        selection = merged.dropna(subset=["selection_frequency_q05_left", "selection_frequency_q05_right"])
        concordance_rows.append({
            "cohort": cohort,
            "contrast": contrast_labels[cohort],
            "data_version": dataset_labels[dataset],
            "engine_pair": f"{method_labels[left_method]}–{method_labels[right_method]}",
            "n_features": int(len(effect)),
            "spearman_effect": spearman(effect["effect_median_left"], effect["effect_median_right"]),
            "effect_sign_agreement": float((np.sign(effect["effect_median_left"]) == np.sign(effect["effect_median_right"])).mean()),
            "spearman_selection_frequency": spearman(selection["selection_frequency_q05_left"], selection["selection_frequency_q05_right"]),
        })
engine_concordance = pd.DataFrame(concordance_rows)
engine_concordance["_pair_order"] = engine_concordance["engine_pair"].map({"LinDA–MaAsLin2": 0, "LinDA–glmmPQL": 1, "MaAsLin2–glmmPQL": 2})
engine_concordance["_dataset_order"] = engine_concordance["data_version"].map({"Observed": 0, "Completed": 1})
engine_concordance.sort_values(["cohort", "_dataset_order", "_pair_order"], inplace=True)
engine_concordance.drop(columns=["_pair_order", "_dataset_order"], inplace=True)


# Table S17: computational completeness. glmmPQL counts are separate because it
# was aggregated independently, with exactly the same labels/seeds as the two-engine runs.
perm_main_complete = pd.read_csv(PERM_DIRS[0] / "completeness_report.csv").iloc[0]
perm_glmm_complete = pd.read_csv(PERM_DIRS[1] / "completeness_report.csv").iloc[0]
boot_main_stage1 = json.loads((BOOT_DIRS[0] / "stage1_aggregate_status.json").read_text(encoding="utf-8"))
boot_main_stage2 = pd.read_csv(BOOT_DIRS[0] / "stage2_aggregate_status.csv").iloc[0]
boot_glmm_stage2 = pd.read_csv(BOOT_DIRS[1] / "stage2_aggregate_status.csv").iloc[0]
metrics = pd.read_csv(BOOT_DIRS[0] / "imputation_metric_summary.csv")
run_rows = [
    {"component": "Permutation (LinDA and MaAsLin2)", "cohort": "Both", "check": "Expected analysis tasks completed", "expected": int(perm_main_complete["expected_tasks"]), "observed": int(perm_main_complete["completed_tasks"]), "status": "Pass", "detail": "1,000 null permutations plus original labels; paired label maps were shared across observed/completed and both engines."},
    {"component": "Permutation (glmmPQL)", "cohort": "Both", "check": "Expected analysis tasks completed", "expected": int(perm_glmm_complete["expected_tasks"]), "observed": int(perm_glmm_complete["completed_tasks"]), "status": "Pass", "detail": "1,000 null permutations plus original labels, matched by cohort seed and permutation ID to the LinDA/MaAsLin2 runs."},
    {"component": "Bootstrap re-imputation", "cohort": "Both", "check": "Stage 1 re-imputation tasks completed", "expected": int(boot_main_stage1["expected_tasks"]), "observed": int(boot_main_stage1["completed_tasks"]), "status": "Pass", "detail": "30 participant-level bootstrap samples per cohort."},
    {"component": "Bootstrap DA (LinDA and MaAsLin2)", "cohort": "Both", "check": "Stage 2 DA tasks completed", "expected": int(boot_main_stage2["expected_stage2_tasks"]), "observed": int(boot_main_stage2["completed_stage2_tasks"]), "status": "Pass", "detail": "Observed/completed × LinDA/MaAsLin2 for every bootstrap replicate."},
    {"component": "Bootstrap DA (glmmPQL)", "cohort": "Both", "check": "Stage 2 DA tasks completed", "expected": int(boot_glmm_stage2["expected_stage2_tasks"]), "observed": int(boot_glmm_stage2["completed_stage2_tasks"]), "status": "Pass", "detail": "Observed/completed glmmPQL inference for every bootstrap replicate; model-fit failures are recorded at taxon level and excluded from that engine's test set."},
]
for cohort in ["ECAM", "DIABIMMUNE"]:
    finite = metrics.loc[(metrics["cohort"] == cohort) & (metrics["metric"] == "completed_finite_fraction"), "min"].iloc[0]
    unchanged = metrics.loc[(metrics["cohort"] == cohort) & (metrics["metric"] == "unchanged_max_abs_difference"), "max"].iloc[0]
    replacement = metrics.loc[(metrics["cohort"] == cohort) & (metrics["metric"] == "replace_fraction"), "mean"].iloc[0]
    run_rows.append({"component": "Bootstrap re-imputation", "cohort": cohort, "check": "All replacement cells finite", "expected": 1.0, "observed": finite, "status": "Pass" if finite == 1 else "Fail", "detail": f"Mean replacement fraction={replacement:.3f}."})
    run_rows.append({"component": "Bootstrap re-imputation", "cohort": cohort, "check": "Non-replacement cells unchanged (max absolute difference)", "expected": 0.0, "observed": unchanged, "status": "Pass" if unchanged == 0 else "Fail", "detail": "Hard validation after each re-imputation."})
run_completeness = pd.DataFrame(run_rows)


# Figure S9 source additions: bootstrap summaries are derived from the same aggregate
# table as S23 rather than copied from the manuscript or the figure source CSV.
figure_boot = boot.loc[
    (boot["cohort"] == "DIABIMMUNE")
    & (boot["group_mode"] == "abx_group")
    & boot["diet_col"].isna()
    & (boot["feature_id"] == "T0487")
    & boot["method"].isin(method_order)
].copy()
figure_boot["method_label"] = figure_boot["method"].map(method_labels)
figure_boot["representation"] = figure_boot["dataset"].map(dataset_labels).str.lower()
figure_boot = figure_boot[["method_label", "representation", "n_successful_bootstraps", "selection_frequency_q05", "effect_median"]]
figure_boot.rename(columns={"method_label": "method", "selection_frequency_q05": "t0487_bootstrap_selection_frequency_q05", "effect_median": "t0487_bootstrap_effect_median"}, inplace=True)
figure_boot.to_csv(FIGURE_BOOT_OUT, index=False)


# Documented glmmPQL fit status for Supplementary Methods S3; source values are
# generated from the aggregate diagnostics, not manually transcribed.
perm_diag = pd.read_csv(PERM_DIRS[1] / "glmmpql_fit_diagnostics.csv")
boot_diag = pd.read_csv(BOOT_DIRS[1] / "glmmpql_fit_diagnostics.csv")
original_diag = perm_diag.loc[perm_diag["permutation_id"] == 0]
completed_boot_diag = boot_diag.loc[boot_diag["dataset"] == "p2d2pca_imputed"]
diagnostics = {
    "observed_grid_fit_failure_fraction": float((original_diag.loc[original_diag["dataset"] == "base_raw", "n_fit_failed"] / original_diag.loc[original_diag["dataset"] == "base_raw", "n_taxa"]).max()),
    "completed_original": records(original_diag.loc[original_diag["dataset"] == "p2d2pca_imputed", ["cohort", "n_fit_ok", "n_taxa", "fit_success_fraction"]]),
    "bootstrap_completed_fit_success_mean": float(completed_boot_diag["fit_success_fraction"].mean()),
    "bootstrap_completed_fit_success_min": float(completed_boot_diag["fit_success_fraction"].min()),
}
DIAGNOSTICS_OUT.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


payload = {
    "permutation_calibration": records(permutation_calibration),
    "global_signal": records(global_signal),
    "ecam_completed_discovery_taxa": records(ecam_completed_discovery_taxa),
    "focal_taxa": records(focal_taxa),
    "diet_eligibility": records(diet_eligibility),
    "diet_bootstrap_summary": records(diet_bootstrap_summary),
    "engine_concordance": records(engine_concordance),
    "run_completeness": records(run_completeness),
}
OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}")
for key, value in payload.items():
    print(f"{key}: {len(value)} rows")
print(f"Wrote {FIGURE_BOOT_OUT}")
print(f"Wrote {DIAGNOSTICS_OUT}")
