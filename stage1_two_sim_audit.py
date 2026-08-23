from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_publication"
SIMS = {
    "sim1": {
        "root": ROOT / "sim1_compare" / "results_sim1_compare",
        "global": "all_replicates_metrics_sim1.csv",
        "global_summary": "summary_with_errorbar_sim1.csv",
        "global_repeat_col": "replicate",
        "bin": "per_bin_metrics_sim1.csv",
        "bin_repeat_col": "replicate",
        "missing_scope": "MaskedCells",
        "truth": "truth_tensor_sim1.nc",
        "truth_var": "da_truth_sim1",
    },
    "sim2": {
        "root": ROOT / "sim2-compare" / "results_sim2_compare",
        "global": "repeat_level_metrics.csv",
        "global_summary": "errorbar_summary.csv",
        "global_repeat_col": "repeat",
        "bin": "repeat_bin_metrics_all.csv",
        "bin_repeat_col": "repeat",
        "missing_scope": "InjectedCells",
        "truth": "truth_ctf.nc",
        "truth_var": "y_true",
    },
}

OURS = "penalized_2d2_pca"
FRACS = [0.5, 0.6, 0.7, 0.8, 0.9]
TIME_BINS = ["0", "1-4", "5-8", "9-12", "13-16", "17-20", "21-24", "25-28", "29-32", "33-36"]

METHOD_DISPLAY = {
    "penalized_2d2_pca": "Penalized-2D²-PCA",
    "equal": "Equal",
    "mean": "Mean",
    "median": "Median",
    "last": "Last",
    "temporal_knn": "Temporal-KNN",
    "weighted_time_average": "Weighted-Time-Avg",
    "glv_mse_legacy": "gLV-MSE (legacy)",
    "glv_ridge_legacy": "gLV-Ridge (legacy)",
    "knn_epanechnikov_legacy": "KNN-Epanechnikov (legacy)",
    "spline_legacy": "Spline (legacy)",
    "weighted_avg_legacy": "Weighted-Avg (legacy)",
}

GLOBAL_METRICS = {
    "MSE": ("mse_after", "mse_before", "low"),
    "Pearson": ("pearson_after", "pearson_before", "high"),
    "WD": ("wd_after", "wd_before", "low"),
}

ABUNDANCE_METRICS = [
    "bin_ratio_wd_after",
    "bin_avg_euclid_after",
    "marginal_ratio_wd_after",
    "marginal_avg_euclid_after",
    "traj_mean_l2_after",
    "traj_sd_l2_after",
    "traj_mean_sd_joint_euclid_after",
    "taxon_avg_bin_wd_after",
]

TRAJECTORY_METRICS = [
    "traj_mean_l2_after",
    "traj_sd_l2_after",
    "traj_mean_sd_joint_euclid_after",
    "taxon_avg_bin_wd_after",
]


def method_display(raw: str) -> str:
    return METHOD_DISPLAY.get(str(raw), str(raw).replace("_", " ").title())


def mean_ci(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(x.size)
    mean = float(x.mean()) if n else np.nan
    sd = float(x.std(ddof=1)) if n >= 2 else np.nan
    sem = sd / math.sqrt(n) if n >= 2 else np.nan
    half = 1.96 * sem if n >= 3 else np.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": mean - half if n >= 3 else np.nan,
        "ci_high": mean + half if n >= 3 else np.nan,
        "ci_half": half,
    }


def relative_improvement(ours: pd.Series, baseline: pd.Series, direction: str) -> pd.Series:
    denom = baseline.abs() if direction == "high" else baseline
    denom = denom.replace(0, np.nan)
    if direction == "high":
        return (ours - baseline) / denom
    return (baseline - ours) / denom


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def esc(x: object) -> str:
        if x is None:
            return ""
        return str(x).replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(map(esc, headers)) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(esc, row)) + " |" for row in rows)
    return "\n".join(lines)


def pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    ax = abs(float(x))
    if ax != 0 and (ax >= 1e4 or ax < 1e-3):
        return f"{x:.3e}"
    return f"{x:.{digits}g}"


def normalize_scope(simulation: str, scope: str) -> str:
    if scope == "AllCells":
        return "AllCells"
    if simulation == "sim1" and scope == "MaskedCells":
        return "MissingCells"
    if simulation == "sim2" and scope == "InjectedCells":
        return "MissingCells"
    return scope


def inventory_families() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sim, cfg in SIMS.items():
        families: dict[str, list[Path]] = {}
        for path in cfg["root"].rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(cfg["root"]).as_posix()
            family = re.sub(r"frac_(?:0[.p]\d+|0p\d+)", "frac_{F}", rel)
            family = re.sub(r"_repeat_\d+", "_repeat_{R}", family)
            families.setdefault(family, []).append(path)
        for family, paths in sorted(families.items()):
            sample = paths[0]
            columns: list[str] = []
            if sample.suffix.lower() == ".csv":
                with sample.open("r", encoding="utf-8-sig", newline="") as handle:
                    columns = next(csv.reader(handle))
            records.append(
                {
                    "simulation": sim,
                    "file_family": family,
                    "file_count": len(paths),
                    "total_bytes": sum(p.stat().st_size for p in paths),
                    "columns": ", ".join(columns),
                }
            )
    return pd.DataFrame.from_records(records)


def truth_metadata_and_strata() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: list[dict[str, object]] = []
    strata_parts: list[pd.DataFrame] = []
    for sim, cfg in SIMS.items():
        path = cfg["root"] / cfg["truth"]
        with xr.open_dataset(path) as ds:
            da = ds[cfg["truth_var"]]
            meta_rows.append(
                {
                    "simulation": sim,
                    "file": str(path),
                    "variable": cfg["truth_var"],
                    "dimensions": ", ".join(f"{k}={v}" for k, v in ds.sizes.items()),
                    "dtype": str(da.dtype),
                    "time_bin_values": ", ".join(map(str, ds["bin"].values.tolist())),
                }
            )
            means = (
                da.mean(dim=["participant", "bin"], skipna=True)
                .to_series()
                .rename("pseudo_truth_taxon_mean")
                .reset_index()[["taxon", "pseudo_truth_taxon_mean"]]
            )
        # The stored values are transformed (negative values occur), so these are
        # within-simulation relative strata, not absolute raw-abundance categories.
        means["abundance_stratum"] = pd.qcut(
            means["pseudo_truth_taxon_mean"].rank(method="first"),
            3,
            labels=["Low", "Mid", "High"],
        ).astype(str)
        means.insert(0, "simulation", sim)
        means["stratification_rule"] = "tertiles of taxon mean on stored pseudo-truth scale"
        strata_parts.append(means)
    return pd.DataFrame(meta_rows), pd.concat(strata_parts, ignore_index=True)


def compute_global_results() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sim, cfg in SIMS.items():
        data = pd.read_csv(cfg["root"] / cfg["global"]).rename(columns={cfg["global_repeat_col"]: "repeat"})
        data["simulation"] = sim
        data["mask_frac"] = data["frac"].astype(float)
        for (frac, source_scope), group in data.groupby(["mask_frac", "scope"], dropna=False):
            for label, (after_col, before_col, direction) in GLOBAL_METRICS.items():
                method_means = group.groupby("method")[after_col].mean().dropna()
                if OURS not in method_means:
                    continue
                ranks = method_means.rank(ascending=direction == "low", method="min")
                baselines = method_means.drop(index=OURS)
                best_raw = baselines.idxmin() if direction == "low" else baselines.idxmax()

                wide = group.pivot(index="repeat", columns="method", values=after_col)
                paired = wide[[OURS, best_raw]].dropna()
                paired_rel = relative_improvement(paired[OURS], paired[best_raw], direction)
                paired_rank = wide.rank(axis=1, ascending=direction == "low", method="min")[OURS]

                ours_values = group.loc[group["method"] == OURS, after_col]
                baseline_values = group.loc[group["method"] == best_raw, after_col]
                ours_stats = mean_ci(ours_values)
                baseline_stats = mean_ci(baseline_values)
                paired_stats = mean_ci(paired_rel)

                ours_rows = group.loc[group["method"] == OURS, ["repeat", before_col, after_col]].dropna()
                recovery = relative_improvement(ours_rows[after_col], ours_rows[before_col], "low")
                before_stats = mean_ci(ours_rows[before_col])
                recovery_stats = mean_ci(recovery)

                records.append(
                    {
                        "simulation": sim,
                        "mask_frac": float(frac),
                        "source_scope": source_scope,
                        "cell_scope": normalize_scope(sim, source_scope),
                        "metric": label,
                        "direction": direction,
                        "n_repeats": ours_stats["n"],
                        "ours_mean": ours_stats["mean"],
                        "ours_sd": ours_stats["sd"],
                        "ours_ci_low": ours_stats["ci_low"],
                        "ours_ci_high": ours_stats["ci_high"],
                        "condition_rank": float(ranks[OURS]),
                        "average_repeat_rank": float(paired_rank.mean()),
                        "repeat_win_rate": float((paired_rank == 1).mean()),
                        "best_baseline_raw": best_raw,
                        "best_baseline": method_display(best_raw),
                        "best_baseline_mean": baseline_stats["mean"],
                        "best_baseline_sd": baseline_stats["sd"],
                        "best_baseline_ci_low": baseline_stats["ci_low"],
                        "best_baseline_ci_high": baseline_stats["ci_high"],
                        "condition_relative_improvement": float(
                            relative_improvement(
                                pd.Series([ours_stats["mean"]]),
                                pd.Series([baseline_stats["mean"]]),
                                direction,
                            ).iloc[0]
                        ),
                        "paired_relative_improvement_mean": paired_stats["mean"],
                        "paired_relative_improvement_ci_low": paired_stats["ci_low"],
                        "paired_relative_improvement_ci_high": paired_stats["ci_high"],
                        "paired_better_rate": float((paired_rel > 0).mean()),
                        "before_mean": before_stats["mean"],
                        "before_ci_low": before_stats["ci_low"],
                        "before_ci_high": before_stats["ci_high"],
                        "relative_recovery_vs_before_mean": recovery_stats["mean"],
                        "relative_recovery_vs_before_ci_low": recovery_stats["ci_low"],
                        "relative_recovery_vs_before_ci_high": recovery_stats["ci_high"],
                        "marginal_ci_overlap": bool(
                            max(ours_stats["ci_low"], baseline_stats["ci_low"])
                            <= min(ours_stats["ci_high"], baseline_stats["ci_high"])
                        ),
                    }
                )
    return pd.DataFrame.from_records(records).sort_values(["simulation", "cell_scope", "metric", "mask_frac"])


def compute_abundance_results() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sim, cfg in SIMS.items():
        data = pd.read_csv(cfg["root"] / "abundance_repeat_level_metrics.csv")
        data["simulation"] = sim
        data["mask_frac"] = data["frac"].astype(float)
        for (frac, metric), group in data[data["metric"].isin(ABUNDANCE_METRICS)].groupby(["mask_frac", "metric"]):
            method_means = group.groupby("method")["value"].mean().dropna()
            if OURS not in method_means:
                continue
            ranks = method_means.rank(ascending=True, method="min")
            baselines = method_means.drop(index=OURS)
            best_raw = baselines.idxmin()
            wide = group.pivot(index="repeat", columns="method", values="value")
            paired = wide[[OURS, best_raw]].dropna()
            paired_rel = relative_improvement(paired[OURS], paired[best_raw], "low")
            paired_rank = wide.rank(axis=1, ascending=True, method="min")[OURS]

            ours_values = group.loc[group["method"] == OURS, "value"]
            baseline_values = group.loc[group["method"] == best_raw, "value"]
            ours_stats = mean_ci(ours_values)
            baseline_stats = mean_ci(baseline_values)
            paired_stats = mean_ci(paired_rel)

            before_metric = metric.replace("_after", "_before")
            before = data[
                (data["mask_frac"] == frac)
                & (data["method"] == OURS)
                & (data["metric"] == before_metric)
            ].set_index("repeat")["value"]
            after = group[group["method"] == OURS].set_index("repeat")["value"]
            before_after = pd.concat([before.rename("before"), after.rename("after")], axis=1).dropna()
            recovery = relative_improvement(before_after["after"], before_after["before"], "low")
            recovery_stats = mean_ci(recovery)

            finite_ours = pd.to_numeric(ours_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            median = float(finite_ours.median()) if len(finite_ours) else np.nan
            maximum = float(finite_ours.max()) if len(finite_ours) else np.nan
            cv = float(finite_ours.std(ddof=1) / abs(finite_ours.mean())) if len(finite_ours) >= 2 and finite_ours.mean() != 0 else np.nan
            mean_median_ratio = float(finite_ours.mean() / median) if median != 0 else np.nan
            unstable = bool(
                (np.isfinite(cv) and cv > 2)
                or (np.isfinite(mean_median_ratio) and abs(mean_median_ratio) > 10)
                or (median != 0 and maximum / abs(median) > 100)
            )

            records.append(
                {
                    "simulation": sim,
                    "mask_frac": float(frac),
                    "analysis_scope": str(group["scope"].iloc[0]),
                    "metric": metric,
                    "n_repeats": ours_stats["n"],
                    "ours_mean": ours_stats["mean"],
                    "ours_sd": ours_stats["sd"],
                    "ours_ci_low": ours_stats["ci_low"],
                    "ours_ci_high": ours_stats["ci_high"],
                    "ours_median": median,
                    "ours_max": maximum,
                    "ours_cv": cv,
                    "ours_mean_median_ratio": mean_median_ratio,
                    "numerical_instability_flag": unstable,
                    "condition_rank": float(ranks[OURS]),
                    "average_repeat_rank": float(paired_rank.mean()),
                    "repeat_win_rate": float((paired_rank == 1).mean()),
                    "best_baseline_raw": best_raw,
                    "best_baseline": method_display(best_raw),
                    "best_baseline_mean": baseline_stats["mean"],
                    "paired_relative_improvement_mean": paired_stats["mean"],
                    "paired_relative_improvement_ci_low": paired_stats["ci_low"],
                    "paired_relative_improvement_ci_high": paired_stats["ci_high"],
                    "paired_better_rate": float((paired_rel > 0).mean()),
                    "relative_recovery_vs_before_mean": recovery_stats["mean"],
                    "relative_recovery_vs_before_ci_low": recovery_stats["ci_low"],
                    "relative_recovery_vs_before_ci_high": recovery_stats["ci_high"],
                    "recovery_vs_before_positive_rate": float((recovery > 0).mean()),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(["simulation", "metric", "mask_frac"])


def compute_time_bin_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for sim, cfg in SIMS.items():
        data = pd.read_csv(cfg["root"] / cfg["bin"]).rename(columns={cfg["bin_repeat_col"]: "repeat"})
        data = data[data["scope"] == cfg["missing_scope"]].copy()
        data["mask_frac"] = data["frac"].astype(float)
        data["time_bin"] = data["bin"].astype(str)
        for (frac, time_bin), group in data.groupby(["mask_frac", "time_bin"]):
            for label, (after_col, _, direction) in GLOBAL_METRICS.items():
                means = group.groupby("method")[after_col].mean().dropna()
                if OURS not in means:
                    continue
                ranks = means.rank(ascending=direction == "low", method="min")
                baselines = means.drop(index=OURS)
                best_raw = baselines.idxmin() if direction == "low" else baselines.idxmax()
                wide = group.pivot(index="repeat", columns="method", values=after_col)
                paired = wide[[OURS, best_raw]].dropna()
                paired_rel = relative_improvement(paired[OURS], paired[best_raw], direction)
                paired_rank = wide.rank(axis=1, ascending=direction == "low", method="min")[OURS]
                records.append(
                    {
                        "simulation": sim,
                        "mask_frac": float(frac),
                        "cell_scope": "MissingCells",
                        "time_bin": time_bin,
                        "metric": label,
                        "condition_rank": float(ranks[OURS]),
                        "average_repeat_rank": float(paired_rank.mean()),
                        "repeat_win_rate": float((paired_rank == 1).mean()),
                        "best_baseline_raw": best_raw,
                        "best_baseline": method_display(best_raw),
                        "paired_relative_improvement_mean": float(paired_rel.mean()),
                        "paired_better_rate": float((paired_rel > 0).mean()),
                        "n_repeats": int(paired_rel.notna().sum()),
                    }
                )
    detail = pd.DataFrame.from_records(records)
    detail["time_bin"] = pd.Categorical(detail["time_bin"], TIME_BINS, ordered=True)
    summary = (
        detail.groupby(["simulation", "metric", "time_bin"], observed=True)
        .agg(
            n_mask_frac=("condition_rank", "count"),
            median_rank=("condition_rank", "median"),
            min_rank=("condition_rank", "min"),
            max_rank=("condition_rank", "max"),
            rank1_mask_frac=("condition_rank", lambda x: int((x == 1).sum())),
            mean_relative_improvement=("paired_relative_improvement_mean", "mean"),
            positive_mask_frac=("paired_relative_improvement_mean", lambda x: int((x > 0).sum())),
            mean_repeat_win_rate=("repeat_win_rate", "mean"),
        )
        .reset_index()
    )
    detail["time_bin"] = detail["time_bin"].astype(str)
    summary["time_bin"] = summary["time_bin"].astype(str)
    return detail, summary


def compute_abundance_stratum_results(
    abundance_condition: pd.DataFrame, taxon_strata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_records: list[dict[str, object]] = []
    for sim, cfg in SIMS.items():
        sim_conditions = abundance_condition[
            (abundance_condition["simulation"] == sim)
            & abundance_condition["metric"].isin(TRAJECTORY_METRICS)
        ]
        best_map = {
            (float(row.mask_frac), row.metric): row.best_baseline_raw
            for row in sim_conditions.itertuples(index=False)
        }
        selected_by_frac = {
            frac: {method for (f, _), method in best_map.items() if f == frac}
            for frac in FRACS
        }
        eval_path = cfg["root"] / "taxon_trajectory_eval_all.csv"
        usecols = ["frac", "repeat", "method", "taxon"] + TRAJECTORY_METRICS + [
            m.replace("_after", "_improve") for m in TRAJECTORY_METRICS
        ]
        pieces: list[pd.DataFrame] = []
        for chunk in pd.read_csv(eval_path, usecols=usecols, chunksize=150_000):
            keep = chunk["method"].eq(OURS)
            for frac, methods in selected_by_frac.items():
                keep |= chunk["frac"].eq(frac) & chunk["method"].isin(methods)
            if keep.any():
                pieces.append(chunk.loc[keep])
        selected = pd.concat(pieces, ignore_index=True)
        strata = taxon_strata[taxon_strata["simulation"] == sim][
            ["taxon", "pseudo_truth_taxon_mean", "abundance_stratum"]
        ]

        for metric in TRAJECTORY_METRICS:
            improve_col = metric.replace("_after", "_improve")
            for frac in FRACS:
                baseline = best_map[(frac, metric)]
                subset = selected[
                    selected["frac"].eq(frac) & selected["method"].isin([OURS, baseline])
                ][["repeat", "taxon", "method", metric, improve_col]]
                wide = subset.pivot(index=["repeat", "taxon"], columns="method", values=metric).dropna()
                ours_improve = (
                    subset[subset["method"] == OURS]
                    .set_index(["repeat", "taxon"])[[improve_col]]
                )
                paired = wide.join(ours_improve).reset_index().merge(strata, on="taxon", how="left")
                for stratum, group in paired.groupby("abundance_stratum", observed=True):
                    ours_mean = float(group[OURS].mean())
                    baseline_mean = float(group[baseline].mean())
                    rel = (baseline_mean - ours_mean) / baseline_mean if baseline_mean != 0 else np.nan
                    detail_records.append(
                        {
                            "simulation": sim,
                            "mask_frac": frac,
                            "metric": metric,
                            "abundance_stratum": stratum,
                            "stratum_definition": "tertiles of taxon mean on stored pseudo-truth scale",
                            "n_taxa": int(group["taxon"].nunique()),
                            "n_taxon_repeat_pairs": int(len(group)),
                            "ours_mean": ours_mean,
                            "best_baseline_raw": baseline,
                            "best_baseline": method_display(baseline),
                            "best_baseline_mean": baseline_mean,
                            "relative_improvement_of_means": rel,
                            "taxon_repeat_win_rate": float((group[OURS] < group[baseline]).mean()),
                            "recovery_vs_before_positive_rate": float((group[improve_col] > 0).mean()),
                            "median_improvement_vs_before": float(group[improve_col].median()),
                        }
                    )
    detail = pd.DataFrame.from_records(detail_records)
    summary = (
        detail.groupby(["simulation", "metric", "abundance_stratum"], observed=True)
        .agg(
            n_taxa=("n_taxa", "max"),
            mean_relative_improvement=("relative_improvement_of_means", "mean"),
            min_relative_improvement=("relative_improvement_of_means", "min"),
            max_relative_improvement=("relative_improvement_of_means", "max"),
            positive_mask_frac=("relative_improvement_of_means", lambda x: int((x > 0).sum())),
            mean_taxon_repeat_win_rate=("taxon_repeat_win_rate", "mean"),
            mean_recovery_vs_before_positive_rate=("recovery_vs_before_positive_rate", "mean"),
        )
        .reset_index()
    )
    return detail, summary


def method_mapping(inventory: pd.DataFrame) -> pd.DataFrame:
    methods: set[str] = set()
    for sim, cfg in SIMS.items():
        data = pd.read_csv(cfg["root"] / cfg["global"], usecols=["method", "method_group"])
        for row in data.drop_duplicates().itertuples(index=False):
            methods.add(row.method)
    rows = []
    for raw in sorted(methods):
        groups = set()
        present = []
        for sim, cfg in SIMS.items():
            data = pd.read_csv(cfg["root"] / cfg["global"], usecols=["method", "method_group"])
            hit = data[data["method"] == raw]
            if not hit.empty:
                present.append(sim)
                groups.update(hit["method_group"].dropna().astype(str).unique())
        rows.append(
            {
                "raw_method_name": raw,
                "display_method_name": method_display(raw),
                "method_group": "; ".join(sorted(groups)),
                "simulation_present": "; ".join(present),
                "mapping_basis": "explicit normalization rule" if raw in METHOD_DISPLAY else "title-case fallback",
            }
        )
    return pd.DataFrame(rows)


def write_data_audit_report(
    inventory: pd.DataFrame,
    truth_meta: pd.DataFrame,
    methods: pd.DataFrame,
) -> None:
    sim1 = SIMS["sim1"]["root"]
    sim2 = SIMS["sim2"]["root"]
    required_rows = [
        ["Global error-bar summary", "summary_with_errorbar_sim1.csv (name-equivalent)", "errorbar_summary.csv", "available in both"],
        ["Repeat-level global metrics", "all_replicates_metrics_sim1.csv (name-equivalent)", "repeat_level_metrics.csv", "available in both"],
        ["Repeat-by-bin metrics", "per_bin_metrics_sim1.csv (name-equivalent)", "repeat_bin_metrics_all.csv", "available in both; bin is time"],
        ["Abundance error-bar summary", "abundance_errorbar_summary.csv", "abundance_errorbar_summary.csv", "available in both"],
        ["Abundance repeat-level metrics", "abundance_repeat_level_metrics.csv", "abundance_repeat_level_metrics.csv", "available in both"],
        ["Bin distribution evaluation", "bin_distribution_eval_all.csv", "bin_distribution_eval_all.csv", "available in both; bin is time"],
        ["Pseudo-group effect", "missing", "missing", "unavailable"],
        ["Taxon-effect model", "missing", "missing", "unavailable"],
        ["Taxon trajectory evaluation", "taxon_trajectory_eval_all.csv", "taxon_trajectory_eval_all.csv", "available in both; cohort/taxon aggregate"],
        ["Taxon marginal evaluation", "taxon_marginal_eval_all.csv", "taxon_marginal_eval_all.csv", "available in both"],
        ["Pseudo-truth tensor", "truth_tensor_sim1.nc", "truth_ctf.nc", "available in both"],
    ]

    method_list = ", ".join(methods["display_method_name"].tolist())
    sections = [
        "# Data audit report (Stage 0-1 input audit)",
        "",
        "## Scope and simulation mapping",
        "",
        f"- sim1: `{sim1}`. The path explicitly contains `sim1`; the pseudo-truth is the Penalized-2D²-PCA-derived tensor described by the user.",
        f"- sim2: `{sim2}`. The path explicitly contains `sim2`; the pseudo-truth file is `truth_ctf.nc`.",
        "- Only these two directories were recursively inventoried. No unrelated project directory was scanned, and no source result file was modified.",
        "- Every derived table written in `figures_publication/` contains a `simulation` column.",
        "",
        "## Required logical inputs and actual files",
        "",
        markdown_table(["Logical input", "sim1", "sim2", "Audit status"], required_rows),
        "",
        "The three apparently missing sim1 standard names are naming differences, not data loss. Their sim1 equivalents have matching schemas and 6,000 global repeat rows / 60,000 time-bin rows.",
        "",
        "## Detected design values",
        "",
        markdown_table(
            ["Item", "sim1", "sim2"],
            [
                ["mask_frac", "0.5, 0.6, 0.7, 0.8, 0.9", "0.5, 0.6, 0.7, 0.8, 0.9"],
                ["source cell scopes", "AllCells, MaskedCells", "AllCells, InjectedCells"],
                ["standard cell scopes", "AllCells, MissingCells", "AllCells, MissingCells"],
                ["methods", method_list, method_list],
                ["global metrics", "MSE, Pearson, WD (before / after / improve)", "MSE, Pearson, WD (before / after / improve)"],
                ["repeat information", "50 replicates per frac × method × scope", "50 repeats per frac × method × scope"],
                ["time bins", ", ".join(TIME_BINS), ", ".join(TIME_BINS)],
                ["truth tensor shape", "130 participants × 546 taxa × 10 time bins", "130 participants × 263 taxa × 10 time bins"],
            ],
        ),
        "",
        "## File families and columns",
        "",
        "Thousands of per-repeat files share identical schemas; listing them one by one would obscure the audit. They are grouped below by normalized filename family, with the exact count and columns. `{F}` and `{R}` denote mask fraction and repeat.",
        "",
    ]
    for sim in ["sim1", "sim2"]:
        sections.extend([f"### {sim}", ""])
        for row in inventory[inventory["simulation"] == sim].itertuples(index=False):
            size_gib = row.total_bytes / (1024**3)
            columns = row.columns if row.columns else "NetCDF; see truth tensor metadata below"
            sections.append(f"- `{row.file_family}` — {row.file_count} file(s), {size_gib:.3f} GiB; columns: `{columns}`")
        sections.append("")

    truth_rows = [
        [r.simulation, Path(r.file).name, r.variable, r.dimensions, r.dtype, r.time_bin_values]
        for r in truth_meta.itertuples(index=False)
    ]
    sections.extend(
        [
            "## Raw and aggregate data availability",
            "",
            markdown_table(["Simulation", "Truth file", "Variable", "Dimensions", "dtype", "bin coordinate"], truth_rows),
            "",
            "- A raw pseudo-truth tensor exists for each simulation.",
            "- No raw per-method `before` or `imputed` participant × taxon × time tensor was found by filename or schema. Only repeat-, taxon-, and time-bin-level derived summaries are available for these roles.",
            "- `taxon_bin_stats_*` and `taxon_marginal_stats_*` contain `dataset_role` and can support aggregate distribution summaries, but cannot reconstruct participant-specific imputed trajectories.",
            "- Consequently, subject × taxon representative trajectory panels and subject-level failure detection are not supported by the current folders.",
            "",
            "## Analysis routing",
            "",
            "- Metric benchmark: global repeat tables and error-bar summaries in both simulations.",
            "- Distribution recovery: global WD plus `abundance_*`, `bin_distribution_eval_*`, and taxon marginal summaries in both simulations.",
            "- Time-course recovery: `taxon_trajectory_eval_*` and `taxon_trajectory_detail_*` in both simulations; these describe taxon/cohort-level trajectory summaries, not individual participant curves.",
            "- Time-bin stability: repeat-bin tables in both simulations. The NetCDF coordinate proves `0`, `1–4`, …, `33–36` is the tensor's time dimension, not an abundance bin.",
            "- Abundance stratification: not directly precomputed. Stage 1 derives fixed within-simulation Low/Mid/High strata from tertiles of each taxon's mean value in the stored pseudo-truth tensor. Because the tensors contain negative values and no scale metadata, these are relative strata on the stored transformed scale, not raw relative-abundance cutoffs.",
            "- Effect/pseudo-group analyses cannot be done because `effect_pseudo_group.csv` and `effect_taxon_effect.csv` (or schema-equivalent files) are absent from both folders.",
            "",
            "## Analyses available in both simulations",
            "",
            "Global MSE/Pearson/WD ranking, best-baseline comparisons, paired relative improvement, 95% CI from 50 repeats, global WD recovery from before, time-bin stability, taxon-marginal summaries, taxon/cohort trajectory summaries, and fixed-rule pseudo-truth-mean strata are available in both simulations.",
            "",
            "## Field standardization and inference",
            "",
            markdown_table(
                ["Source", "Original field/value", "Standardized field/value", "Basis"],
                [
                    ["both", "frac", "mask_frac", "semantic match and values 0.5–0.9"],
                    ["sim1", "replicate", "repeat", "same grouping role as sim2 repeat"],
                    ["sim1", "model_seed", "algo_seed", "same role as sim2 algo_seed"],
                    ["sim1", "MaskedCells", "MissingCells", "cells intentionally masked for evaluation"],
                    ["sim2", "InjectedCells", "MissingCells", "cells replaced/injected for evaluation"],
                    ["sim2 tables", "simulation absent", "simulation=sim2", "explicit directory mapping"],
                    ["sim1 tables", "simulation absent in some files", "simulation=sim1", "explicit directory mapping"],
                    ["both", "bin", "time_bin", "NetCDF dimension/coordinate and tensor layout"],
                    ["both", "pseudo-truth mean tertiles", "abundance_stratum", "predeclared relative-stratification rule; transformed scale caveat retained"],
                ],
            ),
            "",
            "Method names are standardized in `method_name_mapping.csv`; field mappings are also saved in `field_name_mapping.csv`. No inference was silently applied.",
        ]
    )
    (OUT / "data_audit_report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def global_condition_table(global_results: pd.DataFrame, scope: str = "MissingCells") -> str:
    rows: list[list[object]] = []
    subset = global_results[global_results["cell_scope"] == scope]
    for sim in ["sim1", "sim2"]:
        for metric in ["MSE", "Pearson", "WD"]:
            g = subset[(subset["simulation"] == sim) & (subset["metric"] == metric)].sort_values("mask_frac")
            rows.append(
                [
                    sim,
                    metric,
                    ", ".join(str(int(x)) for x in g["condition_rank"]),
                    ", ".join(pct(x) for x in g["paired_relative_improvement_mean"]),
                    ", ".join(g["best_baseline"].tolist()),
                    ", ".join(pct(x) for x in g["repeat_win_rate"]),
                ]
            )
    return markdown_table(
        ["Simulation", "Metric", "Ranks at frac 0.5→0.9", "Paired improvement vs best baseline", "Best baselines", "Repeat win rate"],
        rows,
    )


def abundance_key_table(abundance_results: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    metrics = ["traj_mean_l2_after", "traj_sd_l2_after", "taxon_avg_bin_wd_after", "marginal_ratio_wd_after"]
    for sim in ["sim1", "sim2"]:
        for metric in metrics:
            g = abundance_results[
                (abundance_results["simulation"] == sim) & (abundance_results["metric"] == metric)
            ].sort_values("mask_frac")
            rows.append(
                [
                    sim,
                    metric,
                    ", ".join(str(int(x)) for x in g["condition_rank"]),
                    ", ".join(pct(x) for x in g["paired_relative_improvement_mean"]),
                    int(g["numerical_instability_flag"].sum()),
                ]
            )
    return markdown_table(
        ["Simulation", "Taxon/distribution metric", "Ranks at frac 0.5→0.9", "Paired improvement vs best baseline", "Flagged fracs"],
        rows,
    )


def write_result_interpretation_report(
    global_results: pd.DataFrame,
    abundance_results: pd.DataFrame,
    strata_summary: pd.DataFrame,
    time_summary: pd.DataFrame,
) -> None:
    target = global_results[global_results["cell_scope"] == "MissingCells"]
    wd_rows: list[list[object]] = []
    for sim in ["sim1", "sim2"]:
        for row in target[(target["simulation"] == sim) & (target["metric"] == "WD")].sort_values("mask_frac").itertuples(index=False):
            wd_rows.append(
                [
                    sim,
                    row.mask_frac,
                    f"{num(row.before_mean)} [{num(row.before_ci_low)}, {num(row.before_ci_high)}]",
                    f"{num(row.ours_mean)} [{num(row.ours_ci_low)}, {num(row.ours_ci_high)}]",
                    pct(row.relative_recovery_vs_before_mean),
                    pct(row.paired_relative_improvement_mean),
                ]
            )

    strata_rows: list[list[object]] = []
    for sim in ["sim1", "sim2"]:
        for metric in ["traj_mean_l2_after", "traj_sd_l2_after"]:
            for stratum in ["Low", "Mid", "High"]:
                hit = strata_summary[
                    (strata_summary["simulation"] == sim)
                    & (strata_summary["metric"] == metric)
                    & (strata_summary["abundance_stratum"] == stratum)
                ].iloc[0]
                strata_rows.append(
                    [
                        sim,
                        metric,
                        stratum,
                        int(hit["n_taxa"]),
                        f"{int(hit['positive_mask_frac'])}/5",
                        pct(hit["mean_taxon_repeat_win_rate"]),
                        pct(hit["mean_recovery_vs_before_positive_rate"]),
                    ]
                )

    sim1_time = time_summary[time_summary["simulation"] == "sim1"]
    sim2_time = time_summary[time_summary["simulation"] == "sim2"]
    sim1_time_rank1 = int((sim1_time["rank1_mask_frac"] == 5).sum())
    sim2_time_rank1 = int((sim2_time["rank1_mask_frac"] == 5).sum())

    sections = [
        "# Result interpretation report (Stage 1)",
        "",
        "## Decision",
        "",
        "**Do not automatically proceed to a main figure claiming robust superiority across both pseudo-truth settings.** The result is highly informative, but the defensible message is pseudo-truth dependence and a mask-fraction crossover, not universal dominance.",
        "",
        "A SCI main figure can be justified only after targeted QC of the sim2 taxon-level heavy tails. If produced now, it must be a conservative diagnostic/validation figure that displays the sim1–sim2 disagreement rather than hiding it. No figure was generated in this run.",
        "",
        "## 1. Global performance at the imputation-relevant scope",
        "",
        "The table uses the source scopes `MaskedCells` (sim1) and `InjectedCells` (sim2), standardized as `MissingCells`. Values are ordered by mask_frac 0.5, 0.6, 0.7, 0.8, 0.9. Relative improvements are paired over 50 repeats against the best non-Penalized baseline selected by condition; negative values mean Penalized-2D²-PCA is worse.",
        "",
        global_condition_table(global_results),
        "",
        "### sim1 interpretation",
        "",
        "- Penalized-2D²-PCA ranks first for MSE, Pearson, and WD at every mask fraction in both MissingCells and AllCells.",
        "- At MissingCells, paired MSE improvement over the strongest baseline ranges from 61.0% to 94.6%; Pearson from 1.88% to 99.7%; WD from 9.61% to 92.1%.",
        "- The smallest advantage is WD at frac=0.5, but its paired 95% CI remains above zero. All global comparisons use n=50 and the paired CIs exclude zero.",
        "- This is exceptionally strong self-consistency performance, but sim1's pseudo-truth was constructed by the evaluated method, so it cannot stand alone as independent validation.",
        "",
        "### sim2 interpretation",
        "",
        "- MSE is rank 6 at frac 0.5–0.7, then rank 1 at 0.8–0.9. Relative differences are −72.2%, −73.8%, −36.9%, +22.9%, +34.9%.",
        "- WD is rank 6 at frac 0.5–0.8 and rank 1 only at 0.9. Relative differences are −43.6%, −71.4%, −75.0%, −5.29%, +56.8%.",
        "- Pearson is comparatively favorable: rank 2 at 0.5 and rank 1 at 0.6–0.9 for MissingCells. At 0.5 it remains 1.39% below Mean; thereafter it is 9.16% to 139% above the best baseline under the specified relative formula.",
        "- The same broad crossover appears in AllCells, although low-frac Pearson ranks are weaker there. This is not an artifact of displaying only one scope.",
        "- The paired CIs are narrow and exclude zero in the adverse as well as favorable conditions. Therefore the sim2 disagreement is stable across repeats, not sampling noise.",
        "",
        "## 2. Global distribution recovery (WD)",
        "",
        markdown_table(
            ["Simulation", "mask_frac", "Before WD mean [95% CI]", "Ours WD mean [95% CI]", "Reduction vs before", "Ours vs best baseline"],
            wd_rows,
        ),
        "",
        "Penalized-2D²-PCA substantially reduces WD relative to the zero-injected/masked before state in both simulations (sim1: 96.9–99.8%; sim2: 83.4–86.2%). However, recovery relative to before is not the same as superiority: in sim2, Temporal-KNN/Last or another baseline yields lower WD through frac=0.8. This distinction must be explicit in any caption.",
        "",
        "## 3. Taxon-distribution and trajectory summaries",
        "",
        abundance_key_table(abundance_results),
        "",
        "- In sim1, trajectory/taxon WD metrics are worse than the best baseline at frac 0.5–0.6 and become rank 1 from 0.7 onward. Thus even sim1's global cell-level dominance does not imply uniform trajectory-summary dominance at lower masking.",
        "- In sim2, Penalized-2D²-PCA is rank 7–12 on all four trajectory summary metrics at every frac and never beats the condition's best baseline on the mean. Mean-trajectory error often improves over before, but SD and joint mean/SD recovery show repeat-level failures and large extreme values.",
        "- The sim2 trajectory summaries are strongly right-skewed: for example, at frac=0.6 the Penalized mean trajectory L2 has mean 4.33, median 1.35, and maximum 117; trajectory SD L2 has mean 31.8, median 8.50, and maximum 940. Mean ± 95% CI alone is not an adequate visual summary.",
        "- Ratio-based distribution metrics are numerically unsafe in sim1: `bin_ratio_wd_after` reaches approximately 10^14 and `marginal_ratio_wd_after` reaches approximately 10^14 at high masking while medians may be orders of magnitude smaller. This is consistent with division by near-zero SD in a mean/SD statistic. These metrics should not enter a main figure until the denominator rule is audited and regularized.",
        "- The tables contain cohort/taxon summary trajectories, not participant-specific imputed series. Oversmoothing, mean regression, or failure for a specific subject cannot be established from current outputs.",
        "",
        "## 4. Relative pseudo-truth-mean strata",
        "",
        "Low/Mid/High are fixed tertiles of taxon mean in each simulation's stored pseudo-truth tensor. They are within-simulation relative groups on a transformed scale (negative values occur), not absolute biological abundance categories.",
        "",
        markdown_table(
            ["Simulation", "Metric", "Stratum", "Taxa", "Fracs beating best baseline", "Taxon×repeat win rate", "Improved vs before"],
            strata_rows,
        ),
        "",
        "- sim1 is not uniformly robust across strata: the middle stratum is strongest, while low-stratum trajectory SD is especially weak against the best baseline. This argues against claiming that low-abundance dynamics are uniformly preserved.",
        "- sim2 beats the selected best baseline in 0/5 fractions in every stratum for the trajectory metrics. Mean trajectory levels often improve versus before, but trajectory variability recovery is poor, especially in low and mid strata.",
        "- Because the stored scale is undocumented, this is a sensitivity analysis, not a definitive low-abundance biological result.",
        "",
        "## 5. Time-bin stability",
        "",
        f"The per-bin axis is time. Across 10 time bins x 3 global metrics, sim1 is rank 1 at all five mask fractions in {sim1_time_rank1}/30 time-bin/metric combinations; sim2 achieves this in only {sim2_time_rank1}/30. The complete results are in `stage1_time_bin_summary.csv`. These results may support a supplementary time-bin heatmap, not an abundance-bin panel.",
        "",
        "## 6. Cross-simulation consistency",
        "",
        "1. The advantage does not appear consistently in both simulations. sim1 is uniformly favorable globally; sim2 is favorable mainly for Pearson and at high masking, while cell-level MSE/WD and all taxon-trajectory summaries are adverse over much of the design.",
        "2. The very large sim1–sim2 gap is compatible with self-consistency bias and/or pseudo-truth dependence. It does not prove bias by itself because sim1 and sim2 also differ in taxon count (546 vs 263), tensor scale/distribution, and pseudo-truth construction.",
        "3. sim2 does not currently support a general claim of robustness to an independent pseudo-truth mechanism. It supports a narrower claim: the method becomes globally competitive under severe masking and retains strong Pearson behavior, but distribution/trajectory recovery is unstable.",
        "4. If shown, sim1 and sim2 must be separate rows/facets, with negative relative improvements visible. Pooling is inappropriate.",
        "5. sim2 is the more independent validation setting and therefore should carry greater evidentiary weight. Its adverse results cannot be relegated to a supplement while sim1 is used as the sole main result.",
        "6. sim1 is best framed as a self-consistency/sensitivity experiment, not the primary proof of superiority.",
        "",
        "## 7. Main-figure panel recommendation",
        "",
        "Recommended only after sim2 QC: (A) cross-simulation rank/relative-improvement heatmap for all fracs and three global metrics; (B) MissingCells metric curves with 95% CI and consistent baselines; (C) before→after WD recovery alongside best-baseline WD; (D) robust taxon trajectory summary using median/IQR plus tail/failure rate; (E) fixed-rule pseudo-truth-mean stratum performance. A detailed conditional blueprint is saved as `main_figure_blueprint.md`.",
        "",
        "Supplementary: AllCells results, all baseline methods, time-bin heatmaps, full repeat distributions, and sensitivity analyses using median/trimmed mean for heavy-tailed taxon metrics.",
        "",
        "Do not emphasize or visualize as headline evidence: ratio WD based on mean/SD before denominator QC; a single favorable frac=0.9 panel; pooled sim1+sim2 averages; participant-level trajectory examples (data unavailable); or a low-abundance claim based on time bins.",
        "",
        "## 8. Cherry-picking and reviewer risks",
        "",
        "- Showing only frac=0.9 would hide the sim2 crossover and is unacceptable.",
        "- Showing only sim1 would overstate evidence from a method-derived pseudo-truth.",
        "- Showing only Pearson would conceal adverse MSE/WD and trajectory results.",
        "- Baseline identity changes across conditions; any reduced main-figure baseline set must be fixed by an explicit cross-simulation rule, with the full comparison supplied separately.",
        "- Current pseudo-truth tensors differ in taxon count and likely scale/distribution; absolute metric magnitudes are not directly comparable across simulations.",
        "- Sim2 taxon-level errors have extreme tails that may represent genuine failures, numerical instability, or scale mismatch. This must be audited before publication.",
        "- Mean/SD ratio metrics appear ill-conditioned when SD approaches zero.",
        "- The absence of raw per-method imputed tensors prevents participant-level trajectory verification and representative trajectory selection.",
        "- No pseudo-group or taxon-effect files are present, so heterogeneity inference is incomplete.",
        "",
        "## 9. Recommended paper narrative",
        "",
        "A defensible narrative is: Penalized-2D²-PCA is highly self-consistent under a method-derived pseudo-truth and shows strong correlation plus a high-masking advantage under a CTF-derived pseudo-truth, but its MSE/WD advantage and taxon-trajectory fidelity depend materially on the pseudo-truth construction and masking severity. This is a boundary-of-validity result, not evidence of universal superiority.",
    ]
    (OUT / "result_interpretation_report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def write_blueprint() -> None:
    text = """# Conditional main figure blueprint (no figure generated)

## Gate decision

Do not render the main figure until the sim2 heavy-tail and scale checks listed below are completed. The current data do not support a title or caption claiming robust superiority across both pseudo-truth settings. They do support a conservative figure about pseudo-truth dependence and a masking-severity crossover.

## Recommended scientific message

Penalized-2D²-PCA is strongly self-consistent in sim1 and becomes globally competitive under severe masking in sim2, especially for Pearson, but MSE/WD and taxon-trajectory recovery depend on pseudo-truth construction and masking severity.

## Proposed panels

### A. Cross-simulation rank and relative-improvement heatmap

- Question: Is the method consistently top-ranked across simulation, mask fraction, scope, and metric?
- Data: `stage1_global_condition_results.csv`.
- Layout: rows = sim1/sim2 × MSE/Pearson/WD; columns = frac 0.5–0.9; cell color = paired relative improvement; text = rank.
- Both simulations: yes, shown separately; no pooling.
- Rationale: directly exposes the favorable sim1 block and sim2 crossover without selecting only frac=0.9.

### B. MissingCells global metric curves

- Question: What are the absolute performance trajectories with masking severity?
- Data: repeat-level global metric files and `stage1_global_condition_results.csv`.
- Layout: 2 rows (sim1/sim2) × 3 columns (MSE/Pearson/WD); mean ± 95% CI.
- Methods: Penalized-2D²-PCA, a fixed simple baseline (Mean), a fixed temporal baseline (Temporal-KNN or Weighted-Time-Avg chosen by an across-condition rule), and the strongest remaining baseline. The same displayed set must be used in both simulations.
- Rationale: ranks alone can hide absolute effect size and scale.

### C. Distribution recovery versus before and best baseline

- Question: Does the method repair the zero-injected distribution even when it is not the best method?
- Data: global WD fields in `stage1_global_condition_results.csv`.
- Layout: paired before→ours points/lines for each frac, overlaid with best-baseline WD; separate facets for sim1 and sim2.
- Rationale: distinguishes substantial recovery from comparative superiority.

### D. Robust taxon-trajectory fidelity and failure tails

- Question: Are trajectory-summary gains typical, or driven by failures/outliers?
- Data: `abundance_repeat_level_metrics.csv`, `taxon_trajectory_eval_all.csv`, and `stage1_abundance_condition_results.csv`.
- Layout: median and IQR of log error ratio (ours / fixed baseline) plus the fraction of taxon×repeat pairs won; separate mean-trajectory and SD-trajectory metrics.
- Both simulations: yes.
- Rationale: sim2 is strongly right-skewed; mean ± CI alone is misleading.

### E. Fixed-rule pseudo-truth-mean strata

- Question: Does performance vary across relatively low/mid/high taxon mean levels?
- Data: `stage1_taxon_abundance_strata.csv` and `stage1_abundance_stratum_results.csv`.
- Layout: Low/Mid/High on x, taxon×repeat win rate or log error ratio on y; facets by simulation and trajectory metric.
- Caveat in panel/caption: strata are tertiles on the stored transformed pseudo-truth scale, not raw biological abundance.
- Rationale: honestly shows sim1's low-stratum weakness and sim2's broad failure against the best baseline.

## Excluded candidate panels

- Selected frac=0.9 benchmark: excluded as a stand-alone panel because it is the most favorable sim2 condition and would create cherry-picking risk.
- Ratio-WD distribution panel: excluded until near-zero SD handling is audited; current values reach approximately 10^14.
- Representative participant×taxon trajectories: excluded because raw per-method imputed tensors are absent. Current trajectory files are cohort/taxon summaries.
- “Abundance-bin” panel based on `bin`: excluded because the NetCDF metadata identifies this as time, not abundance.
- Pooled simulation summary: excluded because conclusion directions differ.

## Required QC before rendering

1. Confirm that sim1 and sim2 are on commensurate evaluation scales within each metric and that the sim2 transform/inverse-transform is correct.
2. Inspect the sim2 repeats/taxa generating maxima of 117 (mean trajectory L2) and 940 (trajectory SD L2 at frac=0.6); classify algorithm failure, numerical failure, or genuine difficult cases.
3. Recompute taxon-summary reporting with median/IQR, trimmed mean, and finite-value diagnostics; retain the ordinary mean as the prespecified primary statistic but do not let it stand alone.
4. Define an epsilon or exclusion rule for mean/SD ratios before recomputing ratio WD; record how many taxa are affected.
5. Verify that the same method outputs and method versions are compared in both simulations; several methods are explicitly labeled `legacy`.
6. If a representative trajectory panel is required, export per-method participant × taxon × time imputed tensors and masked-position indicators first.

## Recommended title

Pseudo-truth construction and masking severity govern the apparent recovery advantage of Penalized-2D²-PCA

## Caption draft

Performance of Penalized-2D²-PCA under two pseudo-truth simulation settings. Sim1 uses a Penalized-2D²-PCA-derived pseudo-truth and measures self-consistent recovery; sim2 uses a CTF-derived pseudo-truth and provides a more independent low-rank reconstruction setting. Results are shown separately for all masking fractions and are never pooled. Global MSE and Wasserstein distance are minimized, while Pearson correlation is maximized; points and intervals summarize 50 repeats. Relative improvement is calculated against the strongest non-Penalized baseline within each prespecified condition, with negative values indicating worse performance. Taxon-level panels use robust summaries because sim2 errors are heavy-tailed. Low/Mid/High strata are tertiles of taxon mean on each simulation's stored transformed pseudo-truth scale. The figure is intended to show strong sim1 self-consistency, a sim2 high-masking crossover, and limitations in taxon-trajectory recovery rather than universal superiority.

## Supplementary material

Place complete all-method curves, AllCells results, time-bin heatmaps, repeat-level distributions, outlier diagnostics, denominator-sensitivity analyses, and the full taxon-stratum tables in supplementary figures/tables.
"""
    (OUT / "main_figure_blueprint.md").write_text(text, encoding="utf-8")


def write_field_mapping() -> None:
    rows = [
        ["both", "frac", "mask_frac", "column", "semantic standardization"],
        ["sim1", "replicate", "repeat", "column", "repeat identifier"],
        ["sim1", "model_seed", "algo_seed", "column", "algorithm/model random seed"],
        ["sim1", "MaskedCells", "MissingCells", "scope value", "evaluation cells intentionally masked"],
        ["sim2", "InjectedCells", "MissingCells", "scope value", "evaluation cells replaced with zero"],
        ["both", "bin", "time_bin", "column", "NetCDF tensor dimension and coordinate"],
        ["both", "missing simulation column", "simulation", "attached column", "explicit directory mapping"],
        ["both", "taxon pseudo-truth mean tertile", "abundance_stratum", "derived field", "fixed within-simulation stratification"],
    ]
    pd.DataFrame(rows, columns=["simulation", "source", "standardized", "mapping_type", "basis"]).to_csv(
        OUT / "field_name_mapping.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = inventory_families()
    truth_meta, taxon_strata = truth_metadata_and_strata()
    methods = method_mapping(inventory)
    global_results = compute_global_results()
    abundance_results = compute_abundance_results()
    time_detail, time_summary = compute_time_bin_results()
    strata_detail, strata_summary = compute_abundance_stratum_results(abundance_results, taxon_strata)

    inventory.to_csv(OUT / "stage1_file_inventory.csv", index=False, encoding="utf-8-sig")
    truth_meta.to_csv(OUT / "stage1_truth_tensor_metadata.csv", index=False, encoding="utf-8-sig")
    taxon_strata.to_csv(OUT / "stage1_taxon_abundance_strata.csv", index=False, encoding="utf-8-sig")
    methods.to_csv(OUT / "method_name_mapping.csv", index=False, encoding="utf-8-sig")
    global_results.to_csv(OUT / "stage1_global_condition_results.csv", index=False, encoding="utf-8-sig")
    abundance_results.to_csv(OUT / "stage1_abundance_condition_results.csv", index=False, encoding="utf-8-sig")
    time_detail.to_csv(OUT / "stage1_time_bin_condition_results.csv", index=False, encoding="utf-8-sig")
    time_summary.to_csv(OUT / "stage1_time_bin_summary.csv", index=False, encoding="utf-8-sig")
    strata_detail.to_csv(OUT / "stage1_abundance_stratum_results.csv", index=False, encoding="utf-8-sig")
    strata_summary.to_csv(OUT / "stage1_abundance_stratum_summary.csv", index=False, encoding="utf-8-sig")
    write_field_mapping()

    write_data_audit_report(inventory, truth_meta, methods)
    write_result_interpretation_report(global_results, abundance_results, strata_summary, time_summary)
    write_blueprint()

    print("Stage 1 data audit complete: yes")
    print(f"sim1: {SIMS['sim1']['root']}")
    print(f"sim2: {SIMS['sim2']['root']}")
    print("Automatic approval for a robust-superiority main figure: no")
    print("Figure rendered: no")
    print("Conditional blueprint written: figures_publication/main_figure_blueprint.md")
    print("Primary reason: sim1 is uniformly favorable, whereas sim2 shows low/mid-mask MSE/WD losses and taxon-trajectory heavy tails.")


if __name__ == "__main__":
    main()
