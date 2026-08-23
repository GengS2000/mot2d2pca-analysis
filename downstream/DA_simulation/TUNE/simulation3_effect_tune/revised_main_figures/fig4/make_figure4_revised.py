"""Build the A4-portrait, five-panel downstream DA figure.

Outputs
-------
One A4 portrait composite in vector PDF, 600-dpi PNG, and losslessly
compressed 600-dpi TIFF. Panels are the downstream-focused B-F panels from
the earlier integrated figure, relabelled A-E. The already-reported global
13-method Simulation 3 benchmark is intentionally omitted.

Environment variables
---------------------
FIG4_SOURCE_ROOT : simulation3_effect_tune directory containing the source
                   tensors and publication_figures_imputation_DA directory.
FIG4_OUTPUT_DIR  : output directory; defaults to this script's directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import os
import sys

_LOCAL_DEPS = Path(__file__).resolve().parent / "pydeps"
if _LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(_LOCAL_DEPS))

_REPO_VENDOR = Path(__file__).resolve().parents[4] / "figures_publication" / "_vendor"
if _REPO_VENDOR.is_dir():
    sys.path.insert(0, str(_REPO_VENDOR))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(
    os.environ.get(
        "FIG4_SOURCE_ROOT",
        r"D:\penalized_2d2\penalized_2d2_pca\downstream\DA_simulation\TUNE\simulation3_effect_tune",
    )
).resolve()
REPO_ROOT = ROOT.parents[3]
CORE_CSV = REPO_ROOT / "sim3_compare" / "core_csv"
CTF_CORE_CSV = REPO_ROOT / "CTF_artifacts" / "sim3"
SOURCE_FIG_DIR = ROOT / "publication_figures_imputation_DA"
LINDA_ADDITIVE_METRICS = (
    ROOT
    / "results_linda_additive_compare_all_effects"
    / "linda_metrics_summary_all_effects.csv"
)
LINDA_ADDITIVE_TAXA = (
    ROOT
    / "results_linda_additive_compare_all_effects"
    / "linda_taxon_level_results_all_effects.csv"
)
MAASLIN_TAXA = (
    ROOT
    / "results_maaslin2_compare_all_effects"
    / "maaslin2_taxon_truth_vs_prediction_all_effects.csv"
)
OUT = Path(os.environ.get("FIG4_OUTPUT_DIR", str(SCRIPT_DIR))).resolve()
OUT.mkdir(parents=True, exist_ok=True)

MOT = "#168C80"
CTF_COLOR = "#5A3D8C"
ZEROED = "#8A8A8A"
PURPLE_ACCENT = "#6F5A8D"
GOLD_ACCENT = "#D49A3A"
HEAT_PURPLE = "#2C004B"
HEAT_GOLD = "#FDBF6F"
HEAT_NEUTRAL = "#F7F2E8"
GRID = "#D9D7D3"
STRIP = "#EEEAE3"
TEXT = "#222222"
MOT_LINEWIDTH = 2.7
COMPARATOR_LINEWIDTH = 1.2
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_purple_cream_gold", [HEAT_PURPLE, HEAT_NEUTRAL, HEAT_GOLD]
)
PROP_CMAP = LinearSegmentedColormap.from_list(
    "proportion_gold_purple", [HEAT_GOLD, HEAT_PURPLE]
)

METHOD_ORDER = ["LinDA", "MaAsLin2", "ZIGMM", "glmmPQL"]
CONDITION_ORDER = ["zeroed", "imputed"]
CONDITION_LABEL = {"zeroed": "Random masking", "imputed": "MOT-(2D)$^2$ PCA"}
CONDITION_COLOR = {"zeroed": ZEROED, "imputed": MOT}

# Archived helpers from the predecessor A-F assembly remain below only for
# provenance. They are never invoked by main(); Figure 4 contains exactly A-E.
CORE_METHOD_LABELS = {
    "ctf": "CTF", "equal": "Equal", "glv_mse_legacy": "GLV-MSE",
    "glv_ridge_legacy": "GLV-ridge", "knn_epanechnikov_legacy": "kNN-Epanechnikov",
    "last": "Last value", "mean": "Mean", "median": "Median",
    "penalized_2d2_pca": "MOT-(2D)$^2$ PCA", "spline_legacy": "Spline",
    "temporal_knn": "Temporal kNN", "weighted_avg_legacy": "Weighted average",
    "weighted_time_average": "Weighted time average",
}
CORE_METHOD_COLORS = {
    "ctf": CTF_COLOR, "equal": "#536878", "glv_mse_legacy": "#D4875C",
    "glv_ridge_legacy": "#78967A", "knn_epanechnikov_legacy": "#B76E79",
    "last": "#8873B3", "mean": "#A77B55", "median": "#B9789B",
    "penalized_2d2_pca": MOT, "spline_legacy": "#6E7F8D",
    "temporal_knn": "#B99A3E", "weighted_avg_legacy": "#69A7C4",
    "weighted_time_average": "#526FA5",
}

def set_publication_style() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "lines.linewidth": 1.35,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: mpl.axes.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Publication layout: coordinate ticks and spines carry the scale; faint
    # background grids are intentionally omitted to reduce visual clutter.
    ax.grid(False, which="both")


def panel_content(
    fig: mpl.figure.Figure,
    spec,
    letter: str,
    title: str,
    header_fraction: float = 0.13,
    header_gap: float = 0.025,
):
    layout = spec.subgridspec(
        2,
        1,
        height_ratios=[header_fraction, 1.0 - header_fraction],
        hspace=header_gap,
    )
    header = fig.add_subplot(layout[0])
    header.set_axis_off()
    header.text(
        0.0,
        0.46,
        letter,
        ha="left",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="black",
    )
    header.text(
        0.040,
        0.46,
        title,
        ha="left",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color=TEXT,
    )
    return layout[1]


def load_var(path: Path, variable: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        return ds[variable].load()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def parse_pnum(text: str, prefix: str) -> float:
    return float(text.removeprefix(prefix).replace("p", "."))


def prepare_numeric_recovery() -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | str | int]] = []
    heatmap_payload: dict[str, object] = {}
    effect_dirs = sorted(
        ROOT.glob("group_effect_*"),
        key=lambda p: parse_pnum(p.name, "group_effect_"),
    )

    for effect_dir in effect_dirs:
        effect = parse_pnum(effect_dir.name, "group_effect_")
        truth = load_var(effect_dir / "truth" / "truth_effect.nc", "y_true")
        design = pd.read_csv(effect_dir / "truth" / "effect_pseudo_group.csv")
        taxon_truth = pd.read_csv(effect_dir / "truth" / "effect_taxon_effect.csv")
        diff_taxa = taxon_truth.loc[
            taxon_truth["is_differential"].astype(bool), "taxon"
        ].tolist()
        cd = design.loc[design["pseudo_group"].eq("CD"), "participant"].tolist()
        noncd = design.loc[
            design["pseudo_group"].eq("nonCD"), "participant"
        ].tolist()

        truth_diff = truth.sel(taxon=diff_taxa)
        contrast_true_da = truth_diff.sel(participant=cd).mean("participant") - truth_diff.sel(
            participant=noncd
        ).mean("participant")
        contrast_true = contrast_true_da.values.astype(float)

        frac_dirs = sorted(
            effect_dir.glob("frac_*"), key=lambda p: parse_pnum(p.name, "frac_")
        )
        for frac_dir in frac_dirs:
            frac_name = frac_dir.name.removeprefix("frac_")
            frac = float(frac_name.replace("p", "."))
            mask = load_var(frac_dir / f"mask_frac_{frac_name}.nc", "mask")
            conditions = {
                "imputed": load_var(
                    frac_dir / f"imputed_penalized_2d2_pca_frac_{frac_name}.nc",
                    "y_imputed",
                ),
                "zeroed": load_var(
                    frac_dir / f"zeroed_frac_{frac_name}.nc", "y_zeroed"
                ),
            }

            for condition, pred in conditions.items():
                truth_values = truth.values.astype(float)
                pred_values = pred.values.astype(float)
                missing = mask.values.astype(bool)
                err = pred_values - truth_values

                pred_diff = pred.sel(taxon=diff_taxa)
                contrast_pred_da = pred_diff.sel(participant=cd).mean(
                    "participant"
                ) - pred_diff.sel(participant=noncd).mean("participant")
                contrast_pred = contrast_pred_da.values.astype(float)
                contrast_error = contrast_pred - contrast_true

                rows.append(
                    {
                        "effect_size": effect,
                        "masking_fraction": frac,
                        "condition": condition,
                        "n_diff_taxa": len(diff_taxa),
                        "rmse_all": float(np.sqrt(np.nanmean(err**2))),
                        "rmse_masked": float(np.sqrt(np.nanmean(err[missing] ** 2))),
                        "pearson_all": safe_corr(truth_values, pred_values),
                        "contrast_rmse": float(
                            np.sqrt(np.nanmean(contrast_error**2))
                        ),
                        "contrast_corr": safe_corr(contrast_true, contrast_pred),
                    }
                )

                if np.isclose(effect, 0.6) and np.isclose(frac, 0.7):
                    heatmap_payload[condition] = contrast_pred.copy()

            if np.isclose(effect, 0.6) and np.isclose(frac, 0.7):
                order = np.argsort(np.nanmean(contrast_true, axis=1))
                heatmap_payload.update(
                    {
                        "truth": contrast_true.copy(),
                        "order": order,
                        "bin_labels": [str(x) for x in truth.coords["bin"].values],
                        "taxon_labels": [f"T{i + 1:02d}" for i in range(len(diff_taxa))],
                        "effect_size": effect,
                        "masking_fraction": frac,
                    }
                )

    numeric = pd.DataFrame(rows).sort_values(
        ["effect_size", "masking_fraction", "condition"]
    )
    numeric.to_csv(OUT / "panel_AB_numeric_recovery_metrics.csv", index=False)
    return numeric, heatmap_payload


def load_additive_combined_metrics() -> pd.DataFrame:
    """Replace only LinDA rows in the legacy four-method table.

    The method is still LinDA; "additive" refers to the corrected simulation
    effect construction, not to a renamed inference engine.
    """
    legacy = pd.read_csv(SOURCE_FIG_DIR / "combined_metrics_clean.csv")
    legacy = legacy.loc[~legacy["da_method"].eq("LinDA")].copy()

    linda = pd.read_csv(LINDA_ADDITIVE_METRICS)
    additive = pd.DataFrame(
        {
            "da_method": "LinDA",
            "effect_name": linda["effect_name"],
            "effect_size": linda["effect_level"].astype(float),
            "masking_fraction": (
                linda["frac"]
                .str.removeprefix("frac_")
                .str.replace("p", ".", regex=False)
                .astype(float)
            ),
            "data_condition": linda["method"],
            "TP": linda["TP"],
            "FP": linda["FP"],
            "FN": linda["FN"],
            "TN": linda["TN"],
            "precision": linda["precision"],
            "recall": linda["recall"],
            "F1": linda["F1"],
            "specificity": linda["specificity"],
            "accuracy": linda["accuracy"],
            "FDR": linda["FDR"],
            "n_called_diff": linda["n_called_diff"],
            "n_truth_diff": linda["n_truth_diff"],
            "n_taxa_total": linda["n_taxa_total"],
        }
    )
    combined = pd.concat([legacy, additive], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["da_method", "effect_size", "masking_fraction", "data_condition"]
    )
    combined.to_csv(OUT / "combined_metrics_clean_additive_linda.csv", index=False)
    return combined


def prepare_da_deltas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_additive_combined_metrics()
    data["F1_corrected"] = (
        2.0 * data["TP"] / (2.0 * data["TP"] + data["FP"] + data["FN"])
    )
    keys = ["da_method", "effect_size", "masking_fraction"]
    values = [
        "F1_corrected",
        "recall",
        "FDR",
        "TP",
        "FP",
        "FN",
        "precision",
    ]
    wide = data.pivot(index=keys, columns="data_condition", values=values)
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()

    wide["delta_F1"] = wide["F1_corrected_imputed"] - wide["F1_corrected_zeroed"]
    wide["delta_recall"] = wide["recall_imputed"] - wide["recall_zeroed"]
    wide["delta_FDR"] = wide["FDR_imputed"] - wide["FDR_zeroed"]
    wide["benefit_FDR"] = -wide["delta_FDR"]
    wide["delta_TP"] = wide["TP_imputed"] - wide["TP_zeroed"]
    wide["delta_FP"] = wide["FP_imputed"] - wide["FP_zeroed"]
    wide["delta_FN"] = wide["FN_imputed"] - wide["FN_zeroed"]
    wide["benefit_FP"] = -wide["delta_FP"]
    wide["benefit_FN"] = -wide["delta_FN"]
    wide["delta_precision"] = wide["precision_imputed"] - wide["precision_zeroed"]
    wide["da_method"] = pd.Categorical(
        wide["da_method"], categories=METHOD_ORDER, ordered=True
    )
    wide = wide.sort_values(["da_method", "effect_size", "masking_fraction"])
    wide.to_csv(OUT / "panel_CDE_corrected_DA_deltas.csv", index=False)

    summary_rows = []
    metric_defs = [
        ("Delta F1", "delta_F1"),
        ("Delta Recall", "delta_recall"),
        ("FDR reduction", "benefit_FDR"),
    ]
    for method in METHOD_ORDER:
        subset = wide.loc[wide["da_method"].eq(method)]
        for label, column in metric_defs:
            values_finite = subset[column].dropna().to_numpy(float)
            q25, median, q75 = np.quantile(values_finite, [0.25, 0.50, 0.75])
            summary_rows.append(
                {
                    "da_method": method,
                    "metric": label,
                    "n_pairs": len(values_finite),
                    "q25": q25,
                    "median": median,
                    "q75": q75,
                }
            )
    metric_summary = pd.DataFrame(summary_rows)
    metric_summary.to_csv(OUT / "panel_D_metric_summaries.csv", index=False)

    improvement_defs = [
        ("F1 gain", "delta_F1"),
        ("Recall gain", "delta_recall"),
        ("FDR reduction", "benefit_FDR"),
        ("TP gain", "delta_TP"),
        ("FP reduction", "benefit_FP"),
        ("FN reduction", "benefit_FN"),
    ]
    improvement_rows = []
    for method in METHOD_ORDER:
        subset = wide.loc[wide["da_method"].eq(method)]
        for label, column in improvement_defs:
            values_finite = subset[column].dropna().to_numpy(float)
            improvement_rows.append(
                {
                    "da_method": method,
                    "outcome": label,
                    "n_evaluable": len(values_finite),
                    "n_improved": int(np.sum(values_finite > 0)),
                    "proportion_improved": float(np.mean(values_finite > 0)),
                }
            )
    improvement = pd.DataFrame(improvement_rows)
    improvement.to_csv(OUT / "panel_E_improvement_proportions.csv", index=False)
    return wide, metric_summary, improvement


def prepare_consensus() -> pd.DataFrame:
    """Truth composition of LinDA/MaAsLin2 overlap across 99 settings."""
    linda = pd.read_csv(
        LINDA_ADDITIVE_TAXA,
        usecols=[
            "taxon", "effect_name", "frac", "method", "significant",
            "is_true_diff",
        ],
    ).rename(
        columns={
            "effect_name": "effect",
            "significant": "called_linda",
            "is_true_diff": "truth",
        }
    )
    maas = pd.read_csv(
        MAASLIN_TAXA,
        usecols=["taxon", "effect", "frac", "method", "predicted"],
    ).rename(columns={"predicted": "called_maaslin"})
    paired = linda.merge(maas, on=["taxon", "effect", "frac", "method"])
    for column in ["called_linda", "called_maaslin", "truth"]:
        paired[column] = paired[column].astype(bool)

    rows: list[dict[str, object]] = []
    class_defs = [
        ("Shared", paired["called_linda"] & paired["called_maaslin"]),
        ("LinDA only", paired["called_linda"] & ~paired["called_maaslin"]),
        ("MaAsLin2 only", paired["called_maaslin"] & ~paired["called_linda"]),
    ]
    for condition in CONDITION_ORDER:
        in_condition = paired["method"].eq(condition)
        for discovery_class, mask in class_defs:
            selected = paired.loc[in_condition & mask]
            n_calls = len(selected)
            n_true = int(selected["truth"].sum())
            rows.append(
                {
                    "data_condition": condition,
                    "discovery_class": discovery_class,
                    "n_calls": n_calls,
                    "n_true": n_true,
                    "n_false": n_calls - n_true,
                    "true_fraction": n_true / n_calls if n_calls else np.nan,
                }
            )
    consensus = pd.DataFrame(rows)
    consensus.to_csv(OUT / "panel_E_consensus_truth_composition.csv", index=False)
    return consensus


def prepare_core_benchmark() -> pd.DataFrame:
    core = pd.read_csv(CORE_CSV / "errorbar_summary.csv")
    ctf = pd.read_csv(CTF_CORE_CSV / "errorbar_summary.csv")
    if set(ctf["method"].astype(str).unique()) != {"ctf"}:
        raise ValueError("Unexpected method labels in sim3 CTF errorbar summary")
    core = pd.concat([core, ctf], ignore_index=True, sort=False)
    core = core.loc[
        core["scope"].eq("AllCells")
        & core["metric"].isin(["mse_after", "pearson_after"])
    ].copy()
    core["frac"] = core["frac"].astype(float)
    core.to_csv(OUT / "panel_A_global_benchmark_with_CTF.csv", index=False)
    return core


def prepare_core_ctf_supplement() -> pd.DataFrame:
    """MOT-versus-CTF summaries for both scopes and all core metrics."""
    core = pd.read_csv(CORE_CSV / "errorbar_summary.csv")
    ctf = pd.read_csv(CTF_CORE_CSV / "errorbar_summary.csv")
    data = pd.concat([core, ctf], ignore_index=True, sort=False)
    data = data.loc[
        data["method"].isin(["penalized_2d2_pca", "ctf"])
        & data["scope"].isin(["AllCells", "InjectedCells"])
        & data["metric"].isin(["mse_after", "pearson_after", "wd_after"])
    ].copy()
    data["frac"] = data["frac"].astype(float)
    data.to_csv(OUT / "supplementary_MOT_CTF_core_summary.csv", index=False)
    return data


def plot_core_ctf_supplement(data: pd.DataFrame) -> list[Path]:
    """Create the paired core-metric supplement supporting main-panel A."""
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.7), sharex=True)
    metric_defs = [
        ("mse_after", "MSE ↓"),
        ("pearson_after", "Pearson r ↑"),
        ("wd_after", "Wasserstein distance ↓"),
    ]
    scopes = [("AllCells", "All cells"), ("InjectedCells", "Injected cells")]
    for row, (scope, scope_label) in enumerate(scopes):
        for col, (metric, label) in enumerate(metric_defs):
            ax = axes[row, col]
            for method, color, ls, marker, display in [
                ("penalized_2d2_pca", MOT, "-", "o", "MOT-(2D)$^2$ PCA"),
                ("ctf", CTF_COLOR, "--", "X", "CTF"),
            ]:
                d = data.loc[
                    data["scope"].eq(scope) & data["metric"].eq(metric)
                    & data["method"].eq(method)
                ].sort_values("frac")
                ax.plot(d["frac"], d["mean"], color=color, ls=ls, marker=marker,
                        lw=2.0, ms=4.0, label=display)
                ax.fill_between(
                    d["frac"].to_numpy(float), d["ci95_low"].to_numpy(float),
                    d["ci95_high"].to_numpy(float), color=color, alpha=0.14,
                    linewidth=0,
                )
            if metric in {"mse_after", "wd_after"}:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1.02)
            ax.set_title(f"{scope_label} · {label}", fontsize=9.0, fontweight="bold")
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
            if row == 1:
                ax.set_xlabel("Masking fraction")
            clean_axis(ax, grid_axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "Supplementary: MOT-(2D)$^2$ PCA versus CTF in heterogeneous Simulation 3",
        y=1.035, fontsize=12.0, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    paths = [
        OUT / "supplementary_MOT_CTF_core_benchmark.tiff",
        OUT / "supplementary_MOT_CTF_core_benchmark.pdf",
        OUT / "supplementary_MOT_CTF_core_benchmark.png",
    ]
    save_tiff(fig, paths[0])
    fig.savefig(paths[1], format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(paths[2], format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def legacy_plot_sim3_benchmark_not_used(
    fig: mpl.figure.Figure, spec, core: pd.DataFrame
) -> None:
    content = panel_content(
        fig, spec, "A", "Global recovery benchmark across 13 imputation methods"
    )
    grid = content.subgridspec(
        2, 2, height_ratios=[0.70, 0.30], hspace=0.22, wspace=0.32
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.set_axis_off()

    methods = list(CORE_METHOD_LABELS)
    featured_methods = {"penalized_2d2_pca", "ctf"}
    colors = CORE_METHOD_COLORS

    metric_panels = [
        ("mse_after", "MSE"),
        ("pearson_after", "Pearson r"),
    ]
    handles = []
    for ax, (metric, ylabel) in zip(axes, metric_panels):
        for method in methods:
            d = core.loc[(core["metric"].eq(metric)) & (core["method"].eq(method))]
            d = d.sort_values("frac")
            is_mot = method == "penalized_2d2_pca"
            is_ctf = method == "ctf"
            is_featured = is_mot or is_ctf
            ax.plot(
                d["frac"],
                d["mean"],
                color=colors[method],
                linewidth=MOT_LINEWIDTH if is_mot else COMPARATOR_LINEWIDTH,
                linestyle="-" if is_mot else "--",
                alpha=1.0 if is_mot else 0.78,
                marker="o" if is_mot else None,
                markersize=3.3 if is_mot else 0,
                zorder=5 if is_featured else 2,
            )
            if is_featured:
                ax.fill_between(
                    d["frac"].to_numpy(float),
                    d["ci95_low"].to_numpy(float),
                    d["ci95_high"].to_numpy(float),
                    color=colors[method],
                    alpha=0.14 if is_mot else 0.10,
                    linewidth=0,
                    zorder=3,
                )
        mot_09 = core.loc[
            core["method"].eq("penalized_2d2_pca")
            & core["metric"].eq(metric)
            & np.isclose(core["frac"], 0.9),
            "mean",
        ].iloc[0]
        ax.annotate(
            "rank 1/13",
            xy=(0.9, mot_09),
            xytext=(-31, 11 if metric == "mse_after" else -18),
            textcoords="offset points",
            fontsize=7.2,
            fontweight="bold",
            color=MOT,
            arrowprops=dict(arrowstyle="-", color=MOT, lw=0.8),
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(core["frac"].unique()))
        clean_axis(ax)

    for method in methods:
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[method],
                lw=(
                    MOT_LINEWIDTH
                    if method == "penalized_2d2_pca"
                    else COMPARATOR_LINEWIDTH
                ),
                ls="-" if method == "penalized_2d2_pca" else "--",
                marker="o" if method == "penalized_2d2_pca" else None,
                markersize=3.2,
                label=CORE_METHOD_LABELS[method],
            )
        )
    legend_ax.text(
        0.5,
        1.02,
        "Masking fraction",
        transform=legend_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.3,
    )
    legend_ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        fontsize=6.15,
        handlelength=1.55,
        columnspacing=0.80,
        handletextpad=0.38,
    )


def quantile_summary(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby(["condition", "masking_fraction"], observed=True)[metric]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
        .reset_index()
    )


def plot_condition_curve(
    ax: mpl.axes.Axes,
    summary: pd.DataFrame,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    for condition in CONDITION_ORDER:
        d = summary.loc[summary["condition"].eq(condition)].sort_values(
            "masking_fraction"
        )
        x = d["masking_fraction"].to_numpy(float)
        ax.fill_between(
            x,
            d["q25"].to_numpy(float),
            d["q75"].to_numpy(float),
            color=CONDITION_COLOR[condition],
            alpha=0.16 if condition == "imputed" else 0.12,
            linewidth=0,
        )
        ax.plot(
            x,
            d["median"].to_numpy(float),
            color=CONDITION_COLOR[condition],
            marker="o",
            markersize=3.2,
            linewidth=2.15 if condition == "imputed" else 1.8,
            label=CONDITION_LABEL[condition],
        )
    ax.set_xlabel("Masking fraction")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0.1, 1.0, 0.2))
    if ylim is not None:
        ax.set_ylim(*ylim)
    clean_axis(ax)


def plot_panel_a(fig: mpl.figure.Figure, spec, numeric: pd.DataFrame) -> None:
    content = panel_content(
        fig, spec, "A", "Global numerical recovery across effect sizes"
    )
    grid = content.subgridspec(1, 2, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    rmse = quantile_summary(numeric, "rmse_all")
    pearson = quantile_summary(numeric, "pearson_all")
    plot_condition_curve(axes[0], rmse, "RMSE")
    plot_condition_curve(axes[1], pearson, "Pearson r", ylim=(0.05, 1.02))
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        ncol=2,
        frameon=False,
        fontsize=7.7,
        handlelength=2.0,
        columnspacing=1.1,
    )


def add_heatmap_grid(ax: mpl.axes.Axes, nrows: int, ncols: int) -> None:
    # Keep the helper for consistent call sites, but do not draw cell gridlines.
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(False, which="both")
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#666666")


def plot_panel_b(
    fig: mpl.figure.Figure,
    spec,
    numeric: pd.DataFrame,
    heatmap: dict[str, object],
) -> None:
    content = panel_content(
        fig,
        spec,
        "B",
        "Differential-effect trajectories (effect = 0.60; masking = 0.70)",
        header_fraction=0.17,
        header_gap=0.07,
    )
    grid = content.subgridspec(
        2,
        4,
        height_ratios=[0.61, 0.39],
        width_ratios=[1, 1, 1, 0.065],
        hspace=0.80,
        wspace=0.18,
    )
    heat_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    cax = fig.add_subplot(grid[0, 3])
    curve_ax = fig.add_subplot(grid[1, :3])

    order = np.asarray(heatmap["order"], dtype=int)
    arrays = [
        np.asarray(heatmap["truth"])[order],
        np.asarray(heatmap["zeroed"])[order],
        np.asarray(heatmap["imputed"])[order],
    ]
    titles = ["Truth", "Random masking", "MOT-(2D)$^2$ PCA"]
    vmax = float(max(np.nanmax(np.abs(x)) for x in arrays))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = None
    bin_labels = list(heatmap["bin_labels"])
    taxon_labels = np.asarray(heatmap["taxon_labels"])[order]
    xticks = [0, 3, 6, 9]
    yticks = [0, 5, 10, 15, 20]
    for idx, (ax, values, title) in enumerate(zip(heat_axes, arrays, titles)):
        image = ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=DELTA_CMAP,
            norm=norm,
            origin="lower",
        )
        ax.set_title(
            title,
            fontsize=8.2,
            fontweight="bold",
            pad=3,
            bbox=dict(facecolor=STRIP, edgecolor="#777777", linewidth=0.55, pad=2.0),
        )
        ax.set_xticks(
            xticks,
            [bin_labels[i] for i in xticks],
            rotation=90,
            ha="center",
            va="top",
        )
        ax.tick_params(axis="x", pad=1)
        ax.set_yticks(yticks)
        if idx == 0:
            ax.set_yticklabels(taxon_labels[yticks])
            ax.set_ylabel("Differential taxa")
        else:
            ax.set_yticklabels([])
        add_heatmap_grid(ax, values.shape[0], values.shape[1])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("CD - nonCD\n(log10 scale)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=8.0)
    corr_summary = quantile_summary(numeric, "contrast_corr")
    plot_condition_curve(
        curve_ax,
        corr_summary,
        "Correlation with truth",
        ylim=(0.30, 1.015),
    )
    curve_ax.set_title(
        "Trajectory correlation with truth: median (IQR) across effect sizes",
        fontsize=8.0,
        pad=2,
    )
    curve_ax.legend(loc="lower left", frameon=False, ncol=2)


def plot_panel_c(fig: mpl.figure.Figure, spec, delta: pd.DataFrame) -> None:
    content = panel_content(
        fig,
        spec,
        "C",
        r"Corrected $\Delta$F1 across all effect and masking settings",
        header_fraction=0.17,
        header_gap=0.07,
    )
    grid = content.subgridspec(
        2, 3, width_ratios=[1, 1, 0.055], hspace=0.30, wspace=0.24
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    cax = fig.add_subplot(grid[:, 2])
    limit = float(np.nanmax(np.abs(delta["delta_F1"])))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None

    effects = sorted(delta["effect_size"].unique())
    fracs = sorted(delta["masking_fraction"].unique())
    effect_ticks_values = [0.5, 0.6, 0.7, 0.8]
    effect_ticks = [int(np.argmin(np.abs(np.asarray(effects) - x))) for x in effect_ticks_values]
    frac_ticks_values = [0.1, 0.5, 0.9]
    frac_ticks = [int(np.argmin(np.abs(np.asarray(fracs) - x))) for x in frac_ticks_values]

    for idx, (ax, method) in enumerate(zip(axes, METHOD_ORDER)):
        d = delta.loc[delta["da_method"].eq(method)]
        matrix = (
            d.pivot(
                index="masking_fraction", columns="effect_size", values="delta_F1"
            )
            .reindex(index=fracs, columns=effects)
            .to_numpy(float)
        )
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=DELTA_CMAP,
            norm=norm,
        )
        ax.set_title(
            method,
            fontsize=8.2,
            fontweight="bold",
            pad=3,
            bbox=dict(facecolor=STRIP, edgecolor="#777777", linewidth=0.55, pad=2.1),
        )
        ax.set_xticks(effect_ticks, [f"{x:g}" for x in effect_ticks_values])
        ax.set_yticks(frac_ticks, [f"{x:g}" for x in frac_ticks_values])
        if idx < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Effect size")
        if idx % 2 == 0:
            ax.set_ylabel("Masking fraction")
        else:
            ax.set_yticklabels([])
        add_heatmap_grid(ax, len(fracs), len(effects))
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$\Delta$F1", fontsize=8.0)
    cbar.ax.tick_params(labelsize=8.0)


def plot_panel_d(
    fig: mpl.figure.Figure, spec, metric_summary: pd.DataFrame
) -> None:
    content = panel_content(
        fig,
        spec,
        "D",
        "Median (IQR) downstream change",
        header_fraction=0.17,
        header_gap=0.07,
    )
    # A shallow spacer keeps the three facet strips clear of the panel heading.
    grid = content.subgridspec(
        2, 3, height_ratios=[0.10, 0.90], hspace=0.0, wspace=0.28
    )
    axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    metric_order = ["Delta F1", "Delta Recall", "FDR reduction"]
    titles = [r"$\Delta$F1", r"$\Delta$Recall", r"$-\Delta$FDR"]
    colors = [MOT, PURPLE_ACCENT, GOLD_ACCENT]
    y = np.arange(len(METHOD_ORDER))

    for idx, (ax, metric, title, color) in enumerate(
        zip(axes, metric_order, titles, colors)
    ):
        d = (
            metric_summary.loc[metric_summary["metric"].eq(metric)]
            .set_index("da_method")
            .reindex(METHOD_ORDER)
        )
        ax.hlines(y, d["q25"], d["q75"], color=color, linewidth=3.2, alpha=0.48)
        ax.scatter(
            d["median"],
            y,
            s=30,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.axvline(0, color="#333333", linewidth=0.85, zorder=1)
        ax.set_title(
            title,
            fontsize=8.1,
            fontweight="bold",
            pad=4,
            bbox=dict(facecolor=STRIP, edgecolor="#777777", linewidth=0.55, pad=2.0),
        )
        ax.set_xlim(-0.65, 1.02)
        ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(METHOD_ORDER)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel("Improvement")
        clean_axis(ax, grid_axis="x")


def plot_panel_e(
    fig: mpl.figure.Figure, spec, consensus: pd.DataFrame
) -> None:
    content = panel_content(
        fig,
        spec,
        "E",
        "Cross-method overlap enriches true positives (LinDA vs MaAsLin2)",
        header_fraction=0.30,
        header_gap=0.12,
    )
    grid = content.subgridspec(1, 2, wspace=0.26)
    class_order = ["Shared", "LinDA only", "MaAsLin2 only"]
    colors = [MOT, "#E69F00", "#0072B2"]
    labels = ["Shared", "LinDA\nonly", "MaAsLin2\nonly"]
    titles = {"zeroed": "Random masking", "imputed": "MOT-(2D)$^2$ PCA"}

    for idx, condition in enumerate(CONDITION_ORDER):
        ax = fig.add_subplot(grid[0, idx])
        d = (
            consensus.loc[consensus["data_condition"].eq(condition)]
            .set_index("discovery_class")
            .reindex(class_order)
        )
        x = np.arange(len(class_order))
        bars = ax.bar(
            x,
            d["true_fraction"],
            color=colors,
            edgecolor="#333333",
            linewidth=0.55,
            width=0.68,
        )
        for bar, (_, row) in zip(bars, d.iterrows()):
            value = float(row["true_fraction"])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.035, 1.055),
                f"{value:.1%}\nn={int(row['n_calls']):,}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold" if row.name == "Shared" else "normal",
            )
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x, labels)
        ax.set_title(titles[condition], fontsize=8.4, fontweight="bold", pad=4)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_yticks([0, 0.5, 1.0])
        if idx == 0:
            ax.set_ylabel("True-positive fraction")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Discovery class")
        clean_axis(ax, grid_axis="y")


def save_tiff(fig: mpl.figure.Figure, path: Path, dpi: int = 600) -> None:
    # Render with Matplotlib first, then write lossless DEFLATE-compressed RGB TIFF.
    # tifffile avoids a native Pillow/libtiff crash seen in some Windows builds.
    temporary_png = path.with_name(f"_{path.stem}_temporary.png")
    fig.savefig(
        temporary_png,
        format="png",
        dpi=dpi,
        bbox_inches=None,
        facecolor="white",
    )
    with Image.open(temporary_png) as image:
        image.convert("RGB").save(
            path,
            format="TIFF",
            compression="tiff_deflate",
            dpi=(dpi, dpi),
        )
    temporary_png.unlink()


def render_individual(
    letter: str,
    filename: str,
    figsize: tuple[float, float],
    draw: Callable,
    *args,
) -> Path:
    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(1, 1, left=0.08, right=0.98, bottom=0.09, top=0.98)[0]
    draw(fig, spec, *args)
    path = OUT / filename
    save_tiff(fig, path)
    fig.savefig(path.with_suffix(".png"), format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", dpi=600,
                bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return path


def verify_tiffs(paths: list[Path]) -> None:
    rows = []
    for path in paths:
        with Image.open(path) as image:
            dpi_info = image.info.get("dpi", (np.nan, np.nan))
            x_dpi, y_dpi = float(dpi_info[0]), float(dpi_info[1])
            rows.append(
                {
                    "file": path.name,
                    "width_px": image.width,
                    "height_px": image.height,
                    "mode": "RGB",
                    "dpi_x": x_dpi,
                    "dpi_y": y_dpi,
                    "compression": str(image.info.get("compression", "tiff_deflate")),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "tiff_output_verification.csv", index=False)


def main() -> None:
    set_publication_style()
    numeric, heatmap = prepare_numeric_recovery()
    delta, metric_summary, improvement = prepare_da_deltas()
    consensus = prepare_consensus()

    # Exact ISO A4 portrait dimensions. Saving without bbox_inches="tight"
    # preserves the 210 x 297 mm page and yields 4961 x 7016 pixels at 600 dpi.
    a4_width = 210.0 / 25.4
    a4_height = 297.0 / 25.4
    fig = plt.figure(figsize=(a4_width, a4_height))
    outer = fig.add_gridspec(
        5,
        1,
        left=0.075,
        right=0.910,
        bottom=0.060,
        top=0.985,
        height_ratios=[1.15, 2.10, 2.20, 1.25, 1.25],
        hspace=0.32,
    )
    plot_panel_a(fig, outer[0, 0], numeric)
    plot_panel_b(fig, outer[1, 0], numeric, heatmap)
    plot_panel_c(fig, outer[2, 0], delta)
    plot_panel_d(fig, outer[3, 0], metric_summary)
    plot_panel_e(fig, outer[4, 0], consensus)

    stem = "Figure4_revised_additive_LinDA"
    composite_tiff = OUT / f"{stem}.tiff"
    composite_pdf = OUT / f"{stem}.pdf"
    composite_png = OUT / f"{stem}.png"
    save_tiff(fig, composite_tiff)
    fig.savefig(
        composite_pdf,
        format="pdf",
        bbox_inches=None,
        facecolor="white",
    )
    fig.savefig(
        composite_png,
        format="png",
        dpi=600,
        bbox_inches=None,
        facecolor="white",
    )
    plt.close(fig)

    (OUT / "Figure4_caption.txt").write_text(
        "Figure 4. Numerical and downstream differential-abundance recovery in the "
        "effect-size sweep. Randomly selected entries in the pseudo-truth tensor were "
        "set to zero before imputation. (A) RMSE and Pearson correlation for the "
        "random-masking input and MOT-(2D)^2 PCA-imputed tensors; curves show medians "
        "and shaded bands show interquartile "
        "ranges across 11 effect sizes. (B) Differential-effect trajectories for the "
        "representative effect = 0.60 and masking fraction = 0.70 setting, with the "
        "lower curve summarizing trajectory correlation across effect sizes. (C) "
        "Corrected Delta F1 for four DA methods across all 11 effect sizes and nine "
        "masking fractions; positive values favor imputation. (D) Median and IQR of "
        "Delta F1, Delta Recall, and FDR reduction across parameter-grid settings. "
        "(E) Truth composition of shared and method-specific LinDA/MaAsLin2 discoveries, "
        "aggregated as taxon-grid calls across 99 settings. Shared calls were 100.0% "
        "true positives under random masking and 99.4% after MOT completion. LinDA uses "
        "the corrected additive-effect simulation results; the method name remains LinDA. "
        "F1 was computed directly from TP, FP, and FN counts. "
        "The grid "
        "represents one fixed simulation realization with a common mask family and "
        "should not be interpreted as Monte Carlo replicates.\n",
        encoding="utf-8",
    )

    verify_tiffs([composite_tiff])
    print(f"Outputs written to: {OUT}")
    for path in [composite_pdf, composite_png, composite_tiff]:
        print(path.name)


if __name__ == "__main__":
    main()
