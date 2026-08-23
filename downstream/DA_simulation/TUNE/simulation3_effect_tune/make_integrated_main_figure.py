"""Build the integrated 2 x 3 publication figure for Simulation 3.

Outputs
-------
Six standalone panels (A-F) in PDF plus 600-dpi PNG and compressed TIFF,
plus a 2 x 3 composite in the same formats. Derived numerical summaries
are also written as CSV files for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import sys

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
import tifffile
import xarray as xr


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
CORE_CSV = REPO_ROOT / "sim3_compare" / "core_csv"
CTF_CORE_CSV = REPO_ROOT / "CTF_artifacts" / "sim3"
SOURCE_FIG_DIR = ROOT / "publication_figures_imputation_DA"
OUT = SOURCE_FIG_DIR / "integrated_main_figure"
OUT.mkdir(parents=True, exist_ok=True)

MOT = "#168C80"
CTF_COLOR = "#5A3D8C"
ZEROED = "#8A8A8A"
TRUTH = "#394B59"
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
CONDITION_LABEL = {"zeroed": "Zeroed", "imputed": "MOT-(2D)$^2$ PCA"}
CONDITION_COLOR = {"zeroed": ZEROED, "imputed": MOT}

CORE_METHOD_LABELS = {
    "ctf": "CTF",
    "equal": "Equal",
    "glv_mse_legacy": "GLV-MSE",
    "glv_ridge_legacy": "GLV-ridge",
    "knn_epanechnikov_legacy": "kNN-Epanechnikov",
    "last": "Last value",
    "mean": "Mean",
    "median": "Median",
    "penalized_2d2_pca": "MOT-(2D)$^2$ PCA",
    "spline_legacy": "Spline",
    "temporal_knn": "Temporal kNN",
    "weighted_avg_legacy": "Weighted average",
    "weighted_time_average": "Weighted time average",
}

CORE_METHOD_COLORS = {
    "ctf": CTF_COLOR,
    "equal": "#536878",
    "glv_mse_legacy": "#D4875C",
    "glv_ridge_legacy": "#78967A",
    "knn_epanechnikov_legacy": "#B76E79",
    "last": "#8873B3",
    "mean": "#A77B55",
    "median": "#B9789B",
    "penalized_2d2_pca": MOT,
    "spline_legacy": "#6E7F8D",
    "temporal_knn": "#B99A3E",
    "weighted_avg_legacy": "#69A7C4",
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
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
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
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.75)


def panel_content(fig: mpl.figure.Figure, spec, letter: str, title: str):
    layout = spec.subgridspec(2, 1, height_ratios=[0.105, 0.895], hspace=0.015)
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
        0.075,
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
    numeric.to_csv(OUT / "panel_BC_numeric_recovery_metrics.csv", index=False)
    return numeric, heatmap_payload


def prepare_da_deltas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = SOURCE_FIG_DIR / "combined_metrics_clean.csv"
    data = pd.read_csv(src)
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
    wide.to_csv(OUT / "panel_DEF_corrected_DA_deltas.csv", index=False)

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
    metric_summary.to_csv(OUT / "panel_E_metric_summaries.csv", index=False)

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
    improvement.to_csv(OUT / "panel_F_improvement_proportions.csv", index=False)
    return wide, metric_summary, improvement


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


def plot_panel_a(fig: mpl.figure.Figure, spec, core: pd.DataFrame) -> None:
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


def plot_panel_b(fig: mpl.figure.Figure, spec, numeric: pd.DataFrame) -> None:
    content = panel_content(
        fig, spec, "B", "Global numerical recovery in the DA effect sweep"
    )
    grid = content.subgridspec(1, 2, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    rmse = quantile_summary(numeric, "rmse_all")
    pearson = quantile_summary(numeric, "pearson_all")
    plot_condition_curve(axes[0], rmse, "RMSE")
    plot_condition_curve(axes[1], pearson, "Pearson r", ylim=(0.05, 1.02))
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", frameon=False)


def add_heatmap_grid(ax: mpl.axes.Axes, nrows: int, ncols: int) -> None:
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.82)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#666666")


def plot_panel_c(
    fig: mpl.figure.Figure,
    spec,
    numeric: pd.DataFrame,
    heatmap: dict[str, object],
) -> None:
    content = panel_content(
        fig, spec, "C", "Differential-effect trajectory preservation"
    )
    grid = content.subgridspec(
        2,
        4,
        height_ratios=[0.61, 0.39],
        width_ratios=[1, 1, 1, 0.065],
        hspace=0.54,
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
    titles = ["Truth", "Zeroed", "MOT-(2D)$^2$ PCA"]
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
            fontsize=7.8,
            fontweight="bold",
            pad=3,
            bbox=dict(facecolor=STRIP, edgecolor="#777777", linewidth=0.55, pad=2.0),
        )
        ax.set_xticks(xticks, [bin_labels[i] for i in xticks], rotation=35, ha="right")
        ax.set_yticks(yticks)
        if idx == 0:
            ax.set_yticklabels(taxon_labels[yticks])
            ax.set_ylabel("Differential taxa")
        else:
            ax.set_yticklabels([])
        add_heatmap_grid(ax, values.shape[0], values.shape[1])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("CD - nonCD\n(log10 scale)", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5)
    corr_summary = quantile_summary(numeric, "contrast_corr")
    plot_condition_curve(
        curve_ax,
        corr_summary,
        "Correlation with truth",
        ylim=(0.30, 1.015),
    )
    curve_ax.set_title(
        "Trajectory correlation with truth: median (IQR) across effect sizes",
        fontsize=7.7,
        pad=2,
    )
    curve_ax.legend(loc="lower left", frameon=False, ncol=2)


def plot_panel_d(fig: mpl.figure.Figure, spec, delta: pd.DataFrame) -> None:
    content = panel_content(
        fig, spec, "D", r"Corrected $\Delta$F1 across effect and masking settings"
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
    cbar.set_label(r"$\Delta$F1", fontsize=7.1)
    cbar.ax.tick_params(labelsize=6.7)


def plot_panel_e(
    fig: mpl.figure.Figure, spec, metric_summary: pd.DataFrame
) -> None:
    content = panel_content(
        fig, spec, "E", "Paired downstream improvement by DA method"
    )
    grid = content.subgridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
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
    axes[1].text(
        0.5,
        0.02,
        "positive = MOT better",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#666666",
    )


def plot_panel_f(
    fig: mpl.figure.Figure, spec, improvement: pd.DataFrame
) -> None:
    content = panel_content(
        fig, spec, "F", "Mechanism: fraction of parameter-grid cells improved"
    )
    grid = content.subgridspec(1, 2, width_ratios=[1, 0.045], wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    outcome_order = [
        "F1 gain",
        "Recall gain",
        "FDR reduction",
        "TP gain",
        "FP reduction",
        "FN reduction",
    ]
    matrix = (
        improvement.pivot(
            index="da_method", columns="outcome", values="proportion_improved"
        )
        .reindex(index=METHOD_ORDER, columns=outcome_order)
        .to_numpy(float)
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=PROP_CMAP,
        norm=Normalize(vmin=0, vmax=1),
    )
    xlabels = ["F1\ngain", "Recall\ngain", "FDR\nreduction", "TP\ngain", "FP\nreduction", "FN\nreduction"]
    ax.set_xticks(np.arange(len(outcome_order)), xlabels)
    ax.set_yticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER)
    ax.set_xlabel("Strict improvement relative to zeroed data")
    ax.set_ylabel("DA method")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=7.3,
                fontweight="bold",
                color="white" if value >= 0.58 else TEXT,
            )
    add_heatmap_grid(ax, matrix.shape[0], matrix.shape[1])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Grid cells improved", fontsize=7.0)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.ax.tick_params(labelsize=6.6)


def save_tiff(fig: mpl.figure.Figure, path: Path, dpi: int = 600) -> None:
    # Render with Matplotlib first, then write lossless DEFLATE-compressed RGB TIFF.
    # tifffile avoids a native Pillow/libtiff crash seen in some Windows builds.
    temporary_png = path.with_name(f"_{path.stem}_temporary.png")
    fig.savefig(
        temporary_png,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    with Image.open(temporary_png) as image:
        pixels = np.asarray(image.convert("RGB"))
    tifffile.imwrite(
        path,
        pixels,
        compression="zlib",
        compressionargs={"level": 6},
        resolution=(dpi, dpi),
        resolutionunit="INCH",
        photometric="rgb",
        metadata=None,
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
    core = prepare_core_benchmark()
    core_ctf_supp = prepare_core_ctf_supplement()
    numeric, heatmap = prepare_numeric_recovery()
    delta, metric_summary, improvement = prepare_da_deltas()

    individual_paths = [
        render_individual(
            "A",
            "panel_A_global_benchmark.tiff",
            (8.1, 4.5),
            plot_panel_a,
            core,
        ),
        render_individual(
            "B",
            "panel_B_DA_global_recovery.tiff",
            (7.3, 4.1),
            plot_panel_b,
            numeric,
        ),
        render_individual(
            "C",
            "panel_C_differential_trajectory_recovery.tiff",
            (7.6, 5.2),
            plot_panel_c,
            numeric,
            heatmap,
        ),
        render_individual(
            "D",
            "panel_D_corrected_delta_F1_heatmap.tiff",
            (7.4, 5.0),
            plot_panel_d,
            delta,
        ),
        render_individual(
            "E",
            "panel_E_paired_metric_improvement.tiff",
            (7.3, 4.4),
            plot_panel_e,
            metric_summary,
        ),
        render_individual(
            "F",
            "panel_F_improvement_grid_proportions.tiff",
            (7.0, 4.4),
            plot_panel_f,
            improvement,
        ),
    ]

    fig = plt.figure(figsize=(16.0, 10.4))
    outer = fig.add_gridspec(
        2,
        3,
        left=0.048,
        right=0.988,
        bottom=0.055,
        top=0.955,
        wspace=0.18,
        hspace=0.20,
    )
    plot_panel_a(fig, outer[0, 0], core)
    plot_panel_b(fig, outer[0, 1], numeric)
    plot_panel_c(fig, outer[0, 2], numeric, heatmap)
    plot_panel_d(fig, outer[1, 0], delta)
    plot_panel_e(fig, outer[1, 1], metric_summary)
    plot_panel_f(fig, outer[1, 2], improvement)

    fig.text(
        0.010,
        0.730,
        "TENSOR-LEVEL RECOVERY",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=MOT,
    )
    fig.text(
        0.010,
        0.285,
        "DOWNSTREAM TAXON RECOVERY",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=MOT,
    )

    composite_tiff = OUT / "publication_main_figure_integrated_2x3.tiff"
    save_tiff(fig, composite_tiff)
    fig.savefig(
        OUT / "publication_main_figure_integrated_2x3.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(
        OUT / "publication_main_figure_integrated_2x3.png",
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)

    supplementary_paths = plot_core_ctf_supplement(core_ctf_supp)
    (OUT / "integrated_main_figure_caption.txt").write_text(
        "Simulation 3 integrates tensor-level and downstream differential-"
        "abundance recovery under structured group, group-by-time, subject-slope "
        "and noise effects. Panel A compares 13 imputation methods after adding "
        "CTF as a structural strong baseline; MOT-(2D)^2 PCA has lower MSE and "
        "higher Pearson correlation than CTF across all masking fractions. "
        "Panels B-F retain the prespecified zeroed-versus-MOT downstream analysis. The "
        "simulation is evaluated in the non-reclosed transformed space and is not "
        "described as a closed relative-abundance composition.\n",
        encoding="utf-8",
    )
    supplementary_caption = (
        "Supplementary Figure. Direct comparison of MOT-(2D)^2 PCA and CTF in "
        "heterogeneous Simulation 3. Curves show means and 95% confidence "
        "intervals over 50 matched repeats for AllCells and InjectedCells. MSE and "
        "Wasserstein distance are shown on logarithmic axes and are minimized; "
        "Pearson correlation is maximized. MOT outperforms CTF for all displayed "
        "metric-by-masking conditions. Simulation 3 was not reclosed after effect "
        "injection and is interpreted as transformed-space heterogeneity "
        "robustness.\n"
    )
    (OUT / "supplementary_MOT_CTF_core_caption.txt").write_text(
        supplementary_caption,
        encoding="utf-8",
    )

    verify_tiffs(individual_paths + [composite_tiff, supplementary_paths[0]])
    print(f"Outputs written to: {OUT}")
    for path in individual_paths + [composite_tiff]:
        print(path.name)
    for path in supplementary_paths:
        print(path.name)


if __name__ == "__main__":
    main()
