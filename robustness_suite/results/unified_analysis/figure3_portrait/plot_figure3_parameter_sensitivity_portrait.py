#!/usr/bin/env python3
"""Render the portrait publication figure for MOT-(2D)^2 PCA robustness results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image
import tifffile


INK = "#111827"
MUTED = "#526273"
BLUE = "#0077BB"
SKY = "#33BBEE"
GREEN = "#009988"
ORANGE = "#EE9B00"
VERMILION = "#CC3311"
PURPLE = "#7A5195"
GRID = "#D7E0E8"
REFERENCE = {"absolute_difference_mean": 0.0, "absolute_difference_ci_low": 0.0, "absolute_difference_ci_high": 0.0}


def candidate_key(d: int, q: int, lam: float, sigma: float) -> str:
    lam_text = "0" if float(lam) == 0 else str(float(lam))
    return f"{d}|{q}|{lam_text}|{float(sigma)}"


def get_row(df: pd.DataFrame, key: str, stratum: str = "overall") -> pd.Series:
    row = df[(df.candidate == key) & (df.metric == "nrmse") & (df.stratum == stratum)]
    if len(row) != 1:
        raise ValueError(f"Expected one NRMSE row for {key!r}, {stratum!r}; found {len(row)}")
    return row.iloc[0]


def panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}   {title}", loc="left", fontsize=10.7, fontweight="bold", color=INK, pad=8)


def clean(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#3F4A56")
    ax.spines[["left", "bottom"]].set_linewidth(0.9)
    ax.tick_params(colors="#3F4A56", labelsize=8.5, length=3.5, width=0.9)
    ax.grid(axis=grid_axis, color=GRID, lw=0.75, zorder=0)


def draw_tensor(ax) -> None:
    ax.set_proj_type("ortho")
    rng = np.random.default_rng(7)
    keep = rng.random((4, 4, 4)) > 0.30
    for i, j, k in zip(*np.where(keep)):
        color = BLUE if (i + j + k) % 3 else SKY
        ax.bar3d(i + 0.05, j + 0.05, k + 0.05, 0.78, 0.78, 0.78,
                 color=color, alpha=0.82, shade=True, edgecolor="white", linewidth=0.25)
    for i, j, k in [(0, 1, 3), (2, 3, 1), (3, 0, 2), (1, 2, 0)]:
        ax.bar3d(i + 0.05, j + 0.05, k + 0.05, 0.78, 0.78, 0.78,
                 color=ORANGE, alpha=0.96, edgecolor="white", linewidth=0.25)
    ax.view_init(elev=24, azim=-57)
    ax.set(xlim=(0, 4), ylim=(0, 4), zlim=(0, 4))
    ax.set_xlabel("subjects", labelpad=-7, fontsize=7.2)
    ax.set_ylabel("taxa", labelpad=-7, fontsize=7.2)
    ax.set_zlabel("time", labelpad=-7, fontsize=7.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    # Matplotlib's 3D projection reserves unusually wide internal margins.
    # Zoom the cuboid so its visual footprint matches the neighbouring panel B.
    ax.set_box_aspect((1.15, 1.0, 0.9), zoom=1.28)
    panel_title(ax, "A", "Longitudinal microbiome tensor")


def draw_latent(ax) -> None:
    ax.set_axis_off()
    for offset, alpha in [(0.00, 0.28), (0.06, 0.48), (0.12, 0.78)]:
        rect = patches.FancyBboxPatch((0.05 + offset, 0.33 + offset * 0.25), 0.28, 0.38,
                                      boxstyle="round,pad=0.012,rounding_size=0.025",
                                      fc=SKY, ec=BLUE, lw=1.0, alpha=alpha)
        ax.add_patch(rect)
        ax.plot([0.09 + offset, 0.29 + offset], [0.50 + offset * 0.25] * 2, color=BLUE, lw=2.0)
    ax.text(0.21, 0.22, "shared subject-side\nloading axes (q)", ha="center", va="top", fontsize=7.5, color=INK)
    ax.annotate("", (0.48, 0.52), (0.37, 0.52), arrowprops=dict(arrowstyle="->", color=INK, lw=2.0))
    ax.text(0.55, 0.52, "per-time\nscore matrix", ha="center", va="center", fontsize=7.1,
            bbox=dict(boxstyle="round,pad=.30", fc="white", ec="#9DABB8", lw=0.8))
    ax.annotate("", (0.74, 0.52), (0.64, 0.52), arrowprops=dict(arrowstyle="->", color=INK, lw=2.0))
    ax.add_patch(patches.FancyBboxPatch((0.76, 0.30), 0.21, 0.45,
                                        boxstyle="round,pad=.012,rounding_size=.025",
                                        fc="#E5F4EF", ec=GREEN, lw=1.0))
    for yy in np.linspace(0.36, 0.69, 4):
        ax.plot([0.79, 0.94], [yy, yy], color=GREEN, lw=2.6, solid_capstyle="round")
    for xx in np.linspace(0.81, 0.92, 4):
        ax.plot([xx, xx], [0.34, 0.71], color=GREEN, lw=2.6, solid_capstyle="round")
    ax.text(0.865, 0.22, "shared taxon-side\nloading axes (d)", ha="center", va="top", fontsize=7.5, color=INK)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    panel_title(ax, "B", "Two-sided latent representation")


def draw_sigma(ax) -> None:
    x = np.linspace(0, 32, 400)
    for sigma, color in ((2, VERMILION), (4, ORANGE), (8, GREEN), (16, BLUE)):
        y = np.exp(-(x ** 2) / (2 * sigma ** 2))
        ax.plot(x, y, color=color, lw=2.6, label=rf"$\sigma={sigma}$")
    ax.set(xlim=(0, 32), ylim=(0, 1.08),
           xlabel=r"Temporal distance, $|t_i-t_j|$ (input time units)",
           ylabel=r"Gaussian weight, $w_{ij}$")
    ax.text(31.3, 1.035, r"$w_{ij}=\exp[-(t_i-t_j)^2/(2\sigma^2)]$", ha="right", va="top", fontsize=7.0, color=MUTED)
    ax.legend(frameon=False, ncol=2, fontsize=7.2, loc="upper right", bbox_to_anchor=(1.0, 0.88),
              handlelength=1.8, columnspacing=1.1)
    clean(ax)
    panel_title(ax, "A", "Temporal borrowing range (σ)")


def draw_lambda(ax) -> None:
    ax.set_axis_off()
    xs = np.linspace(0.10, 0.90, 5)
    upper = 0.70 + 0.07 * np.sin(np.arange(5) * 2.0)
    ax.plot(xs, upper, color="#E8A89A", lw=1.2)
    for x, y in zip(xs, upper):
        ax.add_patch(patches.Circle((x, y), 0.045, fc="#FCE9E5", ec=VERMILION, lw=1.8))
        ax.plot([x - 0.02, x + 0.02], [y, y], color=VERMILION, lw=2.2)
    ax.text(0.04, 0.88, r"$\lambda=0$: visits fitted with little temporal coupling", fontsize=7.4,
            color=VERMILION, fontweight="bold")
    path_x = np.linspace(0.10, 0.90, 200)
    path_y = 0.27 + 0.06 * np.sin(2 * np.pi * path_x)
    ax.plot(path_x, path_y, color=BLUE, lw=3.0)
    for x in xs:
        y = 0.27 + 0.06 * np.sin(2 * np.pi * x)
        ax.add_patch(patches.Circle((x, y), 0.043, fc="#E5F2FA", ec=BLUE, lw=1.8))
        ax.plot([x - 0.02, x + 0.02], [y, y], color=BLUE, lw=2.2)
    ax.text(0.04, 0.47, r"larger $\lambda$: stronger temporal alignment", fontsize=7.4,
            color=BLUE, fontweight="bold")
    ax.set(xlim=(0, 1.02), ylim=(0, 1))
    panel_title(ax, "B", "Temporal alignment (λ)")


def error_points(ax, values, labels, colors, *, ylabel=True) -> None:
    x = np.arange(len(values))
    means = np.asarray([float(v["absolute_difference_mean"]) for v in values])
    low = np.asarray([float(v["absolute_difference_ci_low"]) for v in values])
    high = np.asarray([float(v["absolute_difference_ci_high"]) for v in values])
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    for xx, mean, lo, hi, color in zip(x, means, low, high, colors):
        ax.errorbar(xx, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", ms=6.2,
                    color=color, ecolor=color, mec="white", mew=0.9, capsize=3.0, lw=1.8, zorder=3)
    ax.set_xticks(x, labels)
    if ylabel:
        ax.set_ylabel(r"Paired $\Delta$NRMSE (candidate - reference)")
    clean(ax)


def draw_ranks(ax, stage_b: pd.DataFrame, plotted: list[dict]) -> None:
    dvals, qvals = [1, 2, 3, 5], [1, 2, 3, 5]
    values = []
    for d in dvals:
        row = REFERENCE if d == 1 else get_row(stage_b, candidate_key(d, 1, 2.0, 16.0))
        values.append(row)
        plotted.append(plot_record("C", f"d={d}", "taxon-side rank", row))
    for q in qvals:
        row = REFERENCE if q == 1 else get_row(stage_b, candidate_key(1, q, 2.0, 16.0))
        values.append(row)
        plotted.append(plot_record("C", f"q={q}", "subject-side rank", row))
    error_points(ax, values, [str(v) for v in dvals + qvals], [BLUE] * 4 + [GREEN] * 4)
    ax.axvline(3.5, color="#8FA0B0", lw=0.8, ls="--")
    ax.text(1.5, 0.96, "taxon-side rank d", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7.6, color=BLUE, fontweight="bold")
    ax.text(5.5, 0.96, "subject-side rank q", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7.6, color=GREEN, fontweight="bold")
    ax.set_xlabel("Number of latent components")
    panel_title(ax, "C", "Latent-rank sensitivity")


def draw_temporal_effects(ax_f, ax_g, stage_b: pd.DataFrame, plotted: list[dict]) -> None:
    lambda_levels = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
    lambda_values = []
    for value in lambda_levels:
        row = REFERENCE if value == 2.0 else get_row(stage_b, candidate_key(1, 1, value, 16.0))
        lambda_values.append(row)
        plotted.append(plot_record("D", f"lambda={value:g}", "temporal penalty", row))
    error_points(ax_f, lambda_values, [f"{v:g}" for v in lambda_levels],
                 [VERMILION, VERMILION, ORANGE, GREEN, BLUE, BLUE])
    ax_f.set_xlabel(r"Temporal-penalty strength, $\lambda$")
    ax_f.set_ylabel(r"Paired $\Delta$NRMSE")
    panel_title(ax_f, "D", "Temporal regularization strength")

    sigma_levels = [2.0, 4.0, 8.0, 16.0]
    sigma_values = []
    for value in sigma_levels:
        row = REFERENCE if value == 16.0 else get_row(stage_b, candidate_key(1, 1, 2.0, value))
        sigma_values.append(row)
        plotted.append(plot_record("E", f"sigma={value:g}", "temporal bandwidth", row))
    error_points(ax_g, sigma_values, [f"{v:g}" for v in sigma_levels],
                 [VERMILION, ORANGE, GREEN, BLUE])
    ax_g.set_xlabel(r"Gaussian bandwidth, $\sigma$")
    ax_g.set_ylabel("")
    panel_title(ax_g, "E", "Temporal borrowing range")


def validate_equivalence(audit: pd.DataFrame) -> None:
    required = {
        ("no_temporal", "low_rank_only"),
        ("no_mask_awareness", "plain_2d2pca"),
    }
    observed = set()
    for row in audit[audit.metric == "nrmse"].itertuples(index=False):
        if str(row.implementation_equivalent).lower() == "true":
            observed.add((row.variant_a, row.variant_b))
    if not required.issubset(observed):
        raise ValueError("The expected Stage C implementation-equivalence audit did not pass")


def draw_ablation(ax, stage_c: pd.DataFrame, audit: pd.DataFrame, plotted: list[dict]) -> None:
    validate_equivalence(audit)
    specs = [
        ("no_mask_awareness", "No mask awareness", VERMILION),
        ("no_temporal", "No temporal penalty", ORANGE),
        ("temporal_only", "Temporal-only\ninterpolation", PURPLE),
        ("row_2dpca", "Row 2DPCA", GREEN),
        ("column_2dpca", "Column 2DPCA", GREEN),
    ]
    rows = [(get_row(stage_c, key), label, color, key) for key, label, color in specs]
    rows.sort(key=lambda item: float(item[0]["absolute_difference_mean"]), reverse=True)
    y = np.arange(len(rows))
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    for yy, (row, label, color, key) in zip(y, rows):
        mean = float(row.absolute_difference_mean)
        low = float(row.absolute_difference_ci_low)
        high = float(row.absolute_difference_ci_high)
        ax.errorbar(mean, yy, xerr=[[mean - low], [high - mean]], fmt="o", ms=7.0,
                    color=color, ecolor=color, mec="white", mew=0.9, capsize=3.0, lw=1.9, zorder=3)
        plotted.append(plot_record("F", key, label, row))
    ax.set_yticks(y, [item[1] for item in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired $\Delta$NRMSE vs full model (>0 indicates worse performance)")
    clean(ax, "x")
    panel_title(ax, "F", "Ablation of structural components")


def bootstrap_mean(values: np.ndarray, reps: int, seed: int) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, reps, 1000):
        size = min(1000, reps - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        draws.append(values[idx].mean(axis=1))
    sampled = np.concatenate(draws)
    return float(values.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)), len(values)


def row_from_values(mean: float, low: float, high: float, n: int) -> dict:
    return {"absolute_difference_mean": mean, "absolute_difference_ci_low": low,
            "absolute_difference_ci_high": high, "paired_repeat_n": n}


def draw_mechanisms(ax, stage_b: pd.DataFrame, timepoint: pd.DataFrame,
                    bootstrap_reps: int, bootstrap_seed: int, plotted: list[dict]) -> None:
    perturbations = [
        (candidate_key(1, 1, 0.0, 16.0), "λ = 0", VERMILION, "o"),
        (candidate_key(1, 1, 2.0, 2.0), "σ = 2", ORANGE, "^"),
    ]
    rows = [
        ("random|0.2", "Random: 20% entries"),
        ("random|0.5", "Random: 50% entries"),
        ("random|0.8", "Random: 80% entries"),
        ("time_point", "Time point: one visit"),
        ("time_window|0.2", "Time window: 20% visits"),
        ("time_window|0.5", "Time window: 50% visits"),
        ("time_window|0.8", "Time window: 80% visits"),
        ("subject_dropout|0.2", "Dropout: 20% subjects"),
        ("subject_dropout|0.5", "Dropout: 50% subjects"),
        ("subject_dropout|0.8", "Dropout: 80% subjects"),
    ]
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    for y, (stratum, label) in enumerate(rows):
        for index, (candidate, perturbation, color, marker) in enumerate(perturbations):
            if stratum == "time_point":
                values = timepoint.loc[timepoint.perturbation == perturbation.replace("λ", "lambda").replace("σ", "sigma"), "paired_delta_nrmse"].to_numpy(float)
                if len(values) != 90:
                    raise ValueError(f"Expected 90 pooled time-point pairs for {perturbation}; found {len(values)}")
                mean, low, high, n = bootstrap_mean(values, bootstrap_reps, bootstrap_seed + index)
                estimate = row_from_values(mean, low, high, n)
            else:
                estimate = get_row(stage_b, candidate, stratum)
            mean = float(estimate["absolute_difference_mean"])
            low = float(estimate["absolute_difference_ci_low"])
            high = float(estimate["absolute_difference_ci_high"])
            offset = -0.13 if index == 0 else 0.13
            ax.errorbar(mean, y + offset, xerr=[[mean - low], [high - mean]], fmt=marker, ms=5.8,
                        color=color, ecolor=color, mec="white", mew=0.8, capsize=2.2, lw=1.55,
                        zorder=3, label=perturbation if y == 0 else None)
            plotted.append(plot_record("G", perturbation, label, estimate))
    for divider in (2.5, 3.5, 6.5):
        ax.axhline(divider, color=GRID, lw=0.9)
    ax.set_yticks(np.arange(len(rows)), [label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"Paired $\Delta$NRMSE vs reference")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right", title="Temporal perturbation", title_fontsize=7.5)
    clean(ax, "x")
    ax.tick_params(axis="y", labelsize=7.2)
    panel_title(ax, "G", "Mechanism-stratified temporal sensitivity")


def plot_record(panel: str, item: str, group: str, row) -> dict:
    return {
        "panel": panel,
        "item": item,
        "group_or_label": group,
        "mean_paired_delta_nrmse": float(row["absolute_difference_mean"]),
        "ci_low": float(row["absolute_difference_ci_low"]),
        "ci_high": float(row["absolute_difference_ci_high"]),
        "paired_n": int(row.get("paired_repeat_n", 0)) if hasattr(row, "get") else 0,
    }


def build(data_dir: Path, bootstrap_reps: int, bootstrap_seed: int):
    stage_b = pd.read_csv(data_dir / "stage_b_paired_effects.csv")
    stage_c = pd.read_csv(data_dir / "stage_c_paired_effects.csv")
    audit = pd.read_csv(data_dir / "stage_c_equivalence_audit.csv")
    timepoint = pd.read_csv(data_dir / "stage_b_timepoint_one_visit_paired_nrmse.csv")
    plotted: list[dict] = []

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 8.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    grid = fig.add_gridspec(5, 2, height_ratios=[1.45, 1.25, 1.40, 1.55, 2.25],
                            left=0.200, right=0.985, bottom=0.050, top=0.970,
                            hspace=0.72, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])
    ax_f = fig.add_subplot(grid[3, :])
    ax_g = fig.add_subplot(grid[4, :])

    draw_sigma(ax_a)
    draw_lambda(ax_b)
    draw_ranks(ax_c, stage_b, plotted)
    draw_temporal_effects(ax_d, ax_e, stage_b, plotted)
    draw_ablation(ax_f, stage_c, audit, plotted)
    draw_mechanisms(ax_g, stage_b, timepoint, bootstrap_reps, bootstrap_seed, plotted)
    return fig, pd.DataFrame(plotted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Directory containing the delivered Figure 3 CSV files")
    parser.add_argument("--output-stem", default=None, help="Output path without extension")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026071501)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve() if args.data_dir else script_dir / "data"
    stem = Path(args.output_stem).resolve() if args.output_stem else script_dir / "Figure3_parameter_sensitivity_ablation"
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig, plotted = build(data_dir, args.bootstrap_reps, args.bootstrap_seed)
    plotted.to_csv(data_dir / "Figure3_plotted_estimates.csv", index=False)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    tiff_path = stem.with_suffix(".tiff")
    fig.savefig(pdf_path, dpi=args.dpi)
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)
    # Encoding TIFF after closing Matplotlib keeps peak memory low for the
    # 4962 x 7014 px A4 image and preserves explicit 600-dpi metadata.
    with Image.open(png_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    tifffile.imwrite(tiff_path, rgb, photometric="rgb", compression="deflate",
                     resolution=(args.dpi, args.dpi), resolutionunit="INCH")
    for suffix in (".pdf", ".png", ".tiff"):
        path = stem.with_suffix(suffix)
        print(f"{path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
