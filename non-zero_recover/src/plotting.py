from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"[plotting warning] matplotlib unavailable; skipping figure ({exc!r})")
        return None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metric_vs_frac(df: pd.DataFrame, outpath: str | Path, metric: str, title: str = "") -> None:
    """Line plot of a metric against masking fraction for each method."""
    if df.empty or metric not in df.columns:
        return
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    group_cols = [c for c in ["method", "frac"] if c in df.columns]
    g = df.groupby(group_cols, dropna=False)[metric].mean().reset_index()
    for method, sub in g.groupby("method"):
        sub = sub.sort_values("frac")
        ax.plot(sub["frac"], sub[metric], marker="o", label=str(method), linewidth=1.6)
    ax.set_xlabel("Masked fraction")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_detection_plot(df: pd.DataFrame, outpath: str | Path) -> None:
    """Sensitivity and precision curves by replacement mode."""
    if df.empty:
        return
    plt = _plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ax, metric in zip(axes, ["zero_detection_sensitivity", "zero_detection_precision"]):
        if metric not in df.columns:
            continue
        g = df.groupby(["replacement_mode", "frac"], dropna=False)[metric].mean().reset_index()
        for mode, sub in g.groupby("replacement_mode"):
            sub = sub.sort_values("frac")
            ax.plot(sub["frac"], sub[metric], marker="o", label=str(mode))
        ax.set_title(metric.replace("zero_detection_", ""))
        ax.set_xlabel("Masked fraction")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_ranking_heatmap(df: pd.DataFrame, outpath: str | Path, metric: str = "rmse") -> None:
    """Heatmap of method ranks by mechanism/fraction."""
    if df.empty or metric not in df.columns:
        return
    plt = _plt()
    if plt is None:
        return
    g = df.groupby(["mechanism", "frac", "method"], dropna=False)[metric].mean().reset_index()
    rows = []
    for (mech, frac), sub in g.groupby(["mechanism", "frac"]):
        ascending = metric not in {"pearson", "spearman"}
        sub = sub.sort_values(metric, ascending=ascending)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            rows.append({"condition": f"{mech}\n{frac:g}", "method": row["method"], "rank": rank})
    pivot = pd.DataFrame(rows).pivot(index="method", columns="condition", values="rank")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * pivot.shape[1]), max(4, 0.35 * pivot.shape[0])))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index.astype(str))
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns.astype(str), rotation=70, ha="right", fontsize=7)
    ax.set_title(f"Method rank by {metric}")
    fig.colorbar(im, ax=ax, label="Rank")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_boxplot(df: pd.DataFrame, outpath: str | Path, value: str, by: str = "method", title: str = "") -> None:
    """Simple grouped boxplot."""
    if df.empty or value not in df.columns or by not in df.columns:
        return
    plt = _plt()
    if plt is None:
        return
    labels, data = [], []
    for label, sub in df.groupby(by):
        vals = sub[value].dropna().to_numpy(dtype=float)
        if vals.size:
            labels.append(str(label))
            data.append(vals)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels)), 4.5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel(value)
    ax.set_title(title or value)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, outpath: str | Path, x: str, y: str, color: Optional[str] = None, title: str = "") -> None:
    """Scatter plot with optional categorical color."""
    if df.empty or x not in df.columns or y not in df.columns:
        return
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if color and color in df.columns:
        for label, sub in df.groupby(color):
            ax.scatter(sub[x], sub[y], s=12, alpha=0.65, label=str(label))
        ax.legend(fontsize=7)
    else:
        ax.scatter(df[x], df[y], s=12, alpha=0.65)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_hist_overlay(values: dict[str, Iterable[float]], outpath: str | Path, title: str = "") -> None:
    """Overlay histograms for a few one-dimensional distributions."""
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, vals in values.items():
        try:
            vals = np.asarray(vals, dtype=float).ravel()
        except (TypeError, ValueError):
            vals = np.asarray(list(vals), dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=30, histtype="step", density=True, linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("log10(profile + eps)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_ecdf_overlay(values: dict[str, Iterable[float]], outpath: str | Path, title: str = "") -> None:
    """Overlay empirical CDFs for a few one-dimensional distributions."""
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, vals in values.items():
        try:
            vals = np.asarray(vals, dtype=float).ravel()
        except (TypeError, ValueError):
            vals = np.asarray(list(vals), dtype=float).ravel()
        vals = np.sort(vals[np.isfinite(vals)])
        if vals.size:
            y = np.arange(1, vals.size + 1, dtype=float) / vals.size
            ax.step(vals, y, where="post", linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("log10(profile + eps)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_heatmap(matrix: np.ndarray, outpath: str | Path, labels: Optional[Iterable[str]] = None, title: str = "") -> None:
    """Save a correlation heatmap."""
    plt = _plt()
    if plt is None:
        return
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    if labels is not None and matrix.shape[0] <= 40:
        labels = [str(x).split("|")[-1] for x in labels]
        ax.set_xticks(np.arange(matrix.shape[0]), labels=labels, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(matrix.shape[0]), labels=labels, fontsize=5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_trajectory_plot(times, true_y, masked_y, imputed_y, masked_points, outpath: str | Path, title: str = "") -> None:
    """Plot original, masked, and imputed longitudinal trajectories."""
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, true_y, marker="o", label="original true", linewidth=1.5)
    ax.plot(times, masked_y, marker="s", label="masked", linewidth=1.2)
    ax.plot(times, imputed_y, marker="^", label="imputed", linewidth=1.2)
    masked_points = np.asarray(masked_points, dtype=bool)
    if masked_points.any():
        ax.scatter(np.asarray(times)[masked_points], np.asarray(true_y)[masked_points], s=70, facecolors="none", edgecolors="black", label="artificial mask")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("log10(profile + eps)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
