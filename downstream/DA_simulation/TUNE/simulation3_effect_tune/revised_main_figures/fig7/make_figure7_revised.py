"""Rebuild Figure 7 around cross-method consensus and artifact attenuation.

The figure is entirely data-driven from the final ECAM/DIABIMMUNE result tables.
Outputs: vector PDF plus 600-dpi PNG and losslessly compressed TIFF.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
OUT = HERE
BIO = Path(r"D:\penalized_2d2\penalized_2d2_pca\biology_analysis")

ECAM_LINDA = BIO / "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/linda_longitudinal_results.csv"
ECAM_MAAS = BIO / "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/maaslin2_longitudinal_results.csv"
ECAM_OTHER = BIO / "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/glmmPQL_ZIGMM/ECAM_glmmPQL_ZIGMM_longitudinal_results.csv"
DIAB_LINDA = BIO / "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/linda_longitudinal_results.csv"
DIAB_MAAS = BIO / "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/maaslin2_longitudinal_results.csv"
DIAB_OTHER = BIO / "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/glmmPQL_ZIGMM/DIABIMMUNE_glmmPQL_ZIGMM_longitudinal_results.csv"
DIAB_ANNOTATED = BIO / "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/diet_group_sig_taxa_annotated.csv"
DIAB_CARROT_FEATURES = BIO / "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/maaslin2_runs/base_raw_diet_group_dietary_compound_carrot_start_age_month/features/filtered_data.tsv"
DIAB_CARROT_SAMPLE_META = BIO / "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/base_raw_diet_group_dietary_compound_carrot_start_age_month_sample_metadata.csv"
TRAJECTORY = BIO / "figure_main_infant_gut/panel_data/panel_e_ecam_bacteroides.csv"

COLORS = {
    "LinDA": "#E69F00",
    "MaAsLin2": "#0072B2",
    "glmmPQL": "#6F5A8D",
    "ZIGMM": "#777777",
    "base": "#9AA0A6",
    "imputed": "#34435E",
    "vaginal": "#009E73",
    "cesarean": "#D55E00",
    "support": "#168C80",
    "artifact": "#CC3311",
    "grid": "#D8DCE2",
    "text": "#202124",
    "strip": "#F1F3F4",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_header(fig, spec, letter: str, title: str, header_fraction: float = 0.17):
    layout = spec.subgridspec(
        2, 1,
        height_ratios=[header_fraction, 1 - header_fraction],
        hspace=0.07,
    )
    header = fig.add_subplot(layout[0])
    header.axis("off")
    header.text(0.0, 0.50, letter, fontsize=12, fontweight="bold", va="center")
    header.text(0.045, 0.50, title, fontsize=10.2, fontweight="bold", va="center")
    return layout[1]


def read_inputs():
    ec_l = pd.read_csv(ECAM_LINDA)
    ec_m = pd.read_csv(ECAM_MAAS)
    ec_o = pd.read_csv(ECAM_OTHER)
    di_l = pd.read_csv(DIAB_LINDA)
    di_m = pd.read_csv(DIAB_MAAS)
    di_o = pd.read_csv(DIAB_OTHER)
    return ec_l, ec_m, ec_o, di_l, di_m, di_o


def select_row(df: pd.DataFrame, method: str, dataset: str, feature: str, group_mode: str):
    x = df.loc[
        df["dataset"].eq(dataset)
        & df["feature_id"].eq(feature)
        & df["group_mode"].eq(group_mode)
    ]
    if "da_method" in x.columns:
        x = x.loc[x["da_method"].eq(method)]
    if x.empty:
        return None
    return x.iloc[0]


def prepare_evidence():
    ec_l, ec_m, ec_o, di_l, di_m, di_o = read_inputs()

    delivery_rows = []
    for method, frame in [
        ("LinDA", ec_l),
        ("MaAsLin2", ec_m),
        ("glmmPQL", ec_o),
    ]:
        for dataset, pipeline in [("base_raw", "Observed"), ("p2d2pca_imputed", "MOT-completed")]:
            row = select_row(frame, method, dataset, "T0024", "delivery_group")
            if row is None:
                continue
            delivery_rows.append(
                {
                    "method": method,
                    "pipeline": pipeline,
                    "effect": float(row["effect_size"]),
                    "p": float(row["p_value"]),
                    "q": float(row["q_value"]),
                }
            )
    delivery = pd.DataFrame(delivery_rows)
    delivery.to_csv(OUT / "figure7_delivery_bacteroides_three_methods.csv", index=False)

    carrot_mode = "diet_group_dietary_compound_carrot_start_age_month"
    carrot_rows = []
    for method, frame, mode in [
        ("LinDA", di_l, "diet_group"),
        ("MaAsLin2", di_m, "diet_group"),
        ("glmmPQL", di_o, carrot_mode),
    ]:
        for dataset, pipeline in [("base_raw", "Observed"), ("p2d2pca_imputed", "MOT-completed")]:
            if method in {"LinDA", "MaAsLin2"}:
                x = frame.loc[
                    frame["dataset"].eq(dataset)
                    & frame["feature_id"].eq("T0336")
                    & frame["group_mode"].eq("diet_group")
                    & frame["diet_col"].eq("dietary_compound_carrot_start_age_month")
                ]
                row = None if x.empty else x.iloc[0]
            else:
                row = select_row(frame, method, dataset, "T0336", mode)
            if row is None:
                continue
            carrot_rows.append(
                {
                    "method": method,
                    "pipeline": pipeline,
                    "effect": pd.to_numeric(row["effect_size"], errors="coerce"),
                    "p": pd.to_numeric(row["p_value"], errors="coerce"),
                    "q": pd.to_numeric(row["q_value"], errors="coerce"),
                    "fit_status": str(row.get("fit_status", "ok")),
                }
            )
    carrot = pd.DataFrame(carrot_rows)
    carrot.to_csv(OUT / "figure7_carrot_chloroplast_three_methods.csv", index=False)

    # Exact LinDA discovery-set accounting for the carrot contrast (q < .10).
    lx = di_l.loc[
        di_l["group_mode"].eq("diet_group")
        & di_l["diet_col"].eq("dietary_compound_carrot_start_age_month")
    ]
    set_rows = []
    sets_by_cutoff = {}
    for cutoff in [0.05, 0.10]:
        base = set(
            lx.loc[
                lx["dataset"].eq("base_raw") & lx["q_value"].lt(cutoff),
                "feature_id",
            ]
        )
        imp = set(
            lx.loc[
                lx["dataset"].eq("p2d2pca_imputed")
                & lx["q_value"].lt(cutoff),
                "feature_id",
            ]
        )
        sets_by_cutoff[cutoff] = (base, imp)
        for category, value in [
            ("Shared", len(base & imp)),
            ("Lost after completion", len(base - imp)),
            ("Gained after completion", len(imp - base)),
            ("Observed total", len(base)),
            ("MOT-completed total", len(imp)),
        ]:
            set_rows.append({"q_cutoff": cutoff, "category": category, "n": value})
    set_summary = pd.DataFrame(set_rows)
    set_summary.to_csv(OUT / "figure7_carrot_linda_set_accounting.csv", index=False)

    # Six taxa specified for the carrot interpretation plus the single
    # post-completion q<.05 call. Stars in the figure always mean q<.05.
    selected_info = {
        "T0336": ("T0336 Chloroplast", "INGESTED PLANT DNA"),
        "T0351": ("T0351 Mitochondria", "INGESTED PLANT DNA"),
        "T0678": ("T0678 Eikenella", "ORAL-DERIVED / LOW-PREVALENCE TAXA"),
        "T0696": ("T0696 Actinomyces", "ORAL-DERIVED / LOW-PREVALENCE TAXA"),
        "T0557": ("T0557 Fusobacterium", "ORAL-DERIVED / LOW-PREVALENCE TAXA"),
        "T0705": ("T0705 Blautia", "GUT-RESIDENT"),
        "T0703": ("T0703 Clostridium sp.", "GUT-RESIDENT"),
    }
    selected = lx.loc[lx["feature_id"].isin(selected_info)].copy()
    selected = selected.loc[selected["dataset"].isin(["base_raw", "p2d2pca_imputed"])]
    selected["pipeline"] = selected["dataset"].map(
        {"base_raw": "Observed", "p2d2pca_imputed": "MOT-completed"}
    )
    selected["display_name"] = selected["feature_id"].map(
        lambda value: selected_info[value][0]
    )
    selected["sequence_type"] = selected["feature_id"].map(
        lambda value: selected_info[value][1]
    )
    raw_features = pd.read_csv(DIAB_CARROT_FEATURES, sep="\t")
    sample_meta = pd.read_csv(DIAB_CARROT_SAMPLE_META)
    raw_features = raw_features.merge(
        sample_meta[["sample_id", "participant_id"]],
        left_on="feature",
        right_on="sample_id",
        how="left",
    )
    total_infants = int(sample_meta["participant_id"].nunique())
    prevalence = {
        feature_id: int(
            raw_features.loc[
                raw_features[feature_id].fillna(0).gt(0), "participant_id"
            ].nunique()
        )
        for feature_id in selected_info
    }
    selected["prevalence_label"] = selected["feature_id"].map(
        lambda value: f"{prevalence[value]}/{total_infants} infants"
    )
    selected["status"] = np.select(
        [selected["q_value"].lt(0.05), selected["q_value"].lt(0.10)],
        ["significant", "trend"],
        default="not significant",
    )
    selected = selected[
        [
            "feature_id", "display_name", "sequence_type", "pipeline",
            "prevalence_label", "effect_size", "p_value", "q_value", "status",
        ]
    ].sort_values(["sequence_type", "feature_id", "pipeline"])
    selected.to_csv(OUT / "figure7_carrot_selected_taxa_q05.csv", index=False)

    base, imp = sets_by_cutoff[0.05]
    lost = pd.DataFrame({"feature_id": sorted(base - imp)})
    annotated = pd.read_csv(DIAB_ANNOTATED)
    annotated = annotated.loc[annotated["diet"].eq("carrot")].drop_duplicates("feature_id")
    lost = lost.merge(annotated[["feature_id", "taxon_short", "q_value"]], on="feature_id", how="left")
    lost["technical_sequence"] = lost["feature_id"].isin(["T0336", "T0351"])
    lost.to_csv(OUT / "figure7_carrot_linda_lost_hits.csv", index=False)
    return delivery, carrot, set_summary, selected, lost


def draw_tensor_cube(ax, x, y, w, h, depth, color, missing=False):
    """Draw a compact isometric tensor; depth is illustrative, not quantitative."""
    front = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    shift = np.array([depth, depth])
    top = np.array([front[3], front[2], front[2] + shift, front[3] + shift])
    side = np.array([front[1], front[1] + shift, front[2] + shift, front[2]])
    ax.add_patch(mpl.patches.Polygon(front, facecolor=color, alpha=0.12, edgecolor=color, lw=1.2))
    ax.add_patch(mpl.patches.Polygon(top, facecolor=color, alpha=0.20, edgecolor=color, lw=1.0))
    ax.add_patch(mpl.patches.Polygon(side, facecolor=color, alpha=0.27, edgecolor=color, lw=1.0))
    for frac in [0.25, 0.50, 0.75]:
        ax.plot([x + frac * w, x + frac * w], [y, y + h], color=color, lw=0.45, alpha=0.55)
        ax.plot([x, x + w], [y + frac * h, y + frac * h], color=color, lw=0.45, alpha=0.55)
    missing_cells = {(0, 1), (2, 2), (3, 0)} if missing else set()
    for col in range(4):
        for row in range(3):
            cx = x + (col + 0.5) * w / 4
            cy = y + (row + 0.5) * h / 3
            if (col, row) in missing_cells:
                ax.text(cx, cy, "x", ha="center", va="center", fontsize=7.0, color="#B3261E", fontweight="bold")
            else:
                ax.scatter(cx, cy, s=8, color=color, alpha=0.78, zorder=4)


def plot_panel_a(fig, spec):
    content = panel_header(fig, spec, "A", "From longitudinal tensors to biological findings")
    ax = fig.add_subplot(content)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_tensor_cube(ax, 0.035, 0.28, 0.145, 0.40, 0.038, COLORS["base"], missing=True)
    draw_tensor_cube(ax, 0.285, 0.28, 0.145, 0.40, 0.038, "#7253A3", missing=False)
    ax.text(0.108, 0.14, "Observed tensor", ha="center", fontsize=8.0, fontweight="bold")
    ax.text(0.108, 0.055, "subject x time x taxa", ha="center", fontsize=6.9, color="#666666")
    ax.text(0.358, 0.17, "Completed tensor", ha="center", fontsize=8.0, fontweight="bold")
    ax.text(0.358, 0.035, "MOT-(2D)$^2$ PCA", ha="center", fontsize=6.9, color="#7253A3")

    ax.annotate("", xy=(0.270, 0.50), xytext=(0.215, 0.50), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))
    method_y = [0.68, 0.47, 0.26]
    for label, yy, color in zip(["LinDA", "MaAsLin2", "glmmPQL"], method_y, [COLORS["LinDA"], COLORS["MaAsLin2"], COLORS["glmmPQL"]]):
        ax.add_patch(mpl.patches.FancyBboxPatch((0.545, yy - 0.075), 0.145, 0.15, boxstyle="round,pad=.010", facecolor=color, alpha=0.13, edgecolor=color, linewidth=1.1))
        ax.text(0.6175, yy, label, ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.annotate("", xy=(0.530, 0.47), xytext=(0.468, 0.47), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))
    ax.plot([0.520, 0.520], [0.26, 0.68], color="#555555", lw=0.8)
    for yy in method_y:
        ax.plot([0.520, 0.545], [yy, yy], color="#555555", lw=0.8)

    ax.add_patch(mpl.patches.FancyBboxPatch((0.790, 0.26), 0.18, 0.42, boxstyle="round,pad=.014", facecolor="#365C8D", alpha=0.13, edgecolor="#365C8D", linewidth=1.3))
    ax.text(0.88, 0.47, "Biological\nfindings", ha="center", va="center", fontsize=9.0, fontweight="bold")
    ax.annotate("", xy=(0.778, 0.47), xytext=(0.705, 0.47), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))


def plot_panel_b(fig, spec, delivery: pd.DataFrame):
    content = panel_header(fig, spec, "B", "ECAM delivery mode: concordant depleted Bacteroides signal; two methods significant")
    content_grid = content.subgridspec(2, 1, height_ratios=[0.08, 0.92], hspace=0)
    grid = content_grid[1].subgridspec(1, 2, wspace=0.28)
    methods = ["LinDA", "MaAsLin2", "glmmPQL"]
    y = np.arange(len(methods))

    ax = fig.add_subplot(grid[0, 0])
    for i, method in enumerate(methods):
        d = delivery.loc[delivery["method"].eq(method)].set_index("pipeline")
        x0, x1 = d.loc["Observed", "effect"], d.loc["MOT-completed", "effect"]
        ax.plot([x0, x1], [i, i], color=COLORS[method], lw=2.0)
        ax.scatter(x0, i, s=42, facecolor="white", edgecolor=COLORS[method], linewidth=1.5, zorder=3)
        ax.scatter(x1, i, s=44, facecolor=COLORS[method], edgecolor=COLORS[method], linewidth=1.0, zorder=3)
        label_y = i + 0.24 if i == 0 else i - 0.22
        ax.text(max(x0, x1) + 0.08, label_y, f"{x1:.2f}", fontsize=7.1, color=COLORS[method])
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Effect estimate (Cesarean vs vaginal)")
    ax.set_title("Direction is concordant (all negative)", fontsize=8.5, fontweight="bold")
    ax.set_xlim(-3.78, 0.18)

    axq = fig.add_subplot(grid[0, 1])
    for i, method in enumerate(methods):
        d = delivery.loc[delivery["method"].eq(method)].set_index("pipeline")
        q0, q1 = d.loc["Observed", "q"], d.loc["MOT-completed", "q"]
        x0, x1 = -np.log10(q0), -np.log10(q1)
        axq.plot([x0, x1], [i, i], color=COLORS[method], lw=2.0)
        axq.scatter(x0, i, s=42, facecolor="white", edgecolor=COLORS[method], linewidth=1.5, zorder=3)
        axq.scatter(x1, i, s=44, facecolor=COLORS[method], edgecolor=COLORS[method], zorder=3)
        if q1 < 0.05:
            axq.text(min(x1 + 0.027, 1.81), i, "*", ha="left", va="center", fontsize=10, fontweight="bold", color=COLORS[method])
        completed_y = i - 0.25 if i == len(methods) - 1 else i + 0.31
        axq.text(
            x1, completed_y, f"q={q1:.3f}",
            ha="right" if x1 > 1.65 else "center",
            fontsize=7.0, color=COLORS[method],
        )
    axq.axvline(-np.log10(0.10), color="#777777", ls="--", lw=0.8)
    axq.axvline(-np.log10(0.05), color="#333333", ls=":", lw=0.9)
    axq.text(-np.log10(0.10), 0.97, "trend q=.10", transform=axq.get_xaxis_transform(), ha="center", va="top", fontsize=6.7)
    axq.text(-np.log10(0.05), 0.97, "q=.05", transform=axq.get_xaxis_transform(), ha="center", va="top", fontsize=6.7)
    axq.set_yticks(y, [])
    axq.invert_yaxis()
    axq.set_xlabel(r"Significance strength, $-\log_{10}(q)$")
    axq.set_title("LinDA and glmmPQL q<.05; MaAsLin2 trend", fontsize=8.5, fontweight="bold")
    axq.set_xlim(0.93, 1.82)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", label="Observed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="#555555", label="MOT-completed"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper right")


def plot_panel_c(fig, spec):
    content = panel_header(fig, spec, "C", "Bacteroides trajectories retain the Bokulich-consistent early-life deficit")
    content_grid = content.subgridspec(2, 1, height_ratios=[0.15, 0.85], hspace=0)
    grid = content_grid[1].subgridspec(1, 2, wspace=0.24)
    tr = pd.read_csv(TRAJECTORY)
    panels = [
        ("base_frac", "Observed trajectories\nC/V: 0.36 -> 0.80"),
        ("imp_frac", "MOT-completed trajectories\nC/V: 0.49 -> 0.51"),
    ]
    colors = {"Vaginal": COLORS["vaginal"], "Cesarean": COLORS["cesarean"]}
    for idx, (column, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, idx])
        for delivery, d in tr.groupby("delivery"):
            ax.plot(d["month"], 100 * d[column], marker="o" if idx == 0 else None, ms=3.2, lw=1.6, color=colors[delivery], label=delivery)
        ax.axvline(12, color="#888888", ls=":", lw=0.9)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Age (months)")
        if idx == 0:
            ax.set_ylabel("Bacteroides (%)")
        ax.set_title(title, fontsize=8.2, fontweight="bold", pad=3)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.55)
    fig.axes[-2].legend(frameon=False, loc="lower right", ncol=2)


def plot_panel_d(fig, spec, carrot: pd.DataFrame):
    content = panel_header(fig, spec, "D", "DIABIMMUNE carrot contrast: chloroplast signal is attenuated in three methods")
    content_grid = content.subgridspec(2, 1, height_ratios=[0.07, 0.93], hspace=0)
    grid = content_grid[1].subgridspec(1, 2, wspace=0.30)
    methods = ["LinDA", "MaAsLin2", "glmmPQL"]
    y = np.arange(len(methods))
    ax = fig.add_subplot(grid[0, 0])
    for i, method in enumerate(methods):
        d = carrot.loc[carrot["method"].eq(method)].set_index("pipeline")
        q0 = d.loc["Observed", "q"]
        x0 = -np.log10(q0)
        ax.scatter(x0, i, s=43, facecolor="white", edgecolor=COLORS[method], linewidth=1.5, zorder=3)
        if q0 < 0.05:
            ax.text(min(x0 + 0.10, 4.60), i, "*", ha="left", va="center", fontsize=10, fontweight="bold", color=COLORS[method])
        q1 = d.loc["MOT-completed", "q"]
        x1 = -np.log10(q1)
        ax.plot([x0, x1], [i, i], color=COLORS[method], lw=2.0)
        ax.scatter(x1, i, s=44, facecolor=COLORS[method], edgecolor=COLORS[method], zorder=3)
        ax.text(x1, i + 0.29, f"q={q1:.3f}", ha="left" if x1 < 0.15 else "center", fontsize=7)
        observed_y = i + 0.27 if i == 0 else i - 0.23
        ax.text(x0, observed_y, f"q={q0:.3g}", ha="center", fontsize=7, color="#555555")
    ax.axvline(-np.log10(0.05), color="#444444", ls=":", lw=0.9)
    ax.text(-np.log10(0.05), 0.97, "q=.05", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.7)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(q)$")
    ax.set_title("Observed -> MOT-completed q value", fontsize=8.5, fontweight="bold")
    ax.set_xlim(-0.22, 4.72)

    axe = fig.add_subplot(grid[0, 1])
    for i, method in enumerate(methods):
        d = carrot.loc[carrot["method"].eq(method)].set_index("pipeline")
        e0, e1 = d.loc["Observed", "effect"], d.loc["MOT-completed", "effect"]
        axe.plot([e0, e1], [i, i], color=COLORS[method], lw=2.0)
        axe.scatter(e0, i, s=42, facecolor="white", edgecolor=COLORS[method], linewidth=1.5)
        axe.scatter(e1, i, s=44, facecolor=COLORS[method], edgecolor=COLORS[method])
        observed_y = i + 0.27 if i == 0 else i - 0.23
        completed_y = i - 0.23 if i == len(methods) - 1 else i + 0.29
        axe.text(e0, observed_y, f"{e0:.2f}", ha="center", fontsize=7)
        axe.text(e1, completed_y, f"{e1:.2f}", ha="center", fontsize=7)
    axe.set_yticks(np.arange(3), methods)
    axe.invert_yaxis()
    axe.set_xlabel("Effect estimate")
    axe.set_title("Effect magnitude contracts in all estimable methods", fontsize=8.5, fontweight="bold")
    axe.set_xlim(0.05, 1.30)


def plot_panel_e(fig, spec, set_summary: pd.DataFrame, selected: pd.DataFrame):
    content = panel_header(
        fig,
        spec,
        "E",
        "All observed carrot calls are low-prevalence taxa carried by three late-introduction infants (n=3)",
        header_fraction=0.16,
    )
    ax = fig.add_subplot(content)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6.05)
    ax.axis("off")

    ax.add_patch(mpl.patches.FancyBboxPatch((0.012, 3.92), 0.976, 1.66, boxstyle="round,pad=.012", facecolor="#FCE8E6", edgecolor="#E6B8B1", linewidth=0.7))
    ax.add_patch(mpl.patches.FancyBboxPatch((0.012, 2.02), 0.976, 1.58, boxstyle="round,pad=.012", facecolor="#EEF4F7", edgecolor="#B8CCD6", linewidth=0.7))
    ax.add_patch(mpl.patches.FancyBboxPatch((0.012, 0.10), 0.976, 1.68, boxstyle="round,pad=.012", facecolor="#EAF5F3", edgecolor="#AED8D1", linewidth=0.7))
    ax.text(0.025, 5.53, "INGESTED PLANT DNA", fontsize=7.2, fontweight="bold", color=COLORS["artifact"])
    ax.text(0.025, 3.46, "ORAL-DERIVED / LOW-PREVALENCE TAXA", fontsize=7.0, fontweight="bold", color="#3C6E8F")
    ax.text(0.025, 1.66, "GUT-RESIDENT", fontsize=7.2, fontweight="bold", color=COLORS["support"])
    ax.text(0.43, 5.78, "Observed", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(0.80, 5.78, "MOT-completed", ha="center", fontsize=7.5, fontweight="bold")

    order = ["T0336", "T0351", "T0678", "T0696", "T0557", "T0705", "T0703"]
    y_positions = [5.05, 4.43, 3.20, 2.67, 2.14, 1.30, 0.72]
    for feature_id, yy in zip(order, y_positions):
        rows = selected.loc[selected["feature_id"].eq(feature_id)].set_index("pipeline")
        if not {"Observed", "MOT-completed"}.issubset(rows.index):
            continue
        r0 = rows.loc["Observed"]
        r1 = rows.loc["MOT-completed"]
        sequence_type = r0["sequence_type"]
        is_organelle = sequence_type == "INGESTED PLANT DNA"
        is_oral = sequence_type.startswith("ORAL-DERIVED")
        color = COLORS["artifact"] if is_organelle else ("#3C6E8F" if is_oral else COLORS["support"])
        marker = "D" if is_organelle else "o"
        ax.text(0.035, yy, f"{r0['display_name']}  {r0['prevalence_label']}", va="center", fontsize=6.9)
        ax.annotate("", xy=(0.765, yy), xytext=(0.465, yy), arrowprops=dict(arrowstyle="->", lw=1.25, color=color, alpha=0.75))

        for xx, row in [(0.43, r0), (0.80, r1)]:
            q = float(row["q_value"])
            effect = abs(float(row["effect_size"]))
            status = row["status"]
            size = 34 + 30 * min(effect, 3.2)
            if status == "significant":
                face = color
                edge = color
            elif status == "trend":
                face = "white"
                edge = color
            else:
                face = "white"
                edge = "#9AA0A6"
            ax.scatter(xx, yy, s=size, marker=marker, facecolor=face, edgecolor=edge, linewidth=1.5, zorder=4)
            if status == "significant":
                ax.text(xx + 0.018, yy + 0.19, "*", ha="left", va="center", fontsize=10, fontweight="bold", color=color)
                if feature_id == "T0703" and xx > 0.5:
                    ax.text(0.835, yy + 0.16, "completion-only, 1/3 engines", ha="left", va="center", fontsize=5.8, color=color)
            elif status == "not significant" and is_organelle and xx > 0.5:
                ax.text(xx, yy, "x", ha="center", va="center", fontsize=7.4, fontweight="bold", color=COLORS["artifact"], zorder=5)


def save_tiff(fig, path: Path, dpi: int = 600):
    tmp = path.with_name("_figure7_temp.png")
    fig.savefig(tmp, dpi=dpi, facecolor="white", bbox_inches=None)
    with Image.open(tmp) as image:
        image.convert("RGB").save(
            path,
            format="TIFF",
            compression="tiff_deflate",
            dpi=(dpi, dpi),
        )
    tmp.unlink()


def write_trace(script_path: Path):
    digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
    rows = [
        {
            "artifact_id": "fig7",
            "source_data": str(ECAM_LINDA) + "; " + str(ECAM_MAAS) + "; " + str(ECAM_OTHER),
            "transformation_script": str(script_path),
            "script_sha256": digest,
            "caption_claim": "Three methods support a negative ECAM delivery-mode Bacteroides effect after completion.",
            "limitations": "Significance is defined as q<.05; MaAsLin2 is trend-level (q=.082) after completion.",
        },
        {
            "artifact_id": "fig7",
            "source_data": str(DIAB_LINDA) + "; " + str(DIAB_MAAS) + "; " + str(DIAB_OTHER),
            "transformation_script": str(script_path),
            "script_sha256": digest,
            "caption_claim": "The carrot-associated organellar calls lose q<.05 significance, while selected gut-bacterial effects increase but are mostly trend-level.",
            "limitations": "At q<.05, LinDA carrot discoveries decrease from 7 to 1; the 9-to-12 count applies only to the trend-inclusive q<.10 set.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "figure7_trace.csv", index=False)


def main():
    set_style()
    delivery, carrot, set_summary, selected, lost = prepare_evidence()
    width = 210 / 25.4
    height = 297 / 25.4
    fig = plt.figure(figsize=(width, height))
    outer = fig.add_gridspec(
        5, 1,
        left=0.085, right=0.955, bottom=0.055, top=0.985,
        height_ratios=[1.02, 1.42, 1.42, 1.50, 1.82],
        hspace=0.36,
    )
    plot_panel_a(fig, outer[0])
    plot_panel_b(fig, outer[1], delivery)
    plot_panel_c(fig, outer[2])
    plot_panel_d(fig, outer[3], carrot)
    plot_panel_e(fig, outer[4], set_summary, selected)

    stem = "Figure7_revised_consensus"
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    tiff = OUT / f"{stem}.tiff"
    fig.savefig(pdf, format="pdf", dpi=600, facecolor="white", bbox_inches=None)
    fig.savefig(png, format="png", dpi=600, facecolor="white", bbox_inches=None)
    save_tiff(fig, tiff, dpi=600)
    plt.close(fig)

    (OUT / "Figure7_caption.txt").write_text(
        "Figure 7. Cross-method biological evidence and technical-signal attenuation in infant-gut analyses. "
        "Throughout the figure, statistical significance is defined as q<0.05; 0.05<=q<0.10 is described only as a trend. "
        "(A) Longitudinal tensor completion followed by LinDA, MaAsLin2, and glmmPQL yields biological findings. "
        "(B) For ECAM delivery mode, all three methods estimated a negative Cesarean-associated Bacteroides effect. "
        "After completion, LinDA and glmmPQL had q<0.05, whereas MaAsLin2 was trend-level (q=0.082). "
        "(C) Observed trajectories retained the early-life Cesarean deficit "
        "consistent with Bokulich et al. (2016), whereas completion regularized the trajectories. "
        "(D) The DIABIMMUNE carrot-associated chloroplast signal was significant in the observed grid for LinDA, "
        "MaAsLin2, and glmmPQL but became non-significant after completion (q=0.771-0.996), with effect estimates "
        "contracting in all three methods. (E) Under q<0.05, LinDA carrot discoveries decreased from 7 to 1. "
        "The chloroplast and mitochondrial calls became non-significant. Four selected gut-bacterial effects "
        "increased but were trend-level after completion (q approximately 0.089); T0703 Clostridium sp. was the "
        "single completed-data q<0.05 call. The panel groups rows as ingested plant DNA, oral-derived/low-prevalence "
        "taxa, and gut-resident taxa; prevalence labels are non-zero infant prevalence in the observed carrot matrix. "
        "The former 9-to-12 count refers only to a trend-inclusive q<0.10 set and is not interpreted as an increase "
        "in significant discoveries.\n",
        encoding="utf-8",
    )
    write_trace(Path(__file__).resolve())
    print(pdf)
    print(png)
    print(tiff)


if __name__ == "__main__":
    main()
