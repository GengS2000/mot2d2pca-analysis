#!/usr/bin/env python3
"""Create a publication-scale synthesis figure from unified robustness outputs.

The figure intentionally separates: (i) the model's structural assumptions,
(ii) paired parameter sensitivity, and (iii) ablation evidence.  It reads only
the reproducible CSV outputs from analyze_unified_results.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd


TITLE = "Parameter sensitivity reveals the temporal scale and structural requirements of longitudinal microbiome reconstruction"
MODEL = "MOT-(2D)^2 PCA"

# Colourblind-safe core palette (Okabe-Ito inspired).
INK = "#101820"
MUTED = "#506176"
BLUE = "#006AA6"
SKY = "#42A5D5"
GREEN = "#009B72"
ORANGE = "#EE9B00"
VERMILION = "#D95F02"
PURPLE = "#6D4C9A"
PALE = "#E5EEF5"
GRID = "#DCE3E9"


def panel_label(ax, label: str, title: str = "", *,
                label_x: float = -0.055, label_y: float = 1.11,
                gap_inches: float = 0.34) -> None:
    """Place the bold panel letter and a concise title cleanly *above* the axes.

    Both sit above the plotting box (never beside the y-axis), so tick labels
    and axis titles can never collide with the panel letter.  The letter-to-title
    gap is expressed in inches so the spacing looks identical on every panel
    regardless of its width.
    """
    text = ax.text2D if hasattr(ax, "text2D") else ax.text
    text(label_x, label_y, label, transform=ax.transAxes, ha="left", va="baseline",
         fontsize=15, fontweight="bold", color=INK, clip_on=False)
    if title:
        fig_w = ax.figure.get_size_inches()[0]
        ax_w = ax.get_position().width * fig_w
        title_x = label_x + gap_inches / ax_w
        text(title_x, label_y, title, transform=ax.transAxes,
             ha="left", va="baseline", fontsize=10.5, fontweight="bold",
             color=INK, clip_on=False)


def clean(ax, grid_x: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#374151")
    ax.spines[["left", "bottom"]].set_linewidth(1.05)
    ax.tick_params(colors="#374151", labelsize=8.5, length=4, width=1)
    ax.grid(axis="x" if grid_x else "y", color=GRID, lw=.8, zorder=0)


def read_effects(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    b_path = root / "tables" / "stage_b_paired_effects.csv"
    c_path = root / "tables" / "stage_c_paired_effects.csv"
    absent = [str(p) for p in (b_path, c_path) if not p.exists()]
    if absent:
        raise FileNotFoundError("Required unified-analysis tables are missing: " + "; ".join(absent))
    return pd.read_csv(b_path), pd.read_csv(c_path)


def candidate_key(d: int, q: int, lam: float, sigma: float) -> str:
    # Task JSON preserves the grid's zero penalty as "0", not "0.0".
    lam_text = "0" if float(lam) == 0 else str(float(lam))
    return f"{d}|{q}|{lam_text}|{float(sigma)}"


def get_row(df: pd.DataFrame, key: str, stratum: str = "overall") -> pd.Series:
    row = df[(df["candidate"] == key) & (df["metric"] == "nrmse") & (df["stratum"] == stratum)]
    if len(row) != 1:
        raise ValueError(f"Expected one NRMSE row for {key!r}, {stratum!r}; found {len(row)}")
    return row.iloc[0]


def add_tensor_cube(fig, rect):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_proj_type("ortho")
    # Three visibly distinct tensor faces, arranged as a true cuboid.
    x, y, z = np.meshgrid(np.arange(5), np.arange(5), np.arange(5), indexing="ij")
    rng = np.random.default_rng(7)
    keep = rng.random((4, 4, 4)) > .32
    for i, j, k in zip(x[:-1,:-1,:-1][keep], y[:-1,:-1,:-1][keep], z[:-1,:-1,:-1][keep]):
        ax.bar3d(i+.05, j+.05, k+.05, .78, .78, .78,
                 color=BLUE if (i + j + k) % 3 else SKY, alpha=.76, shade=True,
                 edgecolor="white", linewidth=.3)
    # A few warm cells denote masked/withheld entries.
    for i, j, k in [(0, 1, 3), (2, 3, 1), (3, 0, 2), (1, 2, 0)]:
        ax.bar3d(i+.05, j+.05, k+.05, .78, .78, .78, color=ORANGE, alpha=.9,
                 edgecolor="white", linewidth=.3)
    ax.view_init(elev=23, azim=-57)
    ax.set(xlim=(0,4), ylim=(0,4), zlim=(0,4))
    ax.set_xticks([0, 3.8], ["subjects", ""])
    ax.set_yticks([0, 3.8], ["taxa", ""])
    ax.set_zticks([0, 3.8], ["visits", ""])
    ax.tick_params(labelsize=6, pad=-2)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_box_aspect((1, 1, .9))
    return ax


def draw_rank_axes(ax):
    ax.set_axis_off()
    # left: participant-side shared axes (q); right: taxon-side shared axes (d)
    x0, y0, w, h = .07, .22, .32, .48
    for t, alpha in [(0.0, .25), (.12, .45), (.24, .75)]:
        r = patches.FancyBboxPatch((x0+t, y0+t*.34), w, h, boxstyle="round,pad=.012,rounding_size=.025",
                                   fc=SKY, ec=BLUE, lw=.8, alpha=alpha)
        ax.add_patch(r)
        ax.plot([x0+t+.05, x0+t+w-.04], [y0+t*.34+h*.45]*2, color=BLUE, lw=1.8, alpha=.9)
    ax.annotate("shared participant\nloading axes (q)", (.23,.12), ha="center", va="top", fontsize=7.5, color=INK)
    ax.annotate("per-visit score\nmatrices", (.51,.47), ha="center", va="center", fontsize=7.0, color=INK,
                bbox=dict(boxstyle="round,pad=.35", fc="#F7FAFC", ec="#AAB8C5", lw=.7))
    ax.annotate("", (.42,.47), (.35,.47), arrowprops=dict(arrowstyle="->", color=INK, lw=2.1))
    ax.annotate("", (.61,.47), (.69,.47), arrowprops=dict(arrowstyle="->", color=INK, lw=2.1))
    for yy in np.linspace(.25,.68,4):
        ax.plot([.72,.96],[yy,yy], color=GREEN, lw=3.0, solid_capstyle="round")
    for xx in np.linspace(.74,.94,4):
        ax.plot([xx,xx],[.23,.70], color=GREEN, lw=3.0, solid_capstyle="round")
    ax.add_patch(patches.FancyBboxPatch((.70,.20), .29,.54, boxstyle="round,pad=.018,rounding_size=.025",
                                        fc="#E6F4EE", ec=GREEN, lw=.8, alpha=.75))
    ax.annotate("shared taxon\nloading axes (d)", (.84,.12), ha="center", va="top", fontsize=7.5, color=INK)
    ax.set(xlim=(0,1), ylim=(0,1))


def draw_sigma(ax):
    x = np.linspace(-25, 25, 500)
    sigmas = [(2, VERMILION), (4, ORANGE), (8, GREEN), (16, BLUE)]
    for sigma, color in sigmas:
        y = np.exp(-(x**2)/(2*sigma**2))
        ax.plot(x, y, color=color, lw=3.15, label=f"σ = {sigma}")
        ax.fill_between(x, 0, y, color=color, alpha=.09)
    ax.axvline(0, color="#5B6470", lw=1.15, ls="--", ymax=.70)
    ax.text(-23,.82,"local\nborrowing",fontsize=7,color=VERMILION,ha="left",va="top")
    ax.text(23,.82,"broad temporal\nborrowing",fontsize=7,color=BLUE,ha="right",va="top")
    ax.set(xlim=(-25,25), ylim=(0,1.44), xlabel="Temporal separation (input time units)", ylabel="Neighbour weight")
    ax.set_yticks([0,.2,.4,.6,.8,1.0])
    # Legend lives in the empty band above the peak (data y > 1.0), clear of the lobes.
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper center",
              bbox_to_anchor=(.5, .99), handlelength=1.6, columnspacing=1.4,
              handletextpad=.5)
    clean(ax)


def draw_lambda(ax):
    ax.set_axis_off()
    xs = np.linspace(.09,.91,5)
    # weak lambda: local states with little cross-visit coupling
    for i,x in enumerate(xs):
        yy = .72 + .08*np.sin(i*2.1)
        ax.add_patch(patches.Circle((x,yy), .055, fc="#FBE9E7", ec=VERMILION, lw=2.0))
        ax.plot([x-.025,x+.025],[yy,yy], color=VERMILION,lw=2.6)
    for i in range(4):
        ax.annotate("", (xs[i+1],.72+.08*np.sin((i+1)*2.1)), (xs[i],.72+.08*np.sin(i*2.1)),
                    arrowprops=dict(arrowstyle="->", color="#E8B0A7", lw=1.25, alpha=.8))
    ax.text(.02,.90,"weak λ", color=VERMILION, fontsize=8, fontweight="bold")
    ax.text(.50,.90,"visits fitted almost independently", color=MUTED, fontsize=7, ha="center")
    # strong lambda: temporally aligned latent community trajectory
    pathx = np.linspace(.09,.91,200); pathy = .28 + .075*np.sin(2*np.pi*pathx)
    ax.plot(pathx, pathy, color=BLUE, lw=3.6, alpha=.95)
    for x in xs:
        yy = .28 + .075*np.sin(2*np.pi*x)
        ax.add_patch(patches.Circle((x,yy), .052, fc="#E5F2FA", ec=BLUE, lw=2.0))
        ax.plot([x-.025,x+.025],[yy,yy], color=BLUE,lw=2.6)
    ax.text(.02,.47,"strong λ", color=BLUE, fontsize=8, fontweight="bold")
    ax.text(.50,.08,"persistent community structure carries information across visits", color=INK, fontsize=7.1, ha="center")
    ax.annotate("", (.95,.27), (.95,.73), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=.9))
    ax.text(.97,.50,"strength of\ntemporal alignment", rotation=90, va="center", fontsize=6.8, color=MUTED)
    ax.set(xlim=(0,1.06), ylim=(0,1))


def effect_plot(ax, values, labels, colors, title, selected_index=None):
    xs = np.arange(len(values))
    means = np.array([r["absolute_difference_mean"] for r in values])
    lows = np.array([r["absolute_difference_ci_low"] for r in values])
    highs = np.array([r["absolute_difference_ci_high"] for r in values])
    ax.axhline(0, color=INK, lw=1.35, zorder=1)
    for x, mean, lo, hi, col in zip(xs, means, lows, highs, colors):
        ax.errorbar(x, mean, yerr=[[mean-lo],[hi-mean]], fmt="o", ms=7.0,
                    mec="white", mew=1.0, color=col, ecolor=col, capsize=3.5, lw=2.05, zorder=3)
    if selected_index is not None:
        ax.scatter(selected_index, .002, marker="*", s=125, color=INK, zorder=4)
        ax.annotate("validation-selected", (selected_index,.002), xytext=(0,10), textcoords="offset points",
                    ha="center", fontsize=7.2, color=INK, fontweight="bold")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Paired Δ NRMSE\n(candidate − reference)", fontsize=7.7)
    if title:
        ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold", pad=4)
    clean(ax)


def draw_ranks(ax, b):
    dvals = [1,2,3,5]
    qvals = [1,2,3,5]
    vals = [get_row(b, candidate_key(d,1,2.0,16.0)) if d != 1 else None for d in dvals]
    vals += [get_row(b, candidate_key(1,q,2.0,16.0)) if q != 1 else None for q in qvals]
    # Reference configurations are displayed at exactly zero.
    ref = {"absolute_difference_mean":0.,"absolute_difference_ci_low":0.,"absolute_difference_ci_high":0.}
    vals = [ref if r is None else r for r in vals]
    effect_plot(ax, vals, [str(x) for x in dvals]+[str(x) for x in qvals],
                [BLUE]*4+[GREEN]*4, "", selected_index=None)
    ax.axvline(3.5, color="#9AA8B5", lw=.7, ls="--")
    ax.set_xlabel("Number of latent components (left: d; right: q)", fontsize=7.7)


def draw_forest(ax, c):
    labels = [
        ("no_mask_awareness", "No mask awareness"),
        ("plain_2d2pca", "Plain (2D)^2 PCA"),
        ("low_rank_only", "Low-rank only"),
        ("no_temporal", "No temporal penalty"),
        ("temporal_only", "Temporal-only"),
        ("row_2dpca", "Row 2DPCA"),
        ("column_2dpca", "Column 2DPCA"),
    ]
    rows = [(get_row(c, k), label, k) for k, label in labels]
    rows.sort(key=lambda x: x[0]["absolute_difference_mean"], reverse=True)
    colour = {"no_mask_awareness":VERMILION,"plain_2d2pca":VERMILION,
              "low_rank_only":ORANGE,"no_temporal":ORANGE,"temporal_only":PURPLE,
              "row_2dpca":GREEN,"column_2dpca":GREEN}
    y = np.arange(len(rows))
    ax.axvline(0, color=INK, lw=.9, zorder=1)
    for yy,(r,label,key) in zip(y,rows):
        m,lo,hi = r["absolute_difference_mean"],r["absolute_difference_ci_low"],r["absolute_difference_ci_high"]
        ax.errorbar(m,yy,xerr=[[m-lo],[hi-m]],fmt="o",ms=8.0,color=colour[key],ecolor=colour[key],
                    mec="white",mew=1.0,capsize=3.2,lw=2.2,zorder=3)
    ax.set_yticks(y,[x[1] for x in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Paired Δ NRMSE vs Full model (>0 worse)", fontsize=8)
    ax.text(.99,.03,"Points: mean paired difference; lines: bootstrap 95% CI\nNo mask awareness and plain (2D)^2 PCA: NRMSE-equivalent",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=6.7,color=MUTED,
            bbox=dict(boxstyle="round,pad=.25",fc="white",ec="#C9D4DE",lw=.5,alpha=.94))
    clean(ax,grid_x=True)


def draw_mechanisms(ax, b):
    wanted = ["random", "time_point", "time_window", "subject_dropout"]
    fractions = ["0.2", "0.5", "0.8"]
    pairs = [(candidate_key(1,1,0.0,16.0), "λ = 0", VERMILION, "o"),
             (candidate_key(1,1,2.0,2.0), "σ = 2", ORANGE, "^")]
    ylabels=[]; yy=[]; rec=[]
    for mi, mech in enumerate(wanted):
        for fi, frac in enumerate(fractions):
            stratum=f"{mech}|{frac}"
            y=mi*4+fi
            ylabels.append(f"{mech.replace('_',' ')}  {frac}")
            yy.append(y)
            for key,lab,col,mark in pairs:
                row=get_row(b,key,stratum)
                rec.append((y,row,lab,col,mark))
    ax.axvline(0,color=INK,lw=.8,zorder=1)
    for y,row,lab,col,mark in rec:
        m,lo,hi=row["absolute_difference_mean"],row["absolute_difference_ci_low"],row["absolute_difference_ci_high"]
        offset=-.14 if lab=="λ = 0" else .14
        ax.errorbar(m,y+offset,xerr=[[m-lo],[hi-m]],fmt=mark,ms=5.7,color=col,ecolor=col,
                    capsize=2.0,lw=1.55,mec="white",mew=.8,zorder=3,label=lab if y==0 else None)
    for i in range(1,len(wanted)):
        ax.axhline(i*4-.65,color=GRID,lw=.8)
    ax.set_yticks(yy,ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Δ NRMSE vs reference",fontsize=7.5)
    ax.legend(frameon=False,fontsize=7,loc="lower right",title="perturbation",title_fontsize=7)
    clean(ax,grid_x=True)
    ax.tick_params(axis="y",labelsize=6.7)


def build_figure(b: pd.DataFrame, c: pd.DataFrame):
    plt.rcParams.update({
        "font.family":"DejaVu Sans", "font.size":9, "axes.linewidth":1.05,
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.facecolor":"white",
    })
    fig=plt.figure(figsize=(17.0,10.4),facecolor="white")

    # --- Row 1: conceptual schematics (A-D), all sharing one top edge. ---------
    r1_bottom, r1_h = .705, .205
    ax_a=add_tensor_cube(fig,[.045,r1_bottom,.185,r1_h])
    panel_label(ax_a,"A","Longitudinal microbiome tensor",label_x=.02,label_y=1.10)
    ax_b=fig.add_axes([.278,r1_bottom,.175,r1_h]); draw_rank_axes(ax_b)
    panel_label(ax_b,"B","Two-sided latent representation")
    ax_c=fig.add_axes([.508,r1_bottom,.205,r1_h]); draw_sigma(ax_c)
    panel_label(ax_c,"C","Temporal borrowing reach (σ)")
    ax_d=fig.add_axes([.760,r1_bottom,.205,r1_h]); draw_lambda(ax_d)
    panel_label(ax_d,"D","Temporal alignment strength (λ)")

    # --- Row 2: empirical parameter sensitivity (E-G), aligned tops. ----------
    r2_bottom, r2_h = .400, .210
    ax_e=fig.add_axes([.045,r2_bottom,.250,r2_h]); draw_ranks(ax_e,b)
    panel_label(ax_e,"E","Latent-rank sensitivity")
    lambda_keys=[candidate_key(1,1,x,16.0) for x in [0.0,.01,.1,.5,1.0]]
    lambda_values=[get_row(b,k) for k in lambda_keys] + [{"absolute_difference_mean":0.,"absolute_difference_ci_low":0.,"absolute_difference_ci_high":0.}]
    ax_f=fig.add_axes([.360,r2_bottom,.290,r2_h])
    effect_plot(ax_f,lambda_values,["0","0.01","0.1","0.5","1","2"],[VERMILION,VERMILION,ORANGE,GREEN,BLUE,BLUE],"",selected_index=None)
    ax_f.set_xlabel("Temporal-penalty strength λ",fontsize=8.8)
    panel_label(ax_f,"F","Temporal regularization strength")
    sigma_keys=[candidate_key(1,1,2.0,x) for x in [2.0,4.0,8.0]]
    sigma_values=[get_row(b,k) for k in sigma_keys] + [{"absolute_difference_mean":0.,"absolute_difference_ci_low":0.,"absolute_difference_ci_high":0.}]
    ax_g=fig.add_axes([.715,r2_bottom,.250,r2_h])
    effect_plot(ax_g,sigma_values,["2","4","8","16"],[VERMILION,ORANGE,GREEN,BLUE],"",selected_index=None)
    ax_g.set_xlabel("Gaussian temporal bandwidth σ",fontsize=8.8)
    panel_label(ax_g,"G","Temporal borrowing reach")

    # --- Row 3: ablation forest (H) and mechanism stratification (I). ----------
    r3_bottom, r3_h = .075, .235
    ax_h=fig.add_axes([.045,r3_bottom,.470,r3_h]); draw_forest(ax_h,c)
    panel_label(ax_h,"H","Ablation of structural components")
    ax_i=fig.add_axes([.615,r3_bottom,.350,r3_h]); draw_mechanisms(ax_i,b)
    panel_label(ax_i,"I","Mechanism-stratified temporal sensitivity")
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",default="results/unified_analysis",help="Directory created by analyze_unified_results.py")
    parser.add_argument("--output-stem",default=None,help="Output path without extension; defaults to <analysis-dir>/figures/fig_parameter_sensitivity_ablation_highcontrast")
    parser.add_argument("--dpi",type=int,default=600)
    args=parser.parse_args()
    suite=Path.cwd() if (Path.cwd()/"results").exists() else Path(__file__).resolve().parents[1]
    analysis=(suite/args.analysis_dir).resolve()
    stem=Path(args.output_stem).resolve() if args.output_stem else analysis/"figures"/"fig_parameter_sensitivity_ablation_highcontrast"
    stem.parent.mkdir(parents=True,exist_ok=True)
    b,c=read_effects(analysis)
    fig=build_figure(b,c)
    fig.savefig(stem.with_suffix(".pdf"),dpi=args.dpi,bbox_inches="tight",pad_inches=.12)
    fig.savefig(stem.with_suffix(".png"),dpi=args.dpi,bbox_inches="tight",pad_inches=.12)
    try:
        fig.savefig(stem.with_suffix(".tiff"),dpi=args.dpi,bbox_inches="tight",pad_inches=.12,
                    pil_kwargs={"compression":"tiff_lzw"})
    except TypeError:  # older Pillow backends do not accept pil_kwargs
        fig.savefig(stem.with_suffix(".tiff"),dpi=args.dpi,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
    for suffix in (".pdf", ".png", ".tiff"):
        path=stem.with_suffix(suffix)
        print(f"{path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
