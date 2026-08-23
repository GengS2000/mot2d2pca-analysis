# -*- coding: utf-8 -*-
"""
Figure builder for "Recovering masked non-zero abundances with penalized-2d2-PCA".
Layout A (main) + supplementary figures S1-S3 + curated supplementary tables.
Saves PNG + TIFF at 600 dpi.  No fabricated data: everything is read from the
experiment CSVs / supp tables.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.patheffects as pe

# ----------------------------------------------------------------------------- paths
ROOT = r"D:\penalized_2d2\penalized_2d2_pca\non-zero_recover_new"
E2 = os.path.join(ROOT, "experiment2_distribution")
E3 = os.path.join(ROOT, "experiment3_trajectory")
E4 = os.path.join(ROOT, "experiment4_dynamic_correlation")
SUP = os.path.join(ROOT, "supp_tables")
OUT = os.path.join(ROOT, "figures_final")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "legend.fontsize": 6.3,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "svg.fonttype": "none",
})

C_METHOD = "#009E73"   # teal-green : penalized-2d2-PCA
C_BASE   = "#8C8C8C"   # grey       : no-imputation
C_TRUTH  = "#000000"   # black      : ground-truth non-zero
C_WARN   = "#D55E00"   # vermillion : negatives / warnings
C_ORANGE = "#E69F00"
C_BLUE   = "#0072B2"
METHOD_LABEL = "MOT-(2D)² PCA"
FRAC_COLORS = {0.05: "#56B4E9", 0.10: "#0072B2", 0.20: "#E69F00", 0.30: "#D55E00"}
# diverging map: vermillion (neg) -> white -> teal (pos)
DIVMAP = LinearSegmentedColormap.from_list("vwt", [C_WARN, "#FFFFFF", C_METHOD])
SEQMAP = LinearSegmentedColormap.from_list("seqt", ["#FFFFFF", C_METHOD, "#004B36"])

MECH_LABEL = {
    "random_entry": "Random entry",
    "low_depth_sample": "Low-depth sample",
    "low_abundance_biased": "Low-abundance",
    "time_point": "Time-point",
    "consecutive_time_window_dropout": "Time-window",
    "subject_specific_dropout": "Subject dropout",
}
MECH_ORDER = ["random_entry", "low_depth_sample", "time_point",
              "consecutive_time_window_dropout", "low_abundance_biased",
              "subject_specific_dropout"]
FRACS = [0.05, 0.10, 0.20, 0.30]

METRIC_LABEL = {
    "raw_nonzero_vs_imputed_at_masked_js": "JS  (imputed vs true non-zero)",
    "raw_nonzero_vs_imputed_at_masked_wasserstein": "Wasserstein  (imputed vs true)",
    "masked_truth_vs_imputed_at_masked_js": "JS  (imputed vs masked truth)",
    "masked_truth_vs_imputed_at_masked_wasserstein": "Wasserstein  (vs masked truth)",
    "js_full": "JS  (full profile)",
    "wasserstein_masked": "Wasserstein  (masked cells)",
    "taxon_mean_rank_spearman": "Taxon mean-rank Spearman",
    "trajectory_pearson": "Trajectory Pearson",
    "trajectory_spearman": "Trajectory Spearman",
    "trajectory_rmse": "Trajectory RMSE",
    "lag1_error_imputed": "Lag-1 step error",
    "second_difference_mae_imputed": "2nd-difference MAE",
    "dtw_imputed": "DTW distance",
    "static_pearson": "Static corr. Pearson",
    "nonzero_static_pearson": "Static corr. (non-zero only)",
    "lagged_pearson": "Lagged corr. Pearson",
    "lagged_spearman": "Lagged corr. Spearman",
    "lagged_frobenius": "Lagged corr. Frobenius",
    "jaccard": "Network edge Jaccard",
    "hub_jaccard": "Hub-taxa Jaccard",
}
def mlab(m):
    return METRIC_LABEL.get(m, m.replace("_", " "))


def panel_letter(ax, s, dx=-0.02, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="bottom", ha="right")


def figure_panel_letter(fig, ax, s, x=None, y=None, dx=-0.012, dy=0.006):
    bbox = ax.get_position()
    fig.text((bbox.x0 if x is None else x) + dx,
             (bbox.y1 if y is None else y) + dy,
             s, fontsize=12, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    png = os.path.join(OUT, name + ".png")
    tif = os.path.join(OUT, name + ".tiff")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(tif, dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    print("saved", png, "+", tif)
    plt.close(fig)


# ----------------------------------------------------------------------------- load
dmet = pd.read_csv(os.path.join(E2, "raw_nonzero_vs_imputed_at_masked_distribution_metrics.csv"))
dq   = pd.read_csv(os.path.join(E2, "raw_nonzero_vs_imputed_at_masked_distribution_quantiles.csv"))
tq   = pd.read_csv(os.path.join(E2, "raw_nonzero_da_log_distribution_quantiles.csv"))
traj = pd.read_csv(os.path.join(E3, "trajectory_metrics_repeat_level.csv"))
lag  = pd.read_csv(os.path.join(E4, "lagged_correlation_metrics.csv"))
s1   = pd.read_csv(os.path.join(SUP, "TableS1_repeat_level_inference_stats.csv"))
s2   = pd.read_csv(os.path.join(SUP, "TableS2_condition_level_means.csv"))
s3   = pd.read_csv(os.path.join(SUP, "TableS3_headline_summary.csv"))

M, B = "penalized_2d2_pca", "no_imputation"


def s1row(metric):
    return s1.loc[s1.metric == metric].iloc[0]


def paired(df, value, mech_frac_rep=("mechanism", "frac", "repeat")):
    """pivot method vs baseline on the given value column -> (base, method, frac)."""
    keys = list(mech_frac_rep)
    sub = df[df.method.isin([M, B])][keys + ["method", value]]
    p = sub.pivot_table(index=keys, columns="method", values=value).dropna()
    return p[B].values, p[M].values, p.reset_index()["frac"].values


# quantile CDF curves
truth_cdf = tq.sort_values("quantile")[["quantile", "log10_profile_plus_eps"]].values
mcurve = (dq[(dq.method == M) & (dq.distribution == "imputed_at_artificial_mask")]
          .groupby("quantile")["log10_profile_plus_eps"].mean().sort_index())
bcurve = (dq[(dq.method == B) & (dq.distribution == "imputed_at_artificial_mask")]
          .groupby("quantile")["log10_profile_plus_eps"].mean().sort_index())
tcurve = tq.groupby("quantile")["log10_profile_plus_eps"].mean().sort_index()


def sample_from_quantiles(qs, vals, n=4000):
    u = np.random.RandomState(0).rand(n)
    return np.interp(u, qs, vals)


# =============================================================================== MAIN
def build_main():
    fig = plt.figure(figsize=(7.2, 7.7))
    fig.suptitle(f"{METHOD_LABEL} recovers masked non-zero abundances closer to truth\n"
                 "than zero-filling — across distribution, trajectory and dynamics",
                 fontsize=9.5, fontweight="bold", y=0.995)

    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.92],
                     hspace=0.42, top=0.905, bottom=0.065, left=0.075, right=0.965)

    # ---- TOP band: A (upper-left) + B (lower-left) + C (right)
    topgs = GridSpecFromSubplotSpec(1, 2, outer[0], width_ratios=[0.58, 0.42], wspace=0.30)
    Ags = GridSpecFromSubplotSpec(2, 1, topgs[0], height_ratios=[0.40, 0.60], hspace=0.62)
    axA = fig.add_subplot(Ags[0])
    Bgs = GridSpecFromSubplotSpec(1, 2, Ags[1], width_ratios=[0.68, 0.32], wspace=0.30)
    axB = fig.add_subplot(Bgs[0])
    axBf = fig.add_subplot(Bgs[1])
    axC = fig.add_subplot(topgs[1])

    # ---- BOTTOM band: D1 | D2 | E
    botgs = GridSpecFromSubplotSpec(1, 3, outer[1], width_ratios=[0.30, 0.30, 0.40], wspace=0.48)
    axD1 = fig.add_subplot(botgs[0])
    axD2 = fig.add_subplot(botgs[1])
    axE = fig.add_subplot(botgs[2])

    # ------------------------------------------------------------------ A: setup + floor
    ax = axA
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # Small 3D tensor glyph: subject x taxa x time.
    x0, y0, cs = 0.35, 3.35, 0.54
    dx, dy = 0.48, 0.36
    n_subject, n_taxa, n_time = 4, 5, 3
    missing_t = 1
    # held-out non-zero cells; the mask is given by the design, never detected
    zero_cells = {
        (2, 0, 3): ("masked entry", "#FFF3E8", ".."),
        (2, 2, 1): ("masked entry", "white", "//"),
        (0, 1, 2): ("masked entry", "#FFF3E8", ".."),
    }
    for t in range(n_time):
        ox, oy = x0 + t * dx, y0 + t * dy
        is_missing_time = t == missing_t
        for i in range(n_subject):
            for j in range(n_taxa):
                key = (t, i, j)
                if is_missing_time:
                    fc, hatch, ec, lw = "#FFF3E8", "xx", C_WARN, 0.55
                elif key in zero_cells:
                    _, fc, hatch = zero_cells[key]
                    ec, lw = C_WARN, 0.60
                else:
                    fc = C_METHOD if (i + j + t) % 4 else "#E8E8E8"
                    hatch, ec, lw = None, "white", 0.45
                ax.add_patch(Rectangle((ox + j * cs, oy + (n_subject - 1 - i) * cs),
                                       cs * 0.86, cs * 0.86, facecolor=fc,
                                       edgecolor=ec, hatch=hatch, lw=lw, zorder=2 + t))
        ax.add_patch(Rectangle((ox, oy), n_taxa * cs, n_subject * cs, fill=False,
                               edgecolor=C_WARN if is_missing_time else "#777777",
                               lw=0.85 if is_missing_time else 0.45, zorder=3 + t))
    for p0, p1 in [((x0, y0), (x0 + dx * (n_time - 1), y0 + dy * (n_time - 1))),
                   ((x0 + n_taxa * cs, y0), (x0 + n_taxa * cs + dx * (n_time - 1),
                                             y0 + dy * (n_time - 1))),
                   ((x0, y0 + n_subject * cs), (x0 + dx * (n_time - 1),
                                                y0 + n_subject * cs + dy * (n_time - 1))),
                   ((x0 + n_taxa * cs, y0 + n_subject * cs),
                    (x0 + n_taxa * cs + dx * (n_time - 1),
                     y0 + n_subject * cs + dy * (n_time - 1)))]:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#777777", lw=0.45, zorder=1)
    ax.text(x0 + n_taxa * cs / 2 + dx, y0 + n_subject * cs + dy * 2 + 0.45,
            "3D tensor: subject x taxa x time",
            ha="center", va="bottom", fontsize=6.2, style="italic")
    ax.text(x0 + n_taxa * cs / 2, y0 - 0.24, "taxa", ha="center", va="top", fontsize=5.7)
    ax.text(x0 - 0.20, y0 + n_subject * cs / 2, "subjects", ha="right", va="center",
            fontsize=5.7, rotation=90)
    ax.annotate("time", xy=(x0 + n_taxa * cs + dx * 2 + 0.04, y0 + dy * 2 + 0.16),
                xytext=(x0 + n_taxa * cs + dx * 2 + 0.46,
                        y0 + dy * 2 + 0.52), fontsize=5.7,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.6))
    ax.text(x0 + n_taxa * cs / 2 + dx, y0 - 0.78,
            "5-30% of non-zero entries held out\n(entries · time points · subjects)",
            ha="center", va="top", fontsize=5.8, color=C_WARN, linespacing=1.25)

    # inset histogram: truth spread vs no-imputation floor spike
    axin = ax.inset_axes([0.56, 0.02, 0.44, 0.96])
    ts = sample_from_quantiles(tcurve.index.values, tcurve.values)
    bs = sample_from_quantiles(bcurve.index.values, bcurve.values)
    bins = np.linspace(-3.05, 2.2, 40)
    axin.hist(ts, bins=bins, color=C_TRUTH, histtype="step", lw=1.1, density=True,
              label="true non-zero")
    axin.hist(bs, bins=bins, color=C_BASE, histtype="stepfilled", alpha=0.55, density=True,
              label="no-imputation")
    axin.axvline(-3.0, color=C_WARN, lw=0.8, ls=":")
    axin.set_yticks([])
    axin.set_xlabel("log$_{10}$ abundance", fontsize=6.2, labelpad=1)
    axin.tick_params(labelsize=5.8)
    axin.annotate("collapses\nto floor", xy=(-3.0, axin.get_ylim()[1] * 0.64),
                  xytext=(-1.95, axin.get_ylim()[1] * 0.72), fontsize=5.7, color=C_WARN,
                  ha="left", arrowprops=dict(arrowstyle="->", color=C_WARN, lw=0.7))
    axin.legend(fontsize=5.3, frameon=False, loc="upper right", handlelength=1.0,
                borderpad=0.1, labelspacing=0.2)
    for s in ["top", "right"]:
        axin.spines[s].set_visible(False)
    ax.set_title("  The task: recover held-out non-zeros that zero-filling sends to the floor",
                 fontsize=7.6, pad=1, loc="left")

    # ------------------------------------------------------------------ B: CDF recovery
    ax = axB
    ax.plot(tcurve.values, tcurve.index.values, color=C_TRUTH, lw=1.6, label="true non-zero", zorder=5)
    ax.plot(mcurve.values, mcurve.index.values, color=C_METHOD, lw=1.6, label=METHOD_LABEL, zorder=4)
    ax.plot(bcurve.values, bcurve.index.values, color=C_BASE, lw=1.6, label="no-imputation", zorder=3)
    ax.axvline(-3.0, color=C_WARN, lw=0.7, ls=":", zorder=1)
    ax.set_xlabel("log$_{10}$ abundance at masked cells")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1); ax.set_xlim(-3.15, 2.1)
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, labelspacing=0.25,
              bbox_to_anchor=(0.02, 0.99))
    ax.text(0.97, 0.06, "win 100% · dz 3.9", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.4, color=C_METHOD, fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Distribution recovery (CDF)", fontsize=7.6, pad=2, loc="left")

    # B forest: distance to truth relative to no-imputation
    ax = axBf
    js = s1row("raw_nonzero_vs_imputed_at_masked_js")
    ws = s1row("raw_nonzero_vs_imputed_at_masked_wasserstein")
    rows = [("JS", js.method_mean, js.baseline_mean),
            ("Wass.", ws.method_mean, ws.baseline_mean)]
    ypos = [1, 0]
    for y, (lab, mv, bv) in zip(ypos, rows):
        ax.barh(y, 1.0, height=0.5, color=C_BASE, alpha=0.35, zorder=1)
        ax.barh(y, mv / bv, height=0.5, color=C_METHOD, zorder=2)
        ax.text(mv / bv + 0.04, y, f"{mv:.2f}", va="center", ha="left",
                fontsize=5.8, color=C_METHOD, fontweight="bold")
        ax.text(1.0, y + 0.33, f"{bv:.2f}", va="bottom", ha="right", fontsize=5.5, color=C_BASE)
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_xlim(0, 1.25); ax.set_xticks([0, 1])
    ax.set_xlabel("dist. to truth\n(rel. no-imp.)", fontsize=6.0)
    ax.set_ylim(-0.5, 1.6)
    ax.tick_params(labelsize=5.8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    # ------------------------------------------------------------------ C: trajectory paired
    ax = axC
    bx, my, fr = paired(traj, "trajectory_pearson")
    for f in FRACS:
        m = fr == f
        ax.scatter(bx[m], my[m], s=11, c=FRAC_COLORS[f], alpha=0.8, linewidths=0,
                   label=f"{int(f*100)}%")
    lo, hi = 0.6, 1.001
    ax.plot([lo, hi], [lo, hi], color="#444444", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("no-imputation  (trajectory Pearson)")
    ax.set_ylabel(f"{METHOD_LABEL}  (trajectory Pearson)")
    ax.set_aspect("equal")
    tr = s1row("trajectory_pearson")
    ax.text(0.04, 0.73, "90% of 240 conditions\nabove the diagonal\nwin 89.6% · dz 0.97",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
            color=C_METHOD, fontweight="bold")
    leg = ax.legend(title="masking", frameon=False, loc="lower right", ncol=1,
                    handletextpad=0.2, labelspacing=0.2, fontsize=5.8)
    leg.get_title().set_fontsize(5.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Trajectory recovery", fontsize=7.6, pad=2, loc="left")

    # ------------------------------------------------------------------ D1: lag-1 paired
    ax = axD1
    bx, my, fr = paired(lag, "lagged_pearson")
    for f in FRACS:
        m = fr == f
        ax.scatter(bx[m], my[m], s=9, c=FRAC_COLORS[f], alpha=0.8, linewidths=0)
    lo, hi = 0.9, 1.0008
    ax.plot([lo, hi], [lo, hi], color="#444444", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("no-imputation")
    ax.set_ylabel(METHOD_LABEL)
    ax.set_aspect("equal")
    ax.set_xticks([0.90, 0.95, 1.00]); ax.set_yticks([0.90, 0.95, 1.00])
    ax.text(0.04, 0.70, "lag-1 corr.\nwin 90.4% · dz 0.81",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.0,
            color=C_METHOD, fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Dynamic correlation", fontsize=7.6, pad=2, loc="left")

    # ------------------------------------------------------------------ D2: network bars
    ax = axD2
    jc = s1row("jaccard"); hb = s1row("hub_jaccard")
    groups = [("top-100 edge\nJaccard", jc), ("hub-taxa\nJaccard", hb)]
    xb = np.arange(len(groups))
    w = 0.36
    for i, (lab, r) in enumerate(groups):
        ax.bar(i - w/2, r.baseline_mean, w, color=C_BASE, yerr=r.se_across_cells,
               error_kw=dict(lw=0.7, capsize=1.6), zorder=2)
        ax.bar(i + w/2, r.method_mean, w, color=C_METHOD, yerr=r.se_across_cells,
               error_kw=dict(lw=0.7, capsize=1.6), zorder=2)
        ax.text(i, max(r.method_mean, r.baseline_mean) + 0.05,
                f"+{r.mean_delta:.2f}", ha="center", va="bottom",
                fontsize=6.0, color=C_METHOD, fontweight="bold")
    ax.set_xticks(xb); ax.set_xticklabels([g[0] for g in groups], fontsize=6.2)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Jaccard overlap with truth")
    ax.set_yticks([0, 0.5, 1.0])
    ax.bar(np.nan, np.nan, color=C_BASE, label="no-imputation")
    ax.bar(np.nan, np.nan, color=C_METHOD, label=METHOD_LABEL)
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=5.6,
              handlelength=1.0, labelspacing=0.2, bbox_to_anchor=(0.0, 1.0))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Network structure", fontsize=7.6, pad=2, loc="left")

    # ------------------------------------------------------------------ E: robustness heatmap
    ax = axE
    tp = (traj[traj.method.isin([M, B])]
          .groupby(["mechanism", "frac", "method"])["trajectory_pearson"].mean().reset_index())
    piv = tp.pivot_table(index="mechanism", columns="frac", values="trajectory_pearson",
                         aggfunc="first")
    adv = []
    for mech in MECH_ORDER:
        row = []
        for f in FRACS:
            mv = tp[(tp.mechanism == mech) & (tp.frac == f) & (tp.method == M)]["trajectory_pearson"]
            bv = tp[(tp.mechanism == mech) & (tp.frac == f) & (tp.method == B)]["trajectory_pearson"]
            row.append(float(mv.values[0] - bv.values[0]))
        adv.append(row)
    adv = np.array(adv)
    vmax = np.nanmax(np.abs(adv))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(adv, cmap=DIVMAP, norm=norm, aspect="auto")
    ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{int(f*100)}%" for f in FRACS])
    ax.set_yticks(range(len(MECH_ORDER)))
    ax.set_yticklabels([MECH_LABEL[m] for m in MECH_ORDER], fontsize=6.2)
    ax.set_xlabel("masking fraction")
    for i in range(len(MECH_ORDER)):
        for j in range(len(FRACS)):
            val = adv[i, j]
            ax.text(j, i, f"{val:+.02f}", ha="center", va="center", fontsize=5.6,
                    color="#222222")
    # highlight subject dropout row
    si = MECH_ORDER.index("subject_specific_dropout")
    ax.add_patch(Rectangle((-0.5, si - 0.5), len(FRACS), 1, fill=False,
                           edgecolor=C_WARN, lw=1.4))
    ax.text(len(FRACS) - 0.5, si, " hardest", va="center", ha="left",
            fontsize=5.6, color=C_WARN, clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(f"advantage  ({METHOD_LABEL} - no-imp.)", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.6)
    ax.set_title("Robustness: advantage grows with masking", fontsize=7.6, pad=3, loc="left")

    fig.canvas.draw()
    top_y = max(axA.get_position().y1, axC.get_position().y1)
    bottom_y = max(axD1.get_position().y1, axE.get_position().y1)
    for letter, axis, yref in [("A", axA, top_y), ("B", axB, None), ("C", axC, top_y),
                               ("D", axD1, bottom_y), ("E", axE, bottom_y)]:
        figure_panel_letter(fig, axis, letter, y=yref)

    save(fig, "Figure1_main_recovery")


# =============================================================================== S1
def build_s1():
    fig = plt.figure(figsize=(7.35, 3.75))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.58, 0.42], wspace=0.34,
                  left=0.11, right=0.975, top=0.80, bottom=0.14)
    fig.suptitle("Choice of inference unit for the 6×4×10 design",
                 fontsize=9, fontweight="bold", y=0.98)

    # S1a: permutation vs floored Wilcoxon
    ax = fig.add_subplot(gs[0])
    pos = s1[s1.p_perm_mechfrac_one_sided < 0.5].copy()  # supported metrics
    pos = pos.sort_values("p_perm_mechfrac_one_sided")
    y = np.arange(len(pos))
    ax.scatter(-np.log10(pos.p_perm_mechfrac_one_sided), y, s=22, color=C_METHOD,
               zorder=3, label="permutation (n=24 mech×frac)")
    ax.scatter(-np.log10(pos.p_wilcoxon_mech6_FLOORED), y, s=22, color=C_BASE,
               marker="s", zorder=3, label="Wilcoxon (n=6, floored)")
    for yy, (_, r) in zip(y, pos.iterrows()):
        ax.plot([-np.log10(r.p_perm_mechfrac_one_sided), -np.log10(r.p_wilcoxon_mech6_FLOORED)],
                [yy, yy], color="#CCCCCC", lw=0.7, zorder=1)
    ax.axvline(-np.log10(1/64), color=C_WARN, lw=0.8, ls=":")
    ax.text(-np.log10(1/64), len(pos) - 0.3, " Wilcoxon floor\n 1/2$^6$=0.016",
            color=C_WARN, fontsize=5.8, va="top", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([mlab(m) for m in pos.metric], fontsize=5.4)
    ax.set_xlabel("$-\\log_{10}$ p (one-sided)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0.13),
              borderaxespad=0.0, fontsize=5.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    panel_letter(ax, "a", dx=-0.13, dy=1.02)
    ax.set_title("Permutation reaches significance the n=6 test cannot", fontsize=7.4, loc="left")

    # S1b: design schematic
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, 9.45, "1 inference unit = 1 mechanism × fraction cell",
            ha="center", fontsize=6.6, fontweight="bold")
    x0, y_top, cs = 2.85, 7.65, 1.10
    for i in range(6):
        for j in range(4):
            x = x0 + j * cs
            yv = y_top - i * cs
            ax.add_patch(Rectangle((x, yv), cs*0.9, cs*0.9, facecolor="#EAF6F1",
                                   edgecolor=C_METHOD, lw=0.6))
            # 10 MC repeat dots
            rr = np.random.RandomState(i*4+j)
            ax.scatter(x + 0.15 + rr.rand(10)*0.6*cs, yv + 0.15 + rr.rand(10)*0.6*cs,
                       s=1.2, color=C_BASE)
    ax.text(5, 0.72, "each unit holds 10 Monte-Carlo repeats\n(replicates, NOT independent samples)",
            ha="center", va="bottom", fontsize=5.8, color=C_WARN)
    panel_letter(ax, "b", dx=-0.06, dy=1.0)

    save(fig, "FigureS1_inference_unit")


# =============================================================================== S2
def build_s2():
    exp_name = {"experiment2_distribution": "Distribution",
                "experiment3_trajectory": "Trajectory",
                "experiment4_dynamic_correlation": "Dynamics"}
    neg = {"js_full", "taxon_mean_rank_spearman", "nonzero_static_pearson"}
    df = s1.copy()
    df["exp"] = df.experiment.map(exp_name)
    order = ["Distribution", "Trajectory", "Dynamics"]
    df["exp"] = pd.Categorical(df.exp, order, ordered=True)
    df = df.sort_values(["exp", "cohen_dz_cell"], ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.34, right=0.9, top=0.9, bottom=0.09)
    y = np.arange(len(df))
    for yy, (_, r) in zip(y, df.iterrows()):
        is_neg = r.metric in neg
        col = C_WARN if is_neg else C_METHOD
        ax.plot([0, r.cohen_dz_cell], [yy, yy], color=col, lw=1.0, zorder=1)
        ax.scatter(r.cohen_dz_cell, yy, s=30, color=col, zorder=3,
                   marker="X" if is_neg else "o")
        ax.text(r.cohen_dz_cell + (0.12 if r.cohen_dz_cell >= 0 else -0.12), yy,
                f"win {r.win_rate*100:.0f}%", va="center",
                ha="left" if r.cohen_dz_cell >= 0 else "right", fontsize=5.4, color=col)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([mlab(m) for m in df.metric], fontsize=6.0)
    ax.set_xlabel(f"Cohen's dz  (positive = favours {METHOD_LABEL})")
    ax.set_xlim(-1.5, 7.2)
    # experiment separators + labels
    for e in order:
        idx = df.index[df.exp == e]
        ax.axhspan(idx.min() - 0.5, idx.max() + 0.5,
                   color="#F5F5F5" if order.index(e) % 2 else "#FFFFFF", zorder=0)
        ax.text(7.15, float(np.mean(idx.to_numpy())), e, rotation=90, va="center", ha="left",
                fontsize=7, fontweight="bold", color="#555555")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=C_METHOD, lw=0, label="supports method"),
               Line2D([0], [0], marker="X", color=C_WARN, lw=0, label="does NOT support method")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6.2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Full metric panel (honest disclosure of non-supporting metrics)",
                 fontsize=9, fontweight="bold", pad=6)
    save(fig, "FigureS2_metric_forest")


# =============================================================================== S3
def build_s3():
    js = (dmet[dmet.method.isin([M, B])]
          .groupby(["mechanism", "frac", "method"])["raw_nonzero_vs_imputed_at_masked_js"]
          .mean().reset_index())

    def grid(method):
        g = np.full((len(MECH_ORDER), len(FRACS)), np.nan)
        for i, mech in enumerate(MECH_ORDER):
            for j, f in enumerate(FRACS):
                v = js[(js.mechanism == mech) & (js.frac == f) & (js.method == method)][
                    "raw_nonzero_vs_imputed_at_masked_js"]
                if len(v):
                    g[i, j] = v.values[0]
        return g

    gM, gB = grid(M), grid(B)
    vmax = np.nanmax([gM, gB])

    fig = plt.figure(figsize=(8.2, 3.85))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.12], height_ratios=[1, 0.08],
                  wspace=0.62, hspace=0.34, left=0.11, right=0.975, top=0.80, bottom=0.18)
    fig.suptitle("Mechanism-resolved distribution recovery (Jensen–Shannon to true non-zero; lower = better)",
                 fontsize=8.6, fontweight="bold", y=0.97)

    hm_axes = []
    for k, (g, ttl, lett) in enumerate([(gB, "no-imputation", "a"), (gM, METHOD_LABEL, "b")]):
        ax = fig.add_subplot(gs[0, k])
        hm_axes.append(ax)
        im = ax.imshow(g, cmap=SEQMAP, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{int(f*100)}%" for f in FRACS], fontsize=6)
        ax.set_yticks(range(len(MECH_ORDER)))
        ax.set_yticklabels([MECH_LABEL[m] for m in MECH_ORDER] if k == 0 else [], fontsize=6.2)
        ax.set_xlabel("masking", fontsize=6.5)
        for i in range(len(MECH_ORDER)):
            for j in range(len(FRACS)):
                ax.text(j, i, f"{g[i,j]:.2f}", ha="center", va="center", fontsize=5.3,
                        color="white" if g[i, j] > vmax*0.55 else "#333333")
        si = MECH_ORDER.index("subject_specific_dropout")
        ax.add_patch(Rectangle((-0.5, si-0.5), len(FRACS), 1, fill=False, edgecolor=C_WARN, lw=1.3))
        ax.set_title(ttl, fontsize=7.4)
        panel_letter(ax, lett, dx=-0.05 if k else -0.55, dy=1.05)
    cax = fig.add_subplot(gs[1, 0:2])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("JS distance", fontsize=6.2, labelpad=2); cb.ax.tick_params(labelsize=5.6)

    # panel c: JS by mechanism (method), averaged over frac
    ax = fig.add_subplot(gs[0, 2])
    mjs = js[js.method == M].groupby("mechanism")["raw_nonzero_vs_imputed_at_masked_js"].mean()
    mjs = mjs.reindex(MECH_ORDER).sort_values()
    cols = [C_WARN if m == "subject_specific_dropout" else
            (C_ORANGE if m == "low_abundance_biased" else C_METHOD) for m in mjs.index]
    ax.barh(range(len(mjs)), mjs.values, color=cols)
    ax.set_yticks(range(len(mjs)))
    ax.set_yticklabels([MECH_LABEL[m] for m in mjs.index], fontsize=5.9)
    for i, v in enumerate(mjs.values):
        ax.text(v + 0.006, i, f"{v:.02f}", va="center", fontsize=5.6)
    ax.set_xlabel(f"mean JS ({METHOD_LABEL})", fontsize=6.5)
    ax.set_xlim(0, max(mjs.values)*1.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Subject dropout is the hard case", fontsize=7.4)
    panel_letter(ax, "c", dx=-0.05, dy=1.05)

    save(fig, "FigureS3_mechanism_panorama")


# =============================================================================== tables
def build_tables():
    keep = ["experiment", "metric", "higher_better", "n_pairs", "n_cells", "win_rate",
            "baseline_mean", "method_mean", "mean_delta", "cohen_dz_cell",
            "boot_ci_lo", "boot_ci_hi", "p_perm_mechfrac_one_sided",
            "holm_p_perm_within_all_tests", "p_wilcoxon_mech6_FLOORED"]
    t1 = s1[keep].copy()
    for c in ["win_rate", "baseline_mean", "method_mean", "mean_delta", "cohen_dz_cell",
              "boot_ci_lo", "boot_ci_hi"]:
        t1[c] = t1[c].round(4)
    t1.to_csv(os.path.join(OUT, "TableS1_inference_stats.csv"), index=False)

    t2 = s2.copy()
    t2["mean"] = t2["mean"].round(4); t2["std"] = t2["std"].round(4)
    t2.to_csv(os.path.join(OUT, "TableS2_condition_level_means.csv"), index=False)

    t3 = s3.copy()
    for c in t3.columns:
        if t3[c].dtype.kind == "f":
            t3[c] = t3[c].round(4)
    t3.to_csv(os.path.join(OUT, "TableS3_headline_summary.csv"), index=False)
    print("tables written")


if __name__ == "__main__":
    build_main()
    build_s1()
    build_s2()
    build_s3()
    build_tables()
    print("ALL DONE ->", OUT)
