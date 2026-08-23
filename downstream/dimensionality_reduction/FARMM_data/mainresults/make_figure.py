# -*- coding: utf-8 -*-
"""
SCI-publication composite figure (A-I) for the FARMM downstream DR case study.

Claim: recovering structural zeros with MOT-(2D)^2 PCA improves downstream
       TEMPTED dimensionality reduction, and yields a temporally stable,
       biologically interpretable leading component.

Layout: 3x3 (square), 600 dpi, vector (PDF/SVG) + PNG + TIFF(LZW).
Rows:  1) setup / visual        A sparsity     B embed(direct)  C embed(MOT)
       2) quantitative downstream D classify    E cluster        F PC1 subject-loading
       3) component interpretation G feature-b  H temporal xi(t) I microbial balance
"""
import os, csv, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from scipy.stats import wilcoxon, f_oneway

BASE = r"D:\penalized_2d2\penalized_2d2_pca\downstream\dimensionality_reduction\FARMM_data"
ART  = os.path.join(BASE, "artifacts", "FARMM_full")
LD   = os.path.join(BASE, "mainresults", "tempted_loadings")
OUT  = os.path.join(BASE, "mainresults", "figure_main")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

METHOD = "MOT-(2D)² PCA"          # unified paper name for the imputation method

# ----------------------------------------------------------------------------- style
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "legend.frameon": False,
    "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

GROUP_COLORS = {"EEN": "#4C72B0", "Omnivore": "#DD8452", "Vegan": "#55A868"}
GROUP_ORDER  = ["EEN", "Omnivore", "Vegan"]
PIPE_COLORS  = {"direct": "#B7B7B7", "p2d2": "#C44E52"}
PIPE_DIRECT  = "direct_tempted"
PIPE_P2D2    = "p2d2pca_imputed_then_tempted"
PIPE_LABEL   = {"direct": "Direct TEMPTED", "p2d2": f"{METHOD} → TEMPTED"}

def star(p):
    return "n.s." if p >= .05 else ("*" if p >= .01 else ("**" if p >= .001 else "***"))

def strip_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def conf_ellipse(x, y, ax, n_std=1.96, **kw):
    if len(x) < 3: return
    cov = np.cov(x, y); mx, my = np.mean(x), np.mean(y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((mx, my), w, h, angle=theta, **kw))

def sp_name(t):
    """Short, readable species label from a full lineage string."""
    leaf = t.split("|")[-1].replace("s__", "").replace("_", " ")
    return leaf.replace("Lachnospiraceae bacterium", "Lachnospiraceae bact.")

# ----------------------------------------------------------------------------- data
subj = pd.read_csv(os.path.join(ART, "subject_metadata.csv"))
g_map = dict(zip(subj["participant"], subj["group_label"]))
grp_of = subj.set_index("participant")["group_label"]

def load_coords(fn):
    df = pd.read_csv(os.path.join(ART, fn))
    df["group"] = df["participant"].map(g_map)
    return df.dropna(subset=["group"])

co_d = load_coords("tempted_coordinates_direct.csv")
co_p = load_coords("tempted_coordinates_p2d2pca_imputed.csv")

metrics = pd.read_csv(os.path.join(ART, "metrics_all.csv"))
clust   = pd.read_csv(os.path.join(ART, "cluster_evaluation.csv"))
perm    = pd.read_csv(os.path.join(ART, "permanova_subject_level_matched.csv"))
load_t  = pd.read_csv(os.path.join(ART, "subject_loading_tests_group_label.csv"))

# proportion of variation explained (for embedding axis labels)
def prop_expl(pipe):
    p = pd.read_csv(os.path.join(LD, f"{pipe}__individual_biplot__prop_explained.csv"),
                    index_col=0)
    return p.iloc[:, 0].values
pe_d, pe_p = prop_expl(PIPE_DIRECT), prop_expl(PIPE_P2D2)

# --- mask report: per-taxon replacement fraction + overall summary
mask_rows = list(csv.reader(open(os.path.join(ART, "logs", "imputation_mask_report.csv"))))
taxon_fracs = []; section = None; eps = tot = None
for r in mask_rows:
    if not r or all(c == "" for c in r):
        section = None; continue
    if len(r) >= 3 and r[0] == "summary" and r[1] == "eps_or_zero_entries": eps = float(r[2])
    if len(r) >= 3 and r[0] == "summary" and r[1] == "total_entries":     tot = float(r[2])
    if r[0] in ("taxon_rates", "participant_rates", "bin_rates"):
        section = r[0]; continue
    if section == "taxon_rates" and r[0] != "taxon":
        try: taxon_fracs.append(float(r[1]))
        except (ValueError, IndexError): pass
overall_frac = eps / tot
taxon_fracs = np.array(taxon_fracs)

perm_F, perm_p = {}, {}
for _, row in perm.iterrows():
    perm_F[row["pipeline"]] = row["pseudo_F"]; perm_p[row["pipeline"]] = row["p_value"]

# --- component interpretation inputs (feature loadings b, temporal loading xi, balance)
import xarray as xr
BALANCE_K, SHOW_K = 15, 10

def feat_loadings(pipe):
    fb = pd.read_csv(os.path.join(LD, f"{pipe}__individual_biplot__features.csv"), index_col=0)
    fb.columns = ["PC1", "PC2", "PC3"]
    return fb

fb_p = feat_loadings(PIPE_P2D2)

def temporal(pipe):
    st = pd.read_csv(os.path.join(LD, f"{pipe}__state_loadings__table.csv"))
    return st

xi_d, xi_p = temporal(PIPE_DIRECT), temporal(PIPE_P2D2)

def balance_over_time(pipe, ncfile, K=BALANCE_K):
    """PC1 microbial balance = mean(neg-loading taxa) - mean(pos-loading taxa),
    oriented so Vegan is high (matches the subject-loading PC1 sign)."""
    fb = feat_loadings(pipe)
    ds = xr.open_dataset(os.path.join(ART, ncfile)); ds = ds[list(ds.data_vars)[0]]
    tax = list(ds.coords["taxon"].values)
    s = fb["PC1"].reindex(tax).sort_values()
    neg, pos = s.head(K).index.tolist(), s.tail(K).index.tolist()
    Y = ds.transpose("participant", "bin", "taxon")
    bal = (Y.sel(taxon=neg).mean("taxon") - Y.sel(taxon=pos).mean("taxon")).values  # (subj, time)
    parts = [int(p) for p in ds.coords["participant"].values]
    grp = np.array([grp_of.loc[p] for p in parts])
    bins = [str(b) for b in ds.coords["bin"].values]
    return bal, grp, bins

bal_p, grp_p, bins_p = balance_over_time(PIPE_P2D2, "tensor_p2d2pca_imputed.nc")
bal_d, grp_d, _      = balance_over_time(PIPE_DIRECT, "tensor_base_raw.nc")

def anova_eta(bal, grp):
    subj = bal.mean(1)
    gs = [subj[grp == g] for g in GROUP_ORDER]
    F, p = f_oneway(*gs)
    eta2 = sum(len(g) * (g.mean() - subj.mean())**2 for g in gs) / ((subj - subj.mean())**2).sum()
    return F, p, eta2
F_bal_p, p_bal_p, eta_bal_p = anova_eta(bal_p, grp_p)
F_bal_d, p_bal_d, eta_bal_d = anova_eta(bal_d, grp_d)

# ----------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(13.6, 13.6))
gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.30,
                      left=0.065, right=0.985, top=0.965, bottom=0.055)

def panel_letter(ax, letter, dx=-0.15, dy=1.08):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left")

# ===== A : structural sparsity ==============================================
axA = fig.add_subplot(gs[0, 0]); strip_ax(axA)
_, _, patches = axA.hist(taxon_fracs * 100, bins=18, edgecolor="white", linewidth=0.6)
for i, p in enumerate(patches):
    p.set_facecolor(plt.cm.viridis(0.15 + 0.7 * i / len(patches)))
axA.axvline(overall_frac * 100, color="#C44E52", lw=1.8, ls="--")
axA.set_xlabel("Zero / eps entries per taxon (%)")
axA.set_ylabel("Number of taxa")
axA.set_title("Structural sparsity of the tensor")
axA.text(0.04, 0.96, f"{overall_frac*100:.1f}%\nzeros overall",
         transform=axA.transAxes, va="top", ha="left", fontsize=11,
         fontweight="bold", color="#C44E52",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C44E52", lw=1.1))
axA.text(0.035, 0.58, f"{len(taxon_fracs)} taxa × 30 subjects × time,\nrecovered by {METHOD}",
         transform=axA.transAxes, va="top", ha="left", fontsize=7.2,
         color="#555555", style="italic")
panel_letter(axA, "A")

# ===== B/C : embeddings =====================================================
def embed_panel(ax, df, title, pipe_key, pe):
    strip_ax(ax)
    for grp in GROUP_ORDER:
        sub = df[df["group"] == grp]
        ax.scatter(sub["PC1"], sub["PC2"], s=42, c=GROUP_COLORS[grp],
                   edgecolors="white", linewidths=0.6, label=grp, zorder=3)
        conf_ellipse(sub["PC1"].values, sub["PC2"].values, ax,
                     facecolor=GROUP_COLORS[grp], alpha=0.13,
                     edgecolor=GROUP_COLORS[grp], lw=1.1, zorder=1)
    ax.set_xlabel(f"TEMPTED PC1 ({pe[0]*100:.0f}%)")
    ax.set_ylabel(f"TEMPTED PC2 ({pe[1]*100:.0f}%)")
    ax.set_title(title)
    ax.text(0.03, 0.97, f"PERMANOVA\npseudo-F = {perm_F[pipe_key]:.1f}\np = {perm_p[pipe_key]:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8))
    ax.axhline(0, color="#dddddd", lw=0.7, zorder=0); ax.axvline(0, color="#dddddd", lw=0.7, zorder=0)

axB = fig.add_subplot(gs[0, 1]); embed_panel(axB, co_d, "Direct TEMPTED", PIPE_DIRECT, pe_d)
axB.legend(loc="center left", fontsize=7.5, title="Diet group", title_fontsize=7.8,
           handletextpad=0.3, borderpad=0.4)
panel_letter(axB, "B", dx=-0.17)
axC = fig.add_subplot(gs[0, 2]); embed_panel(axC, co_p, f"{METHOD} → TEMPTED", PIPE_P2D2, pe_p)
panel_letter(axC, "C", dx=-0.17)

# ===== D : supervised classification ========================================
axD = fig.add_subplot(gs[1, 0]); strip_ax(axD)
metric_keys = ["AUROC", "AUPRC", "balanced_accuracy"]
metric_lab  = ["AUROC", "AUPRC", "Balanced\naccuracy"]
clf_keys = ["logistic_regression", "random_forest"]; clf_short = {"logistic_regression": "LR", "random_forest": "RF"}
positions, box_data, box_colors = [], [], []
pos = 0; xticks, xtlabs, block_centers, star_annot = [], [], [], []
gap_clf, gap_block = 0.9, 1.6
for mkey in metric_keys:
    block_start = pos
    for ck in clf_keys:
        vals = {}
        for pk, pipe in [("direct", PIPE_DIRECT), ("p2d2", PIPE_P2D2)]:
            s = metrics[(metrics.pipeline == pipe) & (metrics.classifier == ck)].sort_values("split_id")
            vals[pk] = s[mkey].values
            positions.append(pos); box_data.append(s[mkey].values); box_colors.append(PIPE_COLORS[pk]); pos += 1
        try: _, pv = wilcoxon(vals["direct"], vals["p2d2"])
        except ValueError: pv = 1.0
        star_annot.append((positions[-2], positions[-1], max(np.max(vals["direct"]), np.max(vals["p2d2"])), pv))
        xticks.append((positions[-2] + positions[-1]) / 2); xtlabs.append(clf_short[ck]); pos += gap_clf
    block_centers.append((block_start + pos - gap_clf - 1) / 2); pos += gap_block
bp = axD.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True, showfliers=False,
                 medianprops=dict(color="#222222", lw=1.2), whiskerprops=dict(color="#666666", lw=0.9),
                 capprops=dict(color="#666666", lw=0.9), boxprops=dict(lw=0.8, edgecolor="#555555"))
for patch, c in zip(bp["boxes"], box_colors): patch.set_facecolor(c); patch.set_alpha(0.9)
for xp, d in zip(positions, box_data):
    axD.scatter(np.full_like(d, xp) + np.random.uniform(-0.14, 0.14, len(d)), d,
                s=6, c="#333333", alpha=0.35, zorder=3, linewidths=0)
for x1, x2, top, pv in star_annot:
    y = top + 0.03
    axD.plot([x1, x1, x2, x2], [y, y+0.012, y+0.012, y], lw=0.9, c="#333333")
    axD.text((x1+x2)/2, y+0.012, star(pv), ha="center", va="bottom", fontsize=8.5)
axD.set_xticks(xticks); axD.set_xticklabels(xtlabs, fontsize=7.5)
for cxb, lab in zip(block_centers, metric_lab):
    axD.text(cxb, -0.085, lab, transform=axD.get_xaxis_transform(), ha="center", va="top",
             fontsize=9, fontweight="bold")
axD.set_ylabel("Score (20 repeats)"); axD.set_ylim(0.2, 1.05); axD.set_title("Supervised classification")
axD.legend(handles=[Patch(fc=PIPE_COLORS["direct"], ec="#555555", label=PIPE_LABEL["direct"]),
                    Patch(fc=PIPE_COLORS["p2d2"], ec="#555555", label=PIPE_LABEL["p2d2"])],
           loc="lower center", fontsize=7.2)
panel_letter(axD, "D")

# ===== E : unsupervised clustering ==========================================
axE = fig.add_subplot(gs[1, 1]); strip_ax(axE)
cl_metrics = ["ARI", "NMI", "silhouette", "v_measure"]; cl_lab = ["ARI", "NMI", "Silhouette", "V-measure"]
agg = clust.groupby("pipeline")[cl_metrics].agg(["mean", "min", "max"])
xb = np.arange(len(cl_metrics)); w = 0.36
for i, (pk, pipe) in enumerate([("direct", PIPE_DIRECT), ("p2d2", PIPE_P2D2)]):
    means = [agg.loc[pipe, (m, "mean")] for m in cl_metrics]
    lo = [max(0.0, means[j] - agg.loc[pipe, (cl_metrics[j], "min")]) for j in range(len(cl_metrics))]
    hi = [max(0.0, agg.loc[pipe, (cl_metrics[j], "max")] - means[j]) for j in range(len(cl_metrics))]
    axE.bar(xb + (i-0.5)*w, means, w, yerr=[lo, hi], capsize=2.5, color=PIPE_COLORS[pk],
            edgecolor="#555555", lw=0.7, error_kw=dict(lw=0.8, ecolor="#555555"),
            label=PIPE_LABEL[pk], alpha=0.92)
    for j, mv in enumerate(means):
        axE.text(xb[j] + (i-0.5)*w, mv + 0.015, f"{mv:.2f}", ha="center", va="bottom", fontsize=6.8, color="#333333")
axE.set_xticks(xb); axE.set_xticklabels(cl_lab); axE.set_ylabel("Cluster agreement / quality")
axE.set_ylim(0, 0.72); axE.set_title("Unsupervised clustering (3 algorithms)")
axE.legend(loc="upper right", fontsize=7.2)
panel_letter(axE, "E")

# ===== F : PC1 subject-loading gradient =====================================
axF = fig.add_subplot(gs[1, 2]); strip_ax(axF)
data_by_grp = [co_p[co_p["group"] == g]["PC1"].values for g in GROUP_ORDER]
bpF = axF.boxplot(data_by_grp, positions=range(len(GROUP_ORDER)), widths=0.55, patch_artist=True,
                  showfliers=False, medianprops=dict(color="#222222", lw=1.3),
                  whiskerprops=dict(color="#666666", lw=0.9), capprops=dict(color="#666666", lw=0.9),
                  boxprops=dict(lw=0.8, edgecolor="#555555"))
for patch, g in zip(bpF["boxes"], GROUP_ORDER): patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.85)
for xi, g in enumerate(GROUP_ORDER):
    d = co_p[co_p["group"] == g]["PC1"].values
    axF.scatter(np.full_like(d, xi) + np.random.uniform(-0.12, 0.12, len(d)), d,
                s=16, c=GROUP_COLORS[g], edgecolors="white", linewidths=0.5, zorder=3)
axF.plot(range(len(GROUP_ORDER)), [np.mean(d) for d in data_by_grp], "-", color="#333333", lw=1.1, alpha=0.7, zorder=2)
axF.set_xticks(range(len(GROUP_ORDER))); axF.set_xticklabels(GROUP_ORDER)
axF.set_ylabel("TEMPTED PC1 subject loading"); axF.set_title(f"PC1 aligns with diet ({METHOD})")
row_p = load_t[(load_t.pipeline == PIPE_P2D2) & (load_t.component == "PC1")].iloc[0]
row_d = load_t[(load_t.pipeline == PIPE_DIRECT) & (load_t.component == "PC1")].iloc[0]
axF.text(0.03, 0.97,
         f"{METHOD} PC1:\n$\\eta^2$ = {row_p.eta_squared:.2f}, FDR p = {row_p.anova_fdr_bh:.1e}\n\n"
         f"Direct PC1:\n$\\eta^2$ = {row_d.eta_squared:.2f} ({star(row_d.anova_fdr_bh)})",
         transform=axF.transAxes, va="top", ha="left", fontsize=7.2,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.8))
panel_letter(axF, "F")

# ===== G : PC1 feature (taxon) loadings =====================================
axG = fig.add_subplot(gs[2, 0]); strip_ax(axG)
s = fb_p["PC1"].sort_values()
sel = pd.concat([s.head(SHOW_K), s.tail(SHOW_K)])          # neg pole (Vegan) + pos pole (EEN)
yy = np.arange(len(sel))
colors = [GROUP_COLORS["Vegan"] if v < 0 else GROUP_COLORS["EEN"] for v in sel.values]
axG.barh(yy, sel.values, color=colors, edgecolor="#555555", lw=0.5, alpha=0.9)
axG.set_yticks(yy); axG.set_yticklabels([sp_name(t) for t in sel.index], fontstyle="italic", fontsize=6.3)
axG.axvline(0, color="#333333", lw=0.8)
axG.set_xlabel("PC1 feature loading  $b_j$")
axG.set_title("PC1 microbial signature")
axG.set_ylim(-0.7, len(sel) - 0.3)
axG.text(0.97, 0.06, "Vegan-enriched\npole", transform=axG.transAxes, ha="right", va="bottom",
         fontsize=7, color=GROUP_COLORS["Vegan"], fontweight="bold")
axG.text(0.05, 0.955, "EEN-enriched\npole", transform=axG.transAxes, ha="left", va="top",
         fontsize=7, color=GROUP_COLORS["EEN"], fontweight="bold")
panel_letter(axG, "G", dx=-0.44)

# ===== H : temporal loading xi(t) ===========================================
axH = fig.add_subplot(gs[2, 1]); strip_ax(axH)
axH.plot(xi_d["time_interval"], xi_d["PC1"], color=PIPE_COLORS["direct"], lw=2.0, label="Direct TEMPTED")
axH.plot(xi_p["time_interval"], xi_p["PC1"], color=PIPE_COLORS["p2d2"], lw=2.0, label=PIPE_LABEL["p2d2"])
axH.axhline(0, color="#dddddd", lw=0.7, zorder=0)
axH.set_xlabel("Time (study day)"); axH.set_ylabel(r"PC1 temporal loading $\xi(t)$")
axH.set_title("PC1 temporal loading")
_ylo = min(xi_d["PC1"].min(), xi_p["PC1"].min()); _yhi = max(xi_d["PC1"].max(), xi_p["PC1"].max())
axH.set_ylim(_ylo - 0.03, _yhi + 0.03)
rng_d = xi_d["PC1"].max() - xi_d["PC1"].min(); rng_p = xi_p["PC1"].max() - xi_p["PC1"].min()
axH.text(0.03, 0.03, f"range: direct = {rng_d:.2f}\n{METHOD} = {rng_p:.3f} (flat / stable)",
         transform=axH.transAxes, va="bottom", ha="left", fontsize=7,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8))
axH.legend(loc="upper right", fontsize=7.0)
panel_letter(axH, "H")

# ===== I : PC1 microbial balance over time ==================================
axI = fig.add_subplot(gs[2, 2]); strip_ax(axI)
tx = np.arange(len(bins_p))
for g in GROUP_ORDER:
    sub = bal_p[grp_p == g]                 # (n_g, time)
    m = sub.mean(0); se = sub.std(0, ddof=1) / np.sqrt(sub.shape[0])
    axI.plot(tx, m, "-", color=GROUP_COLORS[g], lw=2.0, label=g, zorder=3)
    axI.fill_between(tx, m - se, m + se, color=GROUP_COLORS[g], alpha=0.18, lw=0, zorder=1)
axI.set_xticks([0, 5, 10, 15, 16]); axI.set_xticklabels(["0", "5", "10", "15", "PS"])
axI.set_xlabel("Time point"); axI.set_ylabel("PC1 microbial balance (log-ratio)")
axI.set_title("Signature separates diets over time")
axI.set_ylim(-2.05, 1.72)
# compact stats placed in the empty band between the separated group trajectories,
# so it annotates the between-group gap it quantifies without covering any data.
axI.text(0.5, 0.47,
         f"diet ANOVA:  F = {F_bal_p:.1f},  $\\eta^2$ = {eta_bal_p:.2f}  (p = {p_bal_p:.1e})\n"
         f"direct-PC1 signature:  $\\eta^2$ = {eta_bal_d:.2f}, {star(p_bal_d)}",
         transform=axI.transAxes, va="center", ha="center", fontsize=7.2, color="#333333")
axI.legend(loc="lower center", fontsize=7.0, ncol=3, columnspacing=0.8, handletextpad=0.3)
panel_letter(axI, "I")

# ----------------------------------------------------------------------------- save
for ext in ["png", "pdf", "svg"]:
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", facecolor="white"); print("saved", f"{OUT}.{ext}")
fig.savefig(f"{OUT}.tiff", bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
print("saved", f"{OUT}.tiff"); print("DONE")
