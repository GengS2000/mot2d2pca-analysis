"""
Render Supplementary Figures SF1-SF8 from the backbone tables produced by
supp_prepare_data.py.  Each SF is saved as svg + pdf + png under supp/figures/.

  SF1  full base-vs-imputed volcano/scatter (inflation is global & engine-indep.)
  SF2  exhaustive per-contrast hit counts (main panel b is representative)
  SF3  prevalence drives imputation-induced inflation (the mechanism)
  SF4  cross-method / cross-pipeline concordance (UpSet-lite; robustness tiers)
  SF5  Cesarean signature across ALL OTUs (recovery is not one lucky feature)
  SF6  pseudo-replication: rows fed to model vs independent participants
  SF7  feeding eb/fd mirror (breast retains immature community)
  SF8  antibiotics: three-engine q<0.05 null and T0487 direction check

Run:  python supp_prepare_data.py && python supp_make_figures.py
"""
import os
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

import figstyle as S
from figstyle import C

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "supp", "tables")
DAT = os.path.join(HERE, "supp", "data")
FIG = os.path.join(HERE, "supp", "figures")
os.makedirs(FIG, exist_ok=True)
S.apply_rc()

pairs = pd.read_csv(os.path.join(DAT, "taxon_pairs_long.csv"), low_memory=False)
prev = pd.read_csv(os.path.join(DAT, "prevalence_base.csv"))
st2 = pd.read_csv(os.path.join(TAB, "ST2_hitcounts_all_contrasts.csv"))
long = pd.read_csv(os.path.join(TAB, "ST1_all_results_long.csv"), low_memory=False)
MCOL = {"LinDA": C["linda"], "MaAsLin2": C["maaslin"]}
SIGCMAP = LinearSegmentedColormap.from_list("sig", ["#F3F1F7", C["filter"]])


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    manuscript_png = {
        "SF1_volcano_full": "FigureS9_volcano_full.png",
        "SF2_hitcounts_exhaustive": "FigureS10_hitcounts_exhaustive.png",
        "SF3_prevalence_inflation": "FigureS11_prevalence_inflation.png",
        "SF4_concordance_upset": "FigureS12_concordance_upset.png",
        "SF5_cesarean_full_taxonomy": "FigureS13_cesarean_full_taxonomy.png",
        "SF6_pseudoreplication": "FigureS14_pseudoreplication.png",
        "SF7_feeding_mirror": "FigureS15_feeding_mirror.png",
        "SF8_antibiotics_null": "FigureS16_antibiotics_null.png",
    }.get(name)
    if manuscript_png:
        fig.savefig(os.path.join(FIG, manuscript_png), bbox_inches="tight")
    plt.close(fig)


def nlq(q):
    return -np.log10(np.clip(pd.to_numeric(q, errors="coerce"), 1e-4, 1))


# ===================================================================== SF1 ====
def sf1():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.2))
    for r, cohort in enumerate(["DIABIMMUNE", "ECAM"]):
        d = pairs[pairs.cohort == cohort]
        ax = axes[r, 0]
        for m in ("LinDA", "MaAsLin2"):
            g = d[d.method == m]
            ax.scatter(g.abseff_base, g.abseff_imp, s=3, alpha=0.16, color=MCOL[m], edgecolors="none")
        lim = np.nanpercentile(np.r_[d.abseff_base, d.abseff_imp], 99.5)
        ax.plot([0, lim], [0, lim], color="0.5", lw=0.8, ls="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("|effect|  observed"); ax.set_ylabel("|effect|  MOT-completed")
        ax.set_title(f"{cohort}: effect inflation", loc="left")
        fold = (d.abseff_imp.median() / d.abseff_base.median())
        ax.text(0.05, 0.92, f"median inflation ~{fold:.1f}x\n(both engines above y=x)",
                transform=ax.transAxes, fontsize=6.4, va="top")
        S.clean(ax)

        ax = axes[r, 1]
        for m in ("LinDA", "MaAsLin2"):
            g = d[d.method == m]
            ax.scatter(nlq(g.q_value_base), nlq(g.q_value_imp), s=3, alpha=0.16, color=MCOL[m], edgecolors="none")
        thr = -np.log10(0.10)
        ax.axhline(thr, color="0.5", lw=0.7, ls=":"); ax.axvline(thr, color="0.5", lw=0.7, ls=":")
        ax.set_xlabel("$-\\log_{10}q$  observed")
        ax.set_ylabel("$-\\log_{10}q$  MOT-completed")
        ax.set_title(f"{cohort}: significance shift", loc="left")
        ax.text(thr + 0.05, ax.get_ylim()[1] * 0.02, "q=0.10", fontsize=5.5, color="0.5")
        S.clean(ax)
    axes[0, 0].legend(handles=[Line2D([0], [0], marker="o", ls="", color=MCOL[m], label=m) for m in MCOL],
                      loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "SF1_volcano_full")


# ===================================================================== SF2 ====
def sf2():
    def mat(cohort):
        d = st2[st2.cohort == cohort].copy()
        d["diet_col"] = d["diet_col"].fillna("")
        base_col = "base_hits_q10" if cohort == "DIABIMMUNE" else "base_hits_q05"
        imp_col = "imp_hits_q10" if cohort == "DIABIMMUNE" else "imp_hits_q05"
        piv = d.pivot_table(index=["group_mode", "diet_col"], columns="method",
                            values=[base_col, imp_col], fill_value=0)
        piv.columns = [f"{m}\n{'base' if 'base' in v else 'imp'}" for v, m in piv.columns]
        order = ["LinDA\nbase", "LinDA\nimp", "MaAsLin2\nbase", "MaAsLin2\nimp"]
        piv = piv[[c for c in order if c in piv.columns]]
        piv.columns = [c.replace("\nbase", "\nobserved").replace("\nimp", "\nMOT-completed")
                       for c in piv.columns]
        if cohort == "DIABIMMUNE":
            piv.index = [
                dc.replace("dietary_compound_", "").replace("_start_age_month", "")
                if dc else gm for gm, dc in piv.index
            ]
        else:
            piv.index = [
                gm.replace("diet_diet_", "diet_")
                  .replace("abx_group", "antibiotics")
                  .replace("delivery_group", "delivery")
                for gm, _ in piv.index
            ]
        return piv

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 6.6),
                             gridspec_kw=dict(width_ratios=[1, 0.62]))
    for ax, cohort in zip(axes, ["DIABIMMUNE", "ECAM"]):
        piv = mat(cohort)
        vmax = max(piv.values.max(), 1)
        ax.imshow(piv.values, cmap=SIGCMAP, aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=6)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=6)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = int(piv.values[i, j])
                ax.text(j, i, v, ha="center", va="center", fontsize=5.6,
                        color="white" if v > vmax * 0.55 else C["ink"])
        threshold = "q<0.10" if cohort == "DIABIMMUNE" else "q<0.05"
        ax.set_title(f"{cohort}  (significant taxa, {threshold})",
                     loc="left", fontsize=8)
        ax.set_xticks(np.arange(-.5, piv.shape[1]), minor=True)
        ax.set_yticks(np.arange(-.5, piv.shape[0]), minor=True)
        ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    save(fig, "SF2_hitcounts_exhaustive")


# ===================================================================== SF3 ====
def sf3():
    d = pairs[(pairs.cohort == "DIABIMMUNE") & (pairs.group_mode == "diet_group") &
              (pairs.method == "LinDA")].merge(
        prev[prev.cohort == "DIABIMMUNE"][["feature_id", "prevalence"]], on="feature_id", how="left")
    d = d[np.isfinite(d.effect_fold) & (d.prevalence > 0)].copy()
    d["logfold"] = np.log10(d.effect_fold)
    io = (~(pd.to_numeric(d.q_value_base, errors="coerce") < 0.10)) & \
         (pd.to_numeric(d.q_value_imp, errors="coerce") < 0.10)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.scatter(d.prevalence[~io], d.logfold[~io], s=5, alpha=0.25, color="0.6",
               edgecolors="none", label="other taxa")
    ax.scatter(d.prevalence[io], d.logfold[io], s=22, color=C["false"], edgecolors="white",
               linewidths=0.4, label="completion-only hit", zorder=3)
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("prevalence in observed-grid samples")
    ax.set_ylabel("$\\log_{10}$ effect change (MOT-completed / observed)")
    io_folds = d.loc[io, "effect_fold"].replace([np.inf, -np.inf], np.nan).dropna()
    fold_note = (f"median {io_folds.median():.1f}x; "
                 f"range {io_folds.min():.1f}-{io_folds.max():.1f}x")
    ax.text(0.02, 0.05, "completion-only hits cluster at low prevalence\n" + fold_note,
            transform=ax.transAxes, fontsize=6.4, color=C["false"], va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    S.clean(ax)
    fig.tight_layout()
    save(fig, "SF3_prevalence_inflation")


# ===================================================================== SF4 ====
def sf4():
    d = pairs[(pairs.cohort == "DIABIMMUNE") & (pairs.group_mode == "diet_group")]
    sets = {}
    for m in ("LinDA", "MaAsLin2"):
        g = d[d.method == m]
        for tag, col in (("base", "q_value_base"), ("imp", "q_value_imp")):
            key = f"{m}\n{'observed' if tag == 'base' else 'MOT-completed'}"
            sig = g[pd.to_numeric(g[col], errors="coerce") < 0.10]
            sets[key] = set(zip(sig.diet_col, sig.feature_id))
    setnames = list(sets.keys())
    universe = set().union(*sets.values())
    from itertools import product
    patterns = {}
    for elem in universe:
        patt = tuple(elem in sets[s] for s in setnames)
        patterns.setdefault(patt, 0)
        patterns[patt] += 1
    items = sorted([p for p in patterns.items() if any(p[0])], key=lambda kv: -kv[1])

    fig = plt.figure(figsize=(5.6, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.55], width_ratios=[0.28, 1.0],
                          hspace=0.06, wspace=0.05)
    axbar = fig.add_subplot(gs[0, 1]); axmat = fig.add_subplot(gs[1, 1]); axset = fig.add_subplot(gs[1, 0])
    xs = range(len(items))
    axbar.bar(xs, [c for _, c in items], color=C["imputed"], width=0.7)
    for x, (_, c) in zip(xs, items):
        axbar.text(x, c, str(c), ha="center", va="bottom", fontsize=6)
    axbar.set_ylabel("intersection\nsize", fontsize=7); axbar.set_xticks([]); S.clean(axbar)
    for x, (patt, _) in zip(xs, items):
        for yi, present in enumerate(patt):
            axmat.plot(x, yi, "o", ms=7, color=C["ink"] if present else "#DADDE1")
        ons = [yi for yi, p in enumerate(patt) if p]
        if len(ons) > 1:
            axmat.plot([x, x], [min(ons), max(ons)], color=C["ink"], lw=1.2, zorder=0)
    axmat.set_xlim(-0.5, len(items) - 0.5); axmat.set_ylim(-0.5, len(setnames) - 0.5)
    axmat.set_xticks([]); axmat.set_yticks([]); axmat.axis("off")
    axset.set_ylim(-0.5, len(setnames) - 0.5); axset.set_xlim(0, 1)
    for yi, s in enumerate(setnames):
        axset.text(0.95, yi, s, ha="right", va="center", fontsize=6)
    axset.axis("off")
    fig.text(0.14, 0.02,
             "cross-engine + cross-representation overlap = higher support;  "
             "completion-only overlap = reconstruction-sensitive",
             fontsize=6, color="0.35")
    save(fig, "SF4_concordance_upset")


# ===================================================================== SF5 ====
def sf5():
    d = long[(long.cohort == "ECAM") & (long.group_mode == "delivery_group") &
             (long.Genus.isin(["Bacteroides", "Bifidobacterium", "Veillonella"]))].copy()
    d["combo"] = d.method + "\n" + d.pipeline.map(
        {"base": "observed", "imputed": "MOT-completed"})
    d["label"] = d.species_name.fillna(d.taxon).astype(str) + " (" + d.feature_id + ")"
    combos = ["LinDA\nobserved", "LinDA\nMOT-completed",
              "MaAsLin2\nobserved", "MaAsLin2\nMOT-completed"]
    d = d.sort_values(["Genus", "label"])
    labels = list(dict.fromkeys(d.label))
    M = np.full((len(labels), 4), np.nan); D = np.zeros((len(labels), 4))
    for _, r in d.iterrows():
        i = labels.index(r.label); j = combos.index(r.combo)
        M[i, j] = -np.log10(np.clip(r.q_value, 1e-4, 1)); D[i, j] = np.sign(r.effect_size)
    fig, ax = plt.subplots(figsize=(5.2, max(3.2, 0.24 * len(labels))))
    norm = Normalize(0, 3)
    for i in range(len(labels)):
        for j in range(4):
            if np.isnan(M[i, j]):
                continue
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.9, facecolor=SIGCMAP(norm(M[i, j])),
                                       edgecolor="#D5D8DC", lw=0.4))
            if M[i, j] > -np.log10(0.10):
                ax.text(j + 0.47, i + 0.45, "▼" if D[i, j] < 0 else "▲", ha="center", va="center",
                        fontsize=6, color="white" if M[i, j] > 1.5 else C["ink"])
    ax.set_xlim(0, 4); ax.set_ylim(0, len(labels))
    ax.set_xticks([j + 0.47 for j in range(4)]); ax.set_xticklabels(combos, fontsize=6)
    ax.set_yticks([i + 0.45 for i in range(len(labels))]); ax.set_yticklabels(labels, fontsize=5.6)
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SIGCMAP), ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("$-\\log_{10}q$", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    save(fig, "SF5_cesarean_full_taxonomy")


# ===================================================================== SF6 ====
def sf6():
    prov = long.groupby(["cohort", "group_mode", "pipeline"]).agg(
        n_samples=("n_samples", "first"),
        n_participants=("n_participants", "first")).reset_index()
    canonical = [("DIABIMMUNE", "abx_group"), ("ECAM", "delivery_group")]
    labels = ["DIABIMMUNE", "ECAM"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.4, 3.4), gridspec_kw=dict(width_ratios=[1.15, 0.75]))
    x = np.arange(len(canonical)); w = 0.34
    for k, (pip, col) in enumerate([("base", C["base"]), ("imputed", C["imputed"])]):
        vals = [prov[(prov.cohort == cohort) & (prov.group_mode == contrast) &
                     (prov.pipeline == pip)].n_samples.iloc[0]
                for cohort, contrast in canonical]
        ax1.bar(x + (k - 0.5) * w, vals, width=w, color=col,
                label="observed" if pip == "base" else "MOT-completed")
        for xp, v in zip(x + (k - 0.5) * w, vals):
            ax1.text(xp, v, str(int(v)), ha="center", va="bottom", fontsize=6)
    nparts = [prov[(prov.cohort == cohort) & (prov.group_mode == contrast)]
                   .n_participants.max() for cohort, contrast in canonical]
    ax1.scatter(x, nparts, s=28, marker="D", color=C["false"], zorder=4,
                label="participants")
    for xp, v in zip(x, nparts):
        ax1.text(xp, v + 110, str(int(v)), ha="center", va="bottom",
                 fontsize=6, color=C["false"])
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylabel("rows fed to model / participants"); ax1.set_ylim(0, 5600)
    ax1.set_title("Full-grid completion expands modeled rows", loc="left", fontsize=8)
    ax1.legend(frameon=False, fontsize=6, loc="upper center", ncol=2); S.clean(ax1)

    ratios = []
    reconstructed = []
    for cohort, contrast in canonical:
        g = prov[(prov.cohort == cohort) &
                 (prov.group_mode == contrast)].set_index("pipeline")
        base_n, imp_n = g.loc["base", "n_samples"], g.loc["imputed", "n_samples"]
        ratios.append(imp_n / base_n)
        reconstructed.append((imp_n - base_n) / imp_n)
    bars = ax2.bar(x, ratios, width=0.52, color=C["imputed"])
    for bar, ratio, frac in zip(bars, ratios, reconstructed):
        ax2.text(bar.get_x() + bar.get_width() / 2, ratio + 0.15,
                 f"{ratio:.2f}x\n{frac:.1%} grid-added",
                 ha="center", va="bottom", fontsize=6)
    ax2.axhline(1, color="0.5", ls=":", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylabel("MOT-completed / observed model rows"); ax2.set_ylim(0, 8.5)
    ax2.set_title("Expansion differs by cohort", loc="left", fontsize=8)
    S.clean(ax2)
    fig.tight_layout()
    save(fig, "SF6_pseudoreplication")


# ===================================================================== SF7 ====
def sf7():
    eb = long[(long.cohort == "ECAM") & (long.group_mode == "diet_diet_3_eb") & (long.pipeline == "base")]
    fd = long[(long.cohort == "ECAM") & (long.group_mode == "diet_diet_2_fd") & (long.pipeline == "base")]
    taxa = ["[Eubacterium] dolichum", "[Ruminococcus] gnavus", "Clostridium aldenense", "Oscillospira",
            "Blautia producta", "Anaerotruncus", "Coprobacillus cateniformis", "Akkermansia muciniphila",
            "Lactobacillus zeae"]

    def strongest(df, sp):
        s = df[df.species_name == sp]
        if not len(s):
            return np.nan, np.nan
        r = s.loc[pd.to_numeric(s.q_value, errors="coerce").idxmin()]
        return r.effect_size, r.q_value

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    y = np.arange(len(taxa))
    for i, sp in enumerate(taxa):
        e_eb, q_eb = strongest(eb, sp); e_fd, q_fd = strongest(fd, sp)
        for e, q, col, dx in ((e_eb, q_eb, C["down"], -0.16), (e_fd, q_fd, C["up"], 0.16)):
            if np.isnan(e):
                continue
            sigmark = pd.to_numeric(pd.Series([q]), errors="coerce").iloc[0] < 0.05
            ax.scatter(e, i + dx, s=70 if sigmark else 30, color=col,
                       edgecolors="black" if sigmark else "none", linewidths=0.6, zorder=3)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(taxa, fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("effect size  (— exclusive breastfeeding ↑ / formula ↑ +)")
    # figure title omitted (goes in the legend)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=C["down"], label="exclusive breastfeeding (eb)"),
                       Line2D([0], [0], marker="o", ls="", color=C["up"], label="formula-dominant (fd)"),
                       Line2D([0], [0], marker="o", ls="", mfc="none", mec="black", label="q<0.05")],
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=6)
    S.clean(ax)
    fig.tight_layout()
    save(fig, "SF7_feeding_mirror")


# ===================================================================== SF8 ====
def sf8():
    project_root = Path(HERE).parents[2]
    script = project_root / "supplementary" / "figures" / "make_figure_s9_antibiotic.py"
    if not script.exists():
        raise FileNotFoundError(f"Revised three-engine Figure S9 generator not found: {script}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    for fn in (sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8):
        fn()
        print("wrote", fn.__name__)
    print("All SF1-SF8 written to supp/figures/")
