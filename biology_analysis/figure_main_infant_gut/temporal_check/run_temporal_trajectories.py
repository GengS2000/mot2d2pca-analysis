#!/usr/bin/env python
"""
Temporal-trajectory extraction from the exported tensors.

Central result (backs the redrawn main Panel e and SF9/SF10):
the observed and MOT-(2D)^2 PCA-completed grids provide complementary evidence.
Completion recovers cohort-level low-rank structure but regularizes temporal
detail: it compresses developmental succession and represents an attenuating
cesarean Bacteroides deficit as a persistent offset. The observed grid therefore
remains the timing reference, while the completed grid tests structural
coherence.

Outputs
-------
panel_data/panel_e_ecam_bacteroides.csv   main Panel-e trajectories (observed vs imputed)
panel_data/panel_e_persistence.csv        main Panel-e C/V persistence ratios
panel_data/panel_e_verdict.csv            main Panel-e four-exposure verdict strip
supp/tables/TableS17_temporal_fidelity.csv persistence ratios + succession errors
supp/figures/SF9_succession.{png,pdf,svg} DIABIMMUNE full succession vs Yassour
supp/figures/SF10_delivery.{png,pdf,svg}  ECAM delivery trajectories observed vs imputed
temporal_check/data/*.csv                  auditable per-bin trajectories

Run:  python temporal_check/run_temporal_trajectories.py
Requires PYTHONPATH to reach numpy/pandas/xarray/matplotlib (Windows: D:\\data\\.python_deps).
"""
import os, sys, numpy as np, pandas as pd, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, ".."))          # figure_main_infant_gut
ROOT = os.path.abspath(os.path.join(FIGDIR, ".."))          # biology_analysis
sys.path.insert(0, FIGDIR)
import figstyle as S
from figstyle import C
S.apply_rc()

PD_DIR = os.path.join(FIGDIR, "panel_data")
SF_DIR = os.path.join(FIGDIR, "supp", "figures")
ST_DIR = os.path.join(FIGDIR, "supp", "tables")
DAT = os.path.join(HERE, "data")
for d in (PD_DIR, SF_DIR, ST_DIR, DAT):
    os.makedirs(d, exist_ok=True)

VAG, CES = "#1f7a4d", "#b3421a"


def main_var(ds):
    for v in ds.data_vars:
        if ds[v].ndim == 3:
            return v
    return list(ds.data_vars)[0]


def lin_relabund(slice_log):
    lin = 10.0 ** slice_log
    s = lin.sum()
    return lin / s if s > 0 else lin


def field(lin, key):
    for part in str(lin).split(";"):
        part = part.strip()
        if part.startswith(key):
            return part[len(key):].strip()
    return ""


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(SF_DIR, f"{name}.{ext}"), bbox_inches="tight")
    manuscript_png = {
        "SF9_succession": "FigureS17_succession.png",
        "SF10_delivery": "FigureS18_delivery.png",
    }.get(name)
    if manuscript_png:
        fig.savefig(os.path.join(SF_DIR, manuscript_png), bbox_inches="tight")
    plt.close(fig)


# ============================================================ DIABIMMUNE ======
def diabimmune():
    base = os.path.join(ROOT, "DIABIMMUNE_data", "diabimmune_p2d2pca_netcdf_export", "tensors")
    b = xr.open_dataset(os.path.join(base, "tensor_base_raw.nc"))
    i = xr.open_dataset(os.path.join(base, "tensor_p2d2pca_imputed.nc"))
    rep = xr.open_dataset(os.path.join(base, "replace_mask.nc"))["replace_mask"].values.astype(bool)
    B, I = b[main_var(b)].values, i[main_var(i)].values
    months = np.array([float(x) for x in b[main_var(b)]["bin"].values])
    taxa = [str(t) for t in b[main_var(b)]["taxon"].values]

    tax = pd.read_csv(os.path.join(ROOT, "DIABIMMUNE_data", "taxonomy.tsv"),
                      sep="\t").set_index("Feature ID")["Taxon"]
    fam = np.array([field(tax.get(t, ""), "f__") for t in taxa])
    gen = np.array([field(tax.get(t, ""), "g__") for t in taxa])

    groups = {
        "Enterobacteriaceae": fam == "Enterobacteriaceae",
        "Bifidobacterium":    gen == "Bifidobacterium",
        "Bacteroides":        gen == "Bacteroides",
        "Lachnospiraceae":    fam == "Lachnospiraceae",
        "Ruminococcaceae":    fam == "Ruminococcaceae",
        "Clostridium":        gen == "Clostridium",
        "Akkermansia":        gen == "Akkermansia",
        "Faecalibacterium":   gen == "Faecalibacterium",
    }
    bench = {  # Yassour 2016 anchor points (month, approx %)
        "Enterobacteriaceae": [(2, 25), (18, 1)],
        "Bifidobacterium":    [(2, 16), (18, 3)],
        "Lachnospiraceae":    [(2, 4), (12, 16)],
        "Ruminococcaceae":    [(2, 0), (12, 16)],
        "Akkermansia":        [(2, 0), (24, 2)],
        "Faecalibacterium":   [(2, 0), (12, 3)],
    }
    obs_slice = (~rep).any(axis=1)

    rows = []
    for name, mask in groups.items():
        for bi, m in enumerate(months):
            ps = np.where(obs_slice[:, bi])[0]
            base_mean = (np.mean([lin_relabund(B[p, :, bi])[mask].sum() for p in ps])
                         if len(ps) else np.nan)
            rai = np.mean([lin_relabund(I[p, :, bi])[mask].sum() for p in range(I.shape[0])])
            rows.append(dict(group=name, month=m, base_frac=base_mean, base_n=len(ps),
                             imp_frac=rai))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DAT, "diab_succession_trajectories.csv"), index=False)

    # ---- SF9 figure ----
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 4.6), sharex=True)
    for ax, name in zip(axes.ravel(), groups):
        d = df[df.group == name].sort_values("month")
        ax.plot(d.month, d.imp_frac * 100, "-", color=C["imputed"], lw=1.6,
                label="MOT-completed grid")
        db = d[d.base_n >= 3]
        ax.plot(db.month, db.base_frac * 100, "o", color=C["base"], ms=3,
                label="observed grid")
        for (mm, pc) in bench.get(name, []):
            ax.plot(mm, pc, "*", color=C["false"], ms=10, zorder=5)
        ax.set_title(name, fontsize=7.5)
        ax.set_xlim(0, 30)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0, 0].legend(fontsize=5.6, loc="upper right", frameon=False)
    for ax in axes[1, :]:
        ax.set_xlabel("age (months)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean rel. abundance (%)")
    fig.tight_layout()
    save(fig, "SF9_succession")

    # ---- fidelity errors vs Yassour ----
    fid = []
    for name, pts in bench.items():
        dgi = df[df.group == name].set_index("month")
        for (mm, pc) in pts:
            if mm in dgi.index:
                fid.append(dict(cohort="DIABIMMUNE", group=name, month=mm, gold_pct=pc,
                                base_pct=round(dgi.loc[mm, "base_frac"] * 100, 2),
                                imp_pct=round(dgi.loc[mm, "imp_frac"] * 100, 2),
                                base_abs_err=round(abs(dgi.loc[mm, "base_frac"] * 100 - pc), 2),
                                imp_abs_err=round(abs(dgi.loc[mm, "imp_frac"] * 100 - pc), 2),
                                window="early" if mm <= 2 else "late"))
    return df, pd.DataFrame(fid)


# ============================================================ ECAM ============
def ecam():
    base = os.path.join(ROOT, "ECAM_data", "ECAM_otu_tensor_export")
    b = xr.open_dataset(os.path.join(base, "tensor_base_raw.nc"))
    i = xr.open_dataset(os.path.join(base, "tensor_p2d2pca_imputed.nc"))
    obs = xr.open_dataset(os.path.join(base, "participant_bin_observed_sample_mask.nc"))
    obsm = obs[list(obs.data_vars)[0]].values.astype(bool)
    B, I = b[main_var(b)].values, i[main_var(i)].values
    months = np.array([float(x) for x in b[main_var(b)]["bin"].values])
    parts = [str(p) for p in b[main_var(b)]["participant"].values]

    ti = pd.read_csv(os.path.join(base, "tensor_taxon_index.csv"))
    gen = ti.sort_values("taxon_index")["Genus"].fillna("").astype(str).values
    sm = pd.read_csv(os.path.join(base, "subject_metadata.csv"))
    delivery = dict(zip(sm["participant"].astype(str), sm["diagnosis"].astype(str)))
    grp = np.array([delivery.get(p, "NA") for p in parts])

    targets = {"Bacteroides": gen == "Bacteroides",
               "Bifidobacterium": gen == "Bifidobacterium",
               "Veillonella": gen == "Veillonella"}
    mo = np.floor(months).astype(int)
    uniq_mo = np.unique(mo)
    rows = []
    for name, mask in targets.items():
        for delv in ["Vaginal", "Cesarean"]:
            gi = np.where(grp == delv)[0]
            for mm in uniq_mo:
                bins_here = np.where(mo == mm)[0]
                bvals = [lin_relabund(B[p, :, bi])[mask].sum()
                         for p in gi for bi in bins_here if obsm[p, bi]]
                ivals = [lin_relabund(I[p, :, bi])[mask].sum()
                         for p in gi for bi in bins_here]
                rows.append(dict(group=name, delivery=delv, month=mm,
                                 base_frac=np.mean(bvals) if bvals else np.nan, base_n=len(bvals),
                                 imp_frac=np.mean(ivals) if ivals else np.nan, imp_n=len(ivals)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DAT, "ecam_delivery_trajectories.csv"), index=False)

    # ---- main Panel-e input: Bacteroides only ----
    bac = df[df.group == "Bacteroides"][["delivery", "month", "base_frac", "imp_frac", "base_n"]]
    bac.to_csv(os.path.join(PD_DIR, "panel_e_ecam_bacteroides.csv"), index=False)

    # ---- persistence ratios (C/V) year1 vs year2, observed vs imputed ----
    pers = []
    for name in targets:
        d = df[df.group == name]
        for lo, hi, lbl in [(0, 12, "y1_0-12mo"), (12, 24, "y2_12-24mo")]:
            def wm(g, col, ncol):
                gg = d[(d.delivery == g) & (d.month >= lo) & (d.month < hi)].dropna(subset=[col])
                return np.average(gg[col], weights=gg[ncol]) if gg[ncol].sum() else np.nan
            ov = wm("Vaginal", "base_frac", "base_n"); oc = wm("Cesarean", "base_frac", "base_n")
            iv = wm("Vaginal", "imp_frac", "imp_n");  ic = wm("Cesarean", "imp_frac", "imp_n")
            pers.append(dict(cohort="ECAM", genus=name, window=lbl,
                             observed_CoverV=round(oc / ov, 3) if ov else np.nan,
                             imputed_CoverV=round(ic / iv, 3) if iv else np.nan))
    pers = pd.DataFrame(pers)
    pers.to_csv(os.path.join(PD_DIR, "panel_e_persistence.csv"), index=False)

    # ---- SF10 figure ----
    col = {"Vaginal": VAG, "Cesarean": CES}
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), sharex=True)
    for j, name in enumerate(targets):
        axb, axi = axes[0, j], axes[1, j]
        for delv in ["Vaginal", "Cesarean"]:
            d = df[(df.group == name) & (df.delivery == delv)].sort_values("month")
            db = d[d.base_n >= 2]
            axb.plot(db.month, db.base_frac * 100, "o-", color=col[delv], ms=3, lw=1.1, label=delv)
            axi.plot(d.month, d.imp_frac * 100, "-", color=col[delv], lw=1.8, label=delv)
        axb.set_title(f"{name} — observed grid", fontsize=7.5)
        axi.set_title(f"{name} — MOT-completed grid", fontsize=7.5)
        for ax in (axb, axi):
            ax.axvline(12, color="0.55", ls=":", lw=0.8)
            ax.grid(alpha=0.25, lw=0.4); ax.set_xlim(0, 24)
        axi.set_xlabel("age (months)")
    axes[0, 0].legend(fontsize=6, frameon=False); axes[0, 0].set_ylabel("rel. abundance (%)")
    axes[1, 0].set_ylabel("rel. abundance (%)")
    fig.tight_layout()
    save(fig, "SF10_delivery")
    return df, pers


def write_verdict():
    """Four-exposure graded verdict strip for main Panel e (auditable, interpretive)."""
    verdict = [
        dict(exposure="Delivery", observed="Bacteroides deficit, year 1",
             completion="near-threshold signal recovered",
             call="coherent signal; transient timing", role="true"),
        dict(exposure="Antibiotics", observed="no mean-abundance shift",
             completion="no q<.05 hit; 1 sparse boundary OTU",
             call="no stable mean shift", role="null"),
        dict(exposure="Feeding", observed="time-localized signal",
             completion="not retained as a low-rank offset",
             call="maturation tempo, not persistent axis", role="tempo"),
        dict(exposure="Solid-food timing", observed="plant reads + small subgroups",
             completion="artifacts removed; sparse calls flagged",
             call="no reproducible taxon-level axis", role="false"),
    ]
    df = pd.DataFrame(verdict)
    df.to_csv(os.path.join(PD_DIR, "panel_e_verdict.csv"), index=False)


if __name__ == "__main__":
    dfd, fid = diabimmune()
    dfe, pers = ecam()
    write_verdict()

    # ---- TableS17 temporal fidelity ----
    fid.to_csv(os.path.join(DAT, "diab_succession_fidelity.csv"), index=False)
    combo = os.path.join(ST_DIR, "TableS17_temporal_fidelity.csv")
    with open(combo, "w", newline="") as fh:
        fh.write("# TableS17 temporal fidelity of full-grid completion\n")
        fh.write("# Part A: ECAM cesarean/vaginal (C/V) Bacteroides persistence ratio, observed vs imputed\n")
        pers.to_csv(fh, index=False)
        fh.write("# Part B: DIABIMMUNE succession absolute error vs Yassour 2016 benchmark, base vs imputed (%)\n")
        fid.to_csv(fh, index=False)

    print("=== ECAM Bacteroides persistence (C/V) ===")
    print(pers[pers.genus == "Bacteroides"].to_string(index=False))
    print("\n=== DIABIMMUNE early-window fidelity (|error vs Yassour|, %) ===")
    e = fid[fid.window == "early"]
    print(e[["group", "base_abs_err", "imp_abs_err"]].to_string(index=False))
    print(f"\nmean early error  base={e.base_abs_err.mean():.1f}  imputed={e.imp_abs_err.mean():.1f}")
    print("\nWrote panel_data/panel_e_*.csv, supp/tables/TableS17, supp/figures/SF9,SF10")
