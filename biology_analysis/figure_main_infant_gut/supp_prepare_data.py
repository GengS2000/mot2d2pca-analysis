"""
Build all Supplementary Tables (ST1-ST6) and the backbone data for the
Supplementary Figures (SF1-SF8) from the four differential-abundance result
files, plus per-taxon prevalence read directly from the base_raw tensors.

Outputs:
  supp/tables/ST1_all_results_long.csv
  supp/tables/ST2_hitcounts_all_contrasts.csv
  supp/tables/TableS13_gold_standard_full.csv
  supp/tables/TableS14_effect_inflation.csv
  supp/tables/TableS15_imputed_only_prevalence.csv
  supp/tables/TableS16_run_provenance.csv
  supp/tables/TableS18_biological_evidence_ledger.csv
  supp/data/taxon_pairs_long.csv     # base<->imputed pivot; backbone for SF1-SF8
  supp/data/prevalence_base.csv      # per-taxon prevalence in base_raw

Run:  python supp_prepare_data.py
"""
import os
import numpy as np
import pandas as pd
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # ...\data
TAB = os.path.join(HERE, "supp", "tables")
DAT = os.path.join(HERE, "supp", "data")
for d in (TAB, DAT):
    os.makedirs(d, exist_ok=True)

RESULTS = {
    ("DIABIMMUNE", "LinDA"):
        "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/linda_longitudinal_results.csv",
    ("DIABIMMUNE", "MaAsLin2"):
        "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/maaslin2_longitudinal_results.csv",
    ("ECAM", "LinDA"):
        "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/linda_longitudinal_results.csv",
    ("ECAM", "MaAsLin2"):
        "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/maaslin2_longitudinal_results.csv",
}
TENSORS = {
    "DIABIMMUNE": "DIABIMMUNE_data/diabimmune_p2d2pca_netcdf_export/tensors/tensor_base_raw.nc",
    "ECAM": "ECAM_data/ECAM_otu_tensor_export/tensor_base_raw.nc",
}
ENGINE_SETTINGS = {
    "LinDA": "feature.dat.type=proportion; zero.handling=pseudo-count; ~subject_group+bin_c+(1|participant); p.adj=BH",
    "MaAsLin2": "normalization=NONE; transform=LOG; analysis_method=LM; fixed=subject_group+bin_c; random=participant; correction=BH",
}

# ----------------------------------------------------------------- load ST1 --
frames = []
for (cohort, method), rel in RESULTS.items():
    df = pd.read_csv(os.path.join(ROOT, rel))
    df.insert(0, "cohort", cohort)
    df.insert(1, "method", method)
    df = df.rename(columns={"dataset": "pipeline"})
    frames.append(df)
long = pd.concat(frames, ignore_index=True, sort=False)
long["pipeline"] = long["pipeline"].replace({"base_raw": "base", "p2d2pca_imputed": "imputed"})
long.to_csv(os.path.join(TAB, "ST1_all_results_long.csv"), index=False)
print("ST1", long.shape)

# ------------------------------------------------------ backbone: base<->imp --
KEYS = ["cohort", "method", "group_mode", "diet_col", "feature_id"]
long["diet_col"] = long["diet_col"].fillna("")
carry = [c for c in ["taxon", "taxon_label", "species_name", "Genus", "Family", "Phylum",
                     "reference_level", "contrast_level", "n_participants", "group_counts"]
         if c in long.columns]
base = long[long.pipeline == "base"][KEYS + ["effect_size", "p_value", "q_value", "n_samples"] + carry]
imp = long[long.pipeline == "imputed"][KEYS + ["effect_size", "p_value", "q_value", "n_samples"]]
pairs = base.merge(imp, on=KEYS, suffixes=("_base", "_imp"))
pairs["abseff_base"] = pairs["effect_size_base"].abs()
pairs["abseff_imp"] = pairs["effect_size_imp"].abs()
pairs["effect_fold"] = pairs["abseff_imp"] / pairs["abseff_base"].replace(0, np.nan)
pairs.to_csv(os.path.join(DAT, "taxon_pairs_long.csv"), index=False)
print("pairs", pairs.shape)


def sig(qs, thr):
    return (pd.to_numeric(qs, errors="coerce") < thr)


# ------------------------------------------------ ST2 hit counts all contrasts --
rows = []
for (cohort, method, gm, dc), g in pairs.groupby(["cohort", "method", "group_mode", "diet_col"]):
    rows.append(dict(cohort=cohort, method=method, group_mode=gm, diet_col=dc, n_taxa=len(g),
                     base_hits_q05=int(sig(g.q_value_base, 0.05).sum()),
                     imp_hits_q05=int(sig(g.q_value_imp, 0.05).sum()),
                     base_hits_q10=int(sig(g.q_value_base, 0.10).sum()),
                     imp_hits_q10=int(sig(g.q_value_imp, 0.10).sum()),
                     median_abseff_base=round(g.abseff_base.median(), 4),
                     median_abseff_imp=round(g.abseff_imp.median(), 4)))
st2 = pd.DataFrame(rows).sort_values(["cohort", "group_mode", "diet_col", "method"])
st2.to_csv(os.path.join(TAB, "ST2_hitcounts_all_contrasts.csv"), index=False)
print("ST2", st2.shape)

# ---------------------------------------------------- ST4 effect inflation ----
rows = []
for (cohort, method, gm), g in pairs.groupby(["cohort", "method", "group_mode"]):
    mb, mi = g.abseff_base.median(), g.abseff_imp.median()
    rows.append(dict(cohort=cohort, method=method, group_mode=gm, n_taxa=len(g),
                     median_abseff_base=round(mb, 4), median_abseff_imp=round(mi, 4),
                     fold=round(mi / mb, 2) if mb else np.nan,
                     iqr_abseff_base=round(g.abseff_base.quantile(.75) - g.abseff_base.quantile(.25), 4),
                     iqr_abseff_imp=round(g.abseff_imp.quantile(.75) - g.abseff_imp.quantile(.25), 4),
                     n_sig_base_q10=int(sig(g.q_value_base, 0.10).sum()),
                     n_sig_imp_q10=int(sig(g.q_value_imp, 0.10).sum())))
st4 = pd.DataFrame(rows).sort_values(["cohort", "group_mode", "method"])
st4.to_csv(os.path.join(TAB, "TableS14_effect_inflation.csv"), index=False)
print("ST4", st4.shape)

# ---------------------------------------------------- ST6 run provenance ------
prov = (long.groupby(["cohort", "method", "pipeline", "group_mode", "diet_col"])
        .agg(n_samples=("n_samples", "first"),
             n_participants=("n_participants", "first") if "n_participants" in long.columns else ("n_samples", "size"),
             group_counts=("group_counts", "first") if "group_counts" in long.columns else ("n_samples", "size"),
             formula=("formula", "first") if "formula" in long.columns else ("n_samples", "size"),
             n_taxa=("feature_id", "nunique"))
        .reset_index())
prov["engine_settings"] = prov["method"].map(ENGINE_SETTINGS)
prov.to_csv(os.path.join(TAB, "TableS16_run_provenance.csv"), index=False)
print("ST6", prov.shape)

# ------------------------------------------------ prevalence from tensors -----
def read_prevalence(path, cohort):
    ds = netCDF4.Dataset(path)
    dimset = {"participant", "taxon", "bin"}
    varname = None
    for v in ds.variables:
        if v in dimset:
            continue
        if set(ds.variables[v].dimensions) == dimset:
            varname = v
            break
    if varname is None:
        raise RuntimeError(f"No participant x taxon x bin variable in {path}")
    var = ds.variables[varname]
    eps = float(getattr(var, "eps", 0.001))
    arr = np.array(var[:].filled(np.nan) if hasattr(var[:], "filled") else var[:], dtype=float)
    order = [var.dimensions.index(d) for d in ("participant", "taxon", "bin")]
    arr = np.transpose(arr, order)  # participant, taxon, bin
    ntax = arr.shape[1]
    abund = np.power(10.0, arr) - eps
    present = np.isfinite(arr) & (abund > 1e-9)
    finite = np.isfinite(arr)
    # Full-grid recovery encodes unsampled participant-bin cells as all-zero rows.
    # Use observed samples for biological prevalence, while retaining full-grid
    # prevalence separately as an imputation diagnostic.
    observed_grid = present.any(axis=1)
    n_present = present.sum(axis=(0, 2))
    n_cells = np.repeat(observed_grid.sum(), ntax)
    n_grid_cells = finite.sum(axis=(0, 2))
    n_cells_safe = np.where(n_cells == 0, np.nan, n_cells)
    prevalence = n_present / n_cells_safe
    prevalence_full_grid = n_present / np.where(n_grid_cells == 0, np.nan, n_grid_cells)
    ds.close()
    return pd.DataFrame(dict(
        cohort=cohort,
        feature_id=[f"T{i+1:04d}" for i in range(ntax)],
        taxon_index=np.arange(1, ntax + 1),
        prevalence=np.round(prevalence, 4),
        prevalence_full_grid=np.round(prevalence_full_grid, 4),
        zero_fraction=np.round(1 - prevalence, 4),
        n_present=n_present.astype(int),
        n_cells=n_cells.astype(int),
        n_grid_cells=n_grid_cells.astype(int),
    ))


prev = pd.concat([read_prevalence(os.path.join(ROOT, p), c) for c, p in TENSORS.items()],
                 ignore_index=True)
prev.to_csv(os.path.join(DAT, "prevalence_base.csv"), index=False)
print("prevalence", prev.shape)

# --------------------------------------- ST5 imputed-only hits + prevalence ---
io = pairs[(~sig(pairs.q_value_base, 0.10)) & (sig(pairs.q_value_imp, 0.10))].copy()
io = io.merge(prev[["cohort", "feature_id", "prevalence", "prevalence_full_grid",
                    "zero_fraction", "n_present"]],
              on=["cohort", "feature_id"], how="left")
keep = ["cohort", "method", "group_mode", "diet_col", "feature_id"] + \
       [c for c in ["taxon", "species_name"] if c in io.columns] + \
       ["prevalence", "prevalence_full_grid", "zero_fraction", "n_present",
        "effect_size_base", "effect_size_imp",
        "effect_fold", "q_value_base", "q_value_imp"] + \
       [c for c in ["group_counts"] if c in io.columns]
st5 = io[keep].sort_values(["cohort", "method", "prevalence"])
st5.to_csv(os.path.join(TAB, "TableS15_imputed_only_prevalence.csv"), index=False)
print("ST5", st5.shape)

# ------------------------------------------------ ST3 gold-standard (full) ----
# All OTUs of the literature control genera/species across the 4 combos.
ec = long[long.cohort == "ECAM"].copy()
di = long[long.cohort == "DIABIMMUNE"].copy()
gs = []
gs.append(ec[(ec.group_mode == "delivery_group") &
             (ec.Genus.isin(["Bacteroides", "Bifidobacterium", "Veillonella"]))].assign(
    control_type="recover(Cesarean signature)"))
gs.append(ec[(ec.group_mode == "diet_diet_2_fd") &
             (ec.Family.isin(["Ruminococcaceae", "Lachnospiraceae", "Erysipelotrichaceae",
                              "Verrucomicrobiaceae"]))].assign(control_type="confirm(formula Clostridiales/Akkermansia)"))
gs.append(ec[ec.group_mode == "abx_group"].assign(control_type="negative(antibiotics)"))
gs.append(di[(di.group_mode == "diet_group") &
             (di.diet_col == "dietary_compound_carrot_start_age_month") &
             (di.feature_id.isin(["T0336", "T0351"]))].assign(control_type="suppress(contamination)"))
gs.append(di[di.group_mode == "abx_group"].assign(control_type="negative(antibiotics)"))
st3 = pd.concat(gs, ignore_index=True, sort=False)
front = ["control_type", "cohort", "method", "pipeline", "group_mode", "diet_col", "feature_id"]
st3 = st3[front + [c for c in st3.columns if c not in front]]
st3.to_csv(os.path.join(TAB, "TableS13_gold_standard_full.csv"), index=False)
print("ST3", st3.shape)

# ----------------------------------------- ST18 biological evidence ledger -----
# This is a claim-to-evidence bridge, not a numerical confidence score. It makes
# explicit what the joint observed/completed analysis adds over either pipeline
# alone and prevents completion-only threshold crossings from being promoted to
# biological discoveries.
st18 = pd.DataFrame([
    {
        "biological_question": "Delivery mode",
        "observed_grid_evidence": "Bacteroides deficit is strongest in year 1 and attenuates in year 2 (C/V 0.36 to 0.80)",
        "mot_completed_evidence": "Strongest Bacteroides OTU: LinDA q 0.063 to 0.020; MaAsLin2 q 0.071 to 0.082",
        "diagnostic_context": "Expected direction in most Bacteroides OTUs; original-cohort support",
        "incremental_value_over_direct_analysis": "Recovers a sparsity-masked cohort-level signal while the observed trajectory localizes its duration",
        "adjudicated_inference": "Cesarean delivery leaves a coherent but predominantly first-year Bacteroides imprint",
        "evidence_tier": "supported",
    },
    {
        "biological_question": "Antibiotic exposure",
        "observed_grid_evidence": "No q<0.05 taxon-level mean-abundance association in either cohort or engine",
        "mot_completed_evidence": "No q<0.05 hit; one DIABIMMUNE MaAsLin2 OTU at q=0.089",
        "diagnostic_context": "Boundary OTU prevalence 2.75%; not significant with LinDA",
        "incremental_value_over_direct_analysis": "Shows that no stable low-rank mean shift is recovered and isolates a sparse candidate for validation",
        "adjudicated_inference": "No reproducible time-averaged taxon-level main effect; transient or strain-level effects remain possible",
        "evidence_tier": "bounded null",
    },
    {
        "biological_question": "Feeding mode",
        "observed_grid_evidence": "Five LinDA and seven MaAsLin2 q<0.05 discoveries in ECAM",
        "mot_completed_evidence": "Discoveries fall to zero under both engines",
        "diagnostic_context": "Signal is not retained as a stable low-rank cohort offset",
        "incremental_value_over_direct_analysis": "Distinguishes a time-localized developmental deviation from a persistent compositional axis",
        "adjudicated_inference": "Feeding is associated with maturation timing rather than a stable taxonomic offset",
        "evidence_tier": "time-localized",
    },
    {
        "biological_question": "Individual solid-food timing",
        "observed_grid_evidence": "Carrot contrast includes chloroplast and mitochondrial reads; late-carrot n=3",
        "mot_completed_evidence": "Organellar q-values move toward 1, but sparse completion-only bacterial calls appear",
        "diagnostic_context": "Completion-only hits have median observed prevalence 2.8% and median effect inflation 6.2-fold",
        "incremental_value_over_direct_analysis": "Separates non-bacterial artifacts from reconstruction-sensitive sparse calls",
        "adjudicated_inference": "No individual food-timing contrast defines a reproducible taxon-level compositional axis",
        "evidence_tier": "not supported",
    },
])
st18.to_csv(os.path.join(TAB, "TableS18_biological_evidence_ledger.csv"), index=False)
print("ST18", st18.shape)

print("\nAll supplementary tables + backbone data written under supp/")
