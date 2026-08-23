"""Compare LinDA / MaAsLin2 / glmmPQL / ZIGMM on DIABIMMUNE and ECAM.

Merges the four DA methods' real-data results (base_raw vs p2d2pca_imputed) and
reports, per contrast, whether imputation changes the discovery set and whether
the methods agree on the direction of that change.

Focus contrasts: delivery mode, antibiotics, and solid-food introduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "four_method_comparison"
OUT.mkdir(parents=True, exist_ok=True)

STUDIES = {
    "ECAM": ROOT / "ECAM_data" / "artifacts" / "ECAM_otu_linda_longitudinal",
    "DIABIMMUNE": ROOT / "DIABIMMUNE_data" / "artifacts" / "subject_group_existing_methods_r",
}

# Canonical columns every method's table is reduced to.
KEYS = ["study", "da_method", "dataset", "group_mode", "feature_id"]
VALS = ["effect_size", "p_value", "q_value"]
TAXCOLS = ["Phylum", "Class", "Order", "Family", "Genus", "Species", "taxon"]


def _tax(df: pd.DataFrame) -> pd.DataFrame:
    """Keep whichever taxonomy columns this table happens to carry."""
    for c in TAXCOLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def load_linda_maaslin(study: str, art: Path) -> pd.DataFrame:
    frames = []
    for meth, fname in [("LinDA", "linda_longitudinal_results.csv"),
                        ("MaAsLin2", "maaslin2_longitudinal_results.csv")]:
        p = art / fname
        if not p.exists():
            print(f"  [warn] missing {p.name} for {study}")
            continue
        d = pd.read_csv(p)
        d["study"] = study
        d["da_method"] = meth
        # DIABIMMUNE splits diet into one contrast per compound via diet_col;
        # fold that into group_mode so every study has a single contrast key.
        if "diet_col" in d.columns:
            dc = d["diet_col"].fillna("").astype(str)
            d["group_mode"] = np.where(
                (d["group_mode"] == "diet_group") & (dc != ""),
                d["group_mode"] + "::" + dc,
                d["group_mode"],
            )
        d = _tax(d)
        frames.append(d[KEYS + VALS + TAXCOLS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_pql_zigmm(study: str, art: Path, task_map: dict[str, str]) -> pd.DataFrame:
    p = art / "glmmPQL_ZIGMM" / f"{study}_glmmPQL_ZIGMM_longitudinal_results.csv"
    if not p.exists():
        print(f"  [warn] missing {p.name} for {study}")
        return pd.DataFrame()
    d = pd.read_csv(p)
    d["study"] = study
    # The runner's task name is the pipeline stem, i.e. group_mode with the
    # diet_col appended ("diet_diet_2_bd" + "_diet_2"). task_map, built from the
    # LinDA table, maps each stem back to the canonical contrast key so all four
    # methods share one join key.
    gm = d["group_mode"].astype(str)
    unmapped = sorted(set(gm) - set(task_map))
    if unmapped:
        print(f"  [warn] {study}: {len(unmapped)} task(s) not in task_map: {unmapped[:5]}")
    d["group_mode"] = gm.map(task_map).fillna(gm)
    d = _tax(d)
    keep = KEYS + VALS + TAXCOLS
    if "fit_status" in d.columns:
        keep = keep + ["fit_status"]
    return d[keep]


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
def build_task_map(art: Path) -> dict[str, str]:
    """stem used by the glmmPQL/ZIGMM runner -> canonical contrast key.

    The runner names each task "<group_mode>_<diet_col>" (diet_col empty for
    non-diet contrasts), so recover the pairs from LinDA's own table.
    """
    p = art / "linda_longitudinal_results.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p, usecols=lambda c: c in {"group_mode", "diet_col"})
    if "diet_col" not in d.columns:
        d["diet_col"] = ""
    pairs = d.drop_duplicates()
    out: dict[str, str] = {}
    for gm, dc in zip(pairs["group_mode"].astype(str), pairs["diet_col"].fillna("").astype(str)):
        canonical = f"{gm}::{dc}" if (gm == "diet_group" and dc) else gm
        out[f"{gm}_{dc}" if dc else gm] = canonical
    return out


all_frames = []
for study, art in STUDIES.items():
    print(f"Loading {study} ...")
    tmap = build_task_map(art)
    all_frames.append(load_linda_maaslin(study, art))
    all_frames.append(load_pql_zigmm(study, art, tmap))
res = pd.concat([f for f in all_frames if len(f)], ignore_index=True)
res.to_csv(OUT / "all_methods_long.csv", index=False)
print(f"\nCombined: {len(res)} rows")
print(res.groupby(["study", "da_method"])["dataset"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Discovery counts per (study, contrast, method, dataset)
# ---------------------------------------------------------------------------
for a in (0.05, 0.10, 0.25):
    res[f"sig{int(a*100):02d}"] = res["q_value"] < a

counts = (res.groupby(["study", "group_mode", "da_method", "dataset"])
             .agg(n_taxa=("feature_id", "size"),
                  n_tested=("q_value", lambda s: int(s.notna().sum())),
                  sig05=("sig05", "sum"), sig10=("sig10", "sum"), sig25=("sig25", "sum"))
             .reset_index())
counts.to_csv(OUT / "discovery_counts.csv", index=False)

# Wide: base vs imputed side by side, with the delta that matters
wide = counts.pivot_table(index=["study", "group_mode", "da_method"],
                          columns="dataset",
                          values=["n_tested", "sig05", "sig10", "sig25"])
wide.columns = [f"{m}_{c}" for m, c in wide.columns]
wide = wide.reset_index()
for m in ["sig05", "sig10", "sig25"]:
    b, i = f"{m}_base_raw", f"{m}_p2d2pca_imputed"
    if b in wide.columns and i in wide.columns:
        wide[f"d_{m}"] = wide[i] - wide[b]
wide.to_csv(OUT / "discovery_counts_wide.csv", index=False)

print("\n=== discoveries at q<0.10: base_raw -> imputed, by method ===")
show = wide[["study", "group_mode", "da_method", "sig10_base_raw",
             "sig10_p2d2pca_imputed", "d_sig10"]].copy()
print(show.to_string(index=False))

# ---------------------------------------------------------------------------
# Direction agreement between methods on the imputation effect
# ---------------------------------------------------------------------------
print("\n=== per-contrast direction of change in discovery count (q<0.10) ===")
piv = wide.pivot_table(index=["study", "group_mode"], columns="da_method", values="d_sig10")
print(piv.to_string())
piv.to_csv(OUT / "direction_by_method.csv")

# ---------------------------------------------------------------------------
# Focus taxa: Bacteroides / Bifidobacterium (delivery-mode biology) and
# Chloroplast / Cyanobacteria (plant-derived reads from solid food)
# ---------------------------------------------------------------------------
def focus(df: pd.DataFrame, label: str, mask: pd.Series) -> pd.DataFrame:
    s = df[mask].copy()
    s["focus"] = label
    return s


gen = res["Genus"].astype(str)
cls = res["Class"].astype(str)
phy = res["Phylum"].astype(str)
blocks = [
    focus(res, "Bacteroides", gen.str.contains("Bacteroides", na=False)),
    focus(res, "Bifidobacterium", gen.str.contains("Bifidobacterium", na=False)),
    focus(res, "Chloroplast/Cyano", cls.str.contains("Chloroplast", na=False)
          | phy.str.contains("Cyanobacteria", na=False)),
]
foc = pd.concat(blocks, ignore_index=True)
foc.to_csv(OUT / "focus_taxa_long.csv", index=False)

print("\n=== focus taxa: min q per (study, contrast, method, dataset, focus) ===")
fs = (foc.groupby(["study", "group_mode", "focus", "da_method", "dataset"])
         .agg(min_q=("q_value", "min"),
              n_sig10=("sig10", "sum"),
              n_taxa=("feature_id", "size"))
         .reset_index())
fs.to_csv(OUT / "focus_taxa_summary.csv", index=False)
key = fs["group_mode"].isin(["delivery_group", "abx_group"]) | fs["group_mode"].str.startswith("diet")
print(fs[key & (fs.study == "ECAM")].to_string(index=False))

print(f"\nWrote outputs to: {OUT}")
