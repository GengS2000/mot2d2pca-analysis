"""
Prepare tidy per-panel data tables from the four differential-inference result
files (LinDA x MaAsLin2  X  DIABIMMUNE x ECAM).

Outputs (also serve as machine-readable Supplementary Tables):
  panel_data/panel_b_hitcounts.csv    -> base->imputed significant-taxa counts
  panel_data/panel_c_decoupling.csv   -> DIABIMMUNE diet: effect inflation vs significance
  panel_data/panel_d_goldpanel.csv    -> literature gold-standard panel (q + effect, 4 combos)

Run from anywhere:  python prepare_panel_data.py
DATA_ROOT is inferred as the parent of this script's folder (i.e. D:\data).
"""
import os
import math
import csv
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(HERE)              # ...\data
OUT = os.path.join(HERE, "panel_data")
os.makedirs(OUT, exist_ok=True)

FILES = {
    ("DIABIMMUNE", "LinDA"):
        "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/linda_longitudinal_results.csv",
    ("DIABIMMUNE", "MaAsLin2"):
        "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/maaslin2_longitudinal_results.csv",
    ("ECAM", "LinDA"):
        "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/linda_longitudinal_results.csv",
    ("ECAM", "MaAsLin2"):
        "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/maaslin2_longitudinal_results.csv",
}


def load(rel):
    with open(os.path.join(DATA_ROOT, rel), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def median(vals):
    v = [x for x in vals if x is not None]
    return statistics.median(v) if v else float("nan")


D = {k: load(v) for k, v in FILES.items()}

# Native analysis thresholds (kept per dataset, as in the original analysis).
THR = {"DIABIMMUNE": 0.10, "ECAM": 0.05}


def sub(rows, gm, ds):
    return [r for r in rows if r["group_mode"] == gm and r["dataset"] == ds]


def n_sig(rows, gm, ds, thr):
    return sum(1 for r in sub(rows, gm, ds) if (fnum(r["q_value"]) or 1) < thr)


# ---------------------------------------------------------------------------
# Panel b : base -> imputed significant-taxa counts (the engine-dependent flip)
# ---------------------------------------------------------------------------
# Curated, high-signal contrasts (one row per dataset+contrast+method).
b_contrasts = [
    ("DIABIMMUNE", "diet_group", "Solid-food timing (22 foods)"),
    ("ECAM", "delivery_group", "Delivery mode"),
    ("ECAM", "diet_diet_2_bd", "Feeding (breast/formula)"),
    ("DIABIMMUNE", "abx_group", "Antibiotics (DIABIMMUNE)"),
    ("ECAM", "abx_group", "Antibiotics (ECAM)"),
]
with open(os.path.join(OUT, "panel_b_hitcounts.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "contrast", "label", "method", "q_threshold", "base_hits", "imputed_hits"])
    for dataset, gm, label in b_contrasts:
        thr = THR[dataset]
        for method in ("LinDA", "MaAsLin2"):
            rows = D[(dataset, method)]
            w.writerow([dataset, gm, label, method, thr,
                        n_sig(rows, gm, "base_raw", thr),
                        n_sig(rows, gm, "p2d2pca_imputed", thr)])

# ---------------------------------------------------------------------------
# Panel c : DIABIMMUNE diet -- effect-size inflation is identical across engines,
#           but only the liberal engine turns it into significance.
# ---------------------------------------------------------------------------
with open(os.path.join(OUT, "panel_c_decoupling.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method", "pipeline", "median_abs_effect", "n_sig_q10", "n_taxa_rows"])
    for method in ("LinDA", "MaAsLin2"):
        rows = D[("DIABIMMUNE", method)]
        for ds, tag in (("base_raw", "base"), ("p2d2pca_imputed", "imputed")):
            s = sub(rows, "diet_group", ds)
            w.writerow([
                method, tag,
                round(median([abs(fnum(r["effect_size"])) for r in s]), 4),
                sum(1 for r in s if (fnum(r["q_value"]) or 1) < 0.10),
                len(s),
            ])

with open(os.path.join(OUT, "panel_d_decoupling_long.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method", "pipeline", "feature_id", "diet_col",
                "abs_effect", "q_value", "neglog10_q", "sig_q10"])
    for method in ("LinDA", "MaAsLin2"):
        rows = D[("DIABIMMUNE", method)]
        for ds, tag in (("base_raw", "base"), ("p2d2pca_imputed", "imputed")):
            for r in sub(rows, "diet_group", ds):
                q = fnum(r["q_value"])
                eff = fnum(r["effect_size"])
                if q is None or eff is None:
                    continue
                w.writerow([method, tag, r["feature_id"], r["diet_col"],
                            abs(eff), q, -math.log10(max(q, 1e-300)), int(q < 0.10)])

# ---------------------------------------------------------------------------
# Panel d : literature gold-standard panel.
# For multi-OTU taxa we take the strongest (min-q) OTU of that genus/species.
# ---------------------------------------------------------------------------
EC_l, EC_m = D[("ECAM", "LinDA")], D[("ECAM", "MaAsLin2")]
DI_l, DI_m = D[("DIABIMMUNE", "LinDA")], D[("DIABIMMUNE", "MaAsLin2")]


def best(rows, gm, ds, pred, diet=None):
    cand = [(fnum(r["q_value"]), fnum(r["effect_size"]))
            for r in rows if r["group_mode"] == gm and r["dataset"] == ds
            and pred(r) and (diet is None or r["diet_col"] == diet)]
    cand = [c for c in cand if c[0] is not None]
    return min(cand, key=lambda c: c[0]) if cand else (None, None)


def four(gm, pred, Rl, Rm, diet=None):
    out = {}
    for meth, R in (("LinDA", Rl), ("MaAsLin2", Rm)):
        for ds, tag in (("base_raw", "base"), ("p2d2pca_imputed", "imp")):
            q, e = best(R, gm, ds, pred, diet)
            out[f"{meth}_{tag}_q"] = None if q is None else round(q, 4)
            out[f"{meth}_{tag}_eff"] = None if e is None else round(e, 2)
    return out


carrot = "dietary_compound_carrot_start_age_month"
panel_d = []


def add(role, taxon, cohort, contrast, expected, gm, pred, Rl, Rm, diet=None):
    row = dict(role=role, taxon=taxon, cohort=cohort, contrast=contrast, expected=expected)
    row.update(four(gm, pred, Rl, Rm, diet))
    panel_d.append(row)


# recover (persistent, surfaced by imputation): Cesarean signature
add("recover", "Bacteroides", "ECAM", "delivery", "Cesarean DOWN",
    "delivery_group", lambda r: r["Genus"] == "Bacteroides", EC_l, EC_m)
add("recover", "Bifidobacterium", "ECAM", "delivery", "Cesarean DOWN",
    "delivery_group", lambda r: r["Genus"] == "Bifidobacterium", EC_l, EC_m)
add("recover", "Veillonella", "ECAM", "delivery", "Cesarean UP",
    "delivery_group", lambda r: r["Genus"] == "Veillonella", EC_l, EC_m)
# confirm (present in observed, over-smoothed by imputation): formula Clostridiales
add("confirm", "Anaerotruncus", "ECAM", "formula", "Formula UP",
    "diet_diet_2_fd", lambda r: r["species_name"] == "Anaerotruncus", EC_l, EC_m)
add("confirm", "Coprobacillus cateniformis", "ECAM", "formula", "Formula UP",
    "diet_diet_2_fd", lambda r: r["species_name"] == "Coprobacillus cateniformis", EC_l, EC_m)
add("confirm", "Akkermansia muciniphila", "ECAM", "formula", "Formula UP (new)",
    "diet_diet_2_fd", lambda r: r["species_name"] == "Akkermansia muciniphila", EC_l, EC_m)
# no-effect (strongest antibiotic taxon should stay non-significant).
# NOTE: the token must not be the literal string "null" -- pandas parses that as NaN.
add("noeffect", "Antibiotics (strongest taxon)", "ECAM", "antibiotics", "no effect",
    "abx_group", lambda r: True, EC_l, EC_m)
add("noeffect", "Antibiotics (strongest taxon)", "DIABIMMUNE", "antibiotics", "no effect",
    "abx_group", lambda r: True, DI_l, DI_m)
# contamination (should be suppressed by imputation)
add("suppress", "Chloroplast / Streptophyta", "DIABIMMUNE", "carrot", "suppress (plant DNA)",
    "diet_group", lambda r: r["feature_id"] == "T0336", DI_l, DI_m, carrot)
add("suppress", "mitochondria", "DIABIMMUNE", "carrot", "suppress (host DNA)",
    "diet_group", lambda r: r["feature_id"] == "T0351", DI_l, DI_m, carrot)

cols = (["role", "taxon", "cohort", "contrast", "expected"] +
        [f"{m}_{t}_{v}" for m in ("LinDA", "MaAsLin2") for t in ("base", "imp") for v in ("q", "eff")])
with open(os.path.join(OUT, "panel_d_goldpanel.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(panel_d)

print("Wrote panel data to", OUT)
for fn in ("panel_b_hitcounts.csv", "panel_c_decoupling.csv",
           "panel_d_decoupling_long.csv", "panel_d_goldpanel.csv"):
    print("  ", fn)
