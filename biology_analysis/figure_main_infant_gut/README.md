# Main figure — infant-gut MOT-(2D)² PCA results section

Reproducible build of the composite main figure (panels **a–e**) plus its
machine-readable panel data. Landing point: **joint observed-grid and
MOT-(2D)² PCA-completed analysis is a biological structure stress test: it
recovers cohort-coherent signals, rejects technical artifacts, and distinguishes
stable structure from time-localized dynamics.** The
narrative follows the results logic:
**a** design → **b** real-data 2×2 discovery behavior → **c** literature
gold-standard benchmark → **d** DIABIMMUNE counterexample mechanism
(identical effect inflation, divergent significance) → **e** temporal-fidelity
hinge: completion is a smoother that can impose persistence (ECAM
*Bacteroides* observed C/V 0.36→0.80 attenuates, imputed 0.49→0.51 frozen) and
compresses succession, plus the four-exposure graded verdict (delivery
**transient** / antibiotics **null** / feeding **tempo** / solid-food timing
**spurious**).

## Folder contents

```
figure_main_infant_gut/
├── figstyle.py              # shared palette, rcParams, small drawing helpers
├── prepare_panel_data.py    # reads the 4 result CSVs -> panel_data/*.csv (also = Supp. Tables)
├── make_figure.py           # builds the composite + (optional) per-panel SVGs
├── panel_data/              # tidy per-panel inputs (generated)
│   ├── panel_b_hitcounts.csv
│   ├── panel_c_decoupling.csv
│   ├── panel_d_decoupling_long.csv
│   └── panel_d_goldpanel.csv
├── svg/                     # per-panel SVGs (generated with --individual)
└── output/                  # figure_main.{svg,pdf,png} (generated)
```

## Dependencies

`python>=3.9`, `numpy`, `pandas`, `matplotlib` (>=3.5), `Pillow`. No `svgutils` needed —
the composite is assembled inside matplotlib. Fonts: Arial if installed, else
Helvetica/DejaVu Sans (all glyphs used are Arial-safe).

In the current Windows workspace, Pillow and matplotlib runtime dependencies are
installed locally under `D:\data\.python_deps`. Use this before rendering:

```powershell
$env:PYTHONPATH = "D:\data\.python_deps"
```

## Run order

```bash
python prepare_panel_data.py                    # panel_data/{b,c,d}*.csv from result files
python temporal_check/run_temporal_trajectories.py  # panel_data/panel_e_*.csv, SF9/SF10, TableS17
python make_figure.py                           # -> output/figure_main.{svg,pdf,png}
python make_figure.py --individual              # additionally -> svg/panel_[a-e].svg
```

`temporal_check/run_temporal_trajectories.py` reads the exported tensors
(`*/tensor_base_raw.nc`, `*/tensor_p2d2pca_imputed.nc`, masks) and produces the
Panel-e trajectories, the two temporal supplementary figures, and the temporal-
fidelity table. All figures render at **600 dpi** (`savefig.dpi` in `figstyle.py`).

`prepare_panel_data.py` reads the four differential-abundance result tables
relative to the parent folder (`DATA_ROOT = ..`):

| panel | source result file(s) |
|---|---|
| b | `*/linda_longitudinal_results.csv`, `*/maaslin2_longitudinal_results.csv` (both cohorts) |
| c | ECAM + DIABIMMUNE, LinDA + MaAsLin2 (delivery / diet / abx / carrot) |
| d | DIABIMMUNE LinDA + MaAsLin2 (diet_group), summarized and taxon-level |
| e | `panel_data/panel_e_ecam_bacteroides.csv`, `panel_e_persistence.csv`, `panel_e_verdict.csv` (from the tensors via `temporal_check/`) |
| a | schematic (no data file; annotations hard-coded from the analysis) |

## Editing / polishing

- **Vector text is preserved** (`svg.fonttype=none`, `pdf.fonttype=42`), so every
  label is editable in Illustrator/Inkscape.
- The two **schematic panels (a, e)** are laid out to correct proportions and
  fully labelled, but are intended as an *editable draft*: open `svg/panel_a.svg`
  / `svg/panel_e.svg` to refine glyphs, curves and spacing by hand. The three
  **data panels (b, c, d)** are final and should be regenerated from data, not
  edited by hand.
- To restyle globally, edit `figstyle.py` (palette `C`, rcParams). Method colours
  (LinDA orange / MaAsLin2 blue) live only in b,c; direction/verdict colours live
  only in d,e — keep that separation.
- Optional illustrator refinements: convert panel-e verdict marks to custom
  icons or hand-tune label spacing after journal-specific sizing is known.

## Colour / glyph legend (for the caption)

- base_raw = grey (sparse) · imputed = slate (dense)
- LinDA = orange (liberal) · MaAsLin2 = blue (conservative)
- recover/confirm = green tag · no-effect = grey tag · suppress/contamination = vermillion tag
- ▲ enriched · ▼ depleted · ○ null · × removed · cell shade = −log₁₀q

---

# Supplementary figures & tables — preparation plan

The main figure is deliberately curated (one representative contrast per point).
The supplement carries the *full, auditable* evidence. Everything below is
derivable from the same four result files + the `panel_data/` and
`artifacts_method_comparison/` tables already produced.

## Supplementary Tables (machine-readable, one CSV each)

- **ST1 — Full differential-abundance results, all four pipelines.** The four
  `*_longitudinal_results.csv` merged into one long table with a `method`
  (LinDA/MaAsLin2) and `cohort` column; one row per taxon × contrast × pipeline.
  This is the primary data record backing every panel.
- **ST2 — 2×2 hit-count summary, all contrasts, both thresholds.** Extend
  `artifacts_method_comparison/hit_count_2x2_summary.csv` to every contrast (all
  22 DIABIMMUNE foods; all ECAM diet codes), at q<0.05 **and** q<0.10, with
  base/imputed counts and median |effect| per cell. Backs panels b and d.
- **ST3 — Literature gold-standard panel (full).** Every control taxon (all OTUs
  per genus, not just the min-q one), with q and effect for the four combos and
  the expected direction. Backs panel c; lets a reviewer check each cell.
- **ST4 — Effect-size inflation table.** Per contrast × method: median/IQR |effect|
  base vs imputed and the fold-change, plus the effect–significance decoupling
  numbers (n_sig base→imputed). Backs panel d.
- **ST5 — Prevalence / zero-inflation of imputed-only hits.** For every
  imputed-only significant taxon: prevalence and zero-fraction in base_raw, group
  sizes (e.g. carrot early=37/late=3), and base→imputed effect change. This is
  the quantitative core of the "sparse-group inflation" mechanism.
- **ST6 — Model / run provenance.** One row per task: formula, engine settings
  (LinDA: proportion + pseudo-count; MaAsLin2: NONE/LOG/LM/BH), n_samples,
  n_participants, group counts, per-task FDR scope. Establishes that the two
  engines answer the same question on the same data.

## Supplementary Figures

- **SF1 — Full base-vs-imputed effect-size volcano/scatter.** The un-summarised
  version of panel d: every taxon plotted as base |effect| vs imputed |effect|
  (left) and −log₁₀q base vs imputed (right), one row per cohort, coloured by
  engine. Shows the ~3.4×/3.7× inflation is global and engine-independent, and
  that only LinDA drags points across the significance line.
- **SF2 — Per-contrast hit counts, exhaustive.** The full version of panel b:
  small-multiples of base→imputed significant counts for all 22 DIABIMMUNE foods
  and all ECAM diet/delivery/abx contrasts, LinDA vs MaAsLin2. Documents that the
  curated panel-b lines are representative, not cherry-picked.
- **SF3 — Prevalence drives imputation-induced inflation.** Scatter of base_raw
  prevalence (x) vs base→imputed |effect| fold-change (y) for DIABIMMUNE diet
  taxa, highlighting imputed-only hits. The mechanistic figure: low-
  prevalence taxa in small/unbalanced groups get the largest reconstruction-
  driven inflation. Pairs with ST5.
- **SF4 — Cross-method / cross-pipeline concordance.** UpSet or Venn of the
  significant-taxon sets across the four combos, per key contrast. Makes the
  robustness tiers explicit: cross-method+cross-pipeline hits (credible, e.g.
  delivery) vs imputed-only-but-cross-method hits (shared inflation, not
  credible).
- **SF5 — Cesarean signature, full taxonomy.** All Bacteroides/Bifidobacterium/
  Veillonella OTUs (species level) across the four combos with effect direction,
  showing the recovery is consistent across OTUs, not one lucky feature. Supports
  panel e "delivery" lane.
- **SF6 — Full-grid row-expansion diagnostic.** Imputed n grows 1018→1440 in
  DIABIMMUNE and 689→5124 in ECAM, while independent participants remain 40 and
  42. The figure quantifies the risk; participant-level bootstrap with
  re-imputation is still required for formal confirmation.
- **SF7 (optional) — Method schematic detail.** A larger, annotated version of the
  MOT-(2D)² PCA tensor-completion step (the "persistence filter") for readers who
  want the imputation mechanics.

## Suggested main-vs-supp split

Keep in the **main figure** only: the design (a), the engine-flip headline (b),
the gold-standard scorecard (c), the decoupling mechanism (d), the biological
synthesis (e). Push exhaustiveness (all foods, all OTUs, prevalence mechanics,
pseudo-replication robustness) to SF1–SF6, and the raw records to ST1–ST6.
