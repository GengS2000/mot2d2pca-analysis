# Supplement index — infant-gut MOT-(2D)² PCA

Built by `supp_prepare_data.py` (tables + backbone) then `supp_make_figures.py`
(figures). All figures are SVG + PDF + PNG. Backbone data in `supp/data/`.

## How it maps to the two-act narrative

**Act I — method characterization + boundaries** (backs main b/c/d): recovery,
engine-dependent significance, effect changes, and row expansion.
**Act II — biological structure stress test** (backs main e, SF9/SF10):
observed and MOT-completed grids are interpreted jointly to recover
cohort-coherent signals, reject technical artifacts, and distinguish stable
structure from time-localized dynamics. The four exposures are adjudicated as a
graded result: delivery is coherent but predominantly first-year, antibiotics
show no stable mean shift, feeding reflects maturation tempo, and solid-food
timing does not define a reproducible taxon-level axis.

### Supplementary Tables (`supp/tables/`)
| file | contents | backs |
|---|---|---|
| ST1_all_results_long.csv | all 94,468 rows: cohort × method × pipeline × taxon × contrast | everything (primary record) |
| ST2_hitcounts_all_contrasts.csv | base/imputed significant counts per contrast, q<0.05 & q<0.10 | b, SF2 |
| TableS13_gold_standard_full.csv | every OTU of the control genera across the 4 combos | c, SF5 |
| TableS14_effect_inflation.csv | median/IQR |effect| base vs imputed + fold + n_sig | d, SF1 |
| TableS15_imputed_only_prevalence.csv | imputed-only hits with observed-sample prevalence, full-grid prevalence, and effect fold | d, SF3 |
| TableS16_run_provenance.csv | per-task formula, n_samples, n_participants, engine settings | Methods, SF6 |
| TableS17_temporal_fidelity.csv | ECAM C/V *Bacteroides* persistence ratio (obs vs imp, y1/y2); DIABIMMUNE succession error vs Yassour (base vs imp) | main e, SF9, SF10 |
| TableS18_biological_evidence_ledger.csv | claim-to-evidence bridge for the four exposures: observed evidence, MOT-completed evidence, diagnostics, incremental value, and adjudicated inference | main e; biological story closure |

### Supplementary Figures (`supp/figures/`)
| file | shows | supports (main-text claim) |
|---|---|---|
| SF1_volcano_full | base vs imputed |effect| & −log10q, both cohorts/engines | "inflation is global & engine-independent; only significance differs" (Act I) |
| SF2_hitcounts_exhaustive | hit counts, all 22 foods + all ECAM contrasts | "panel b is representative, not cherry-picked" (Act I) |
| SF3_prevalence_inflation | observed-sample prevalence vs effect-inflation fold; imputed-only hits highlighted | low-prevalence taxa in small groups have median 6.2× inflation (Act I mechanism) |
| SF4_concordance_upset | UpSet of significant sets across 4 combinations (DIABIMMUNE diet) | robustness tiers: cross-engine + cross-representation = higher support; completion-only overlap remains reconstruction-sensitive (Act I) |
| SF5_cesarean_full_taxonomy | all Bacteroides/Bifidobacterium/Veillonella OTUs × 4 combos | "Cesarean recovery is consistent across OTUs, not one feature" (Act II delivery) |
| SF6_pseudoreplication | rows fed to model: DIABIMMUNE 1018→1440 and ECAM 689→5124, versus 40/42 participants | full-grid row expansion and reconstruction-uncertainty caveat (Act I / Methods) |
| SF7_feeding_mirror | eb (down) vs fd (up) Clostridiales effects | "breastfeeding retains an immature community; feeding = tempo" (Act II feeding) |
| SF8_antibiotics_null | minimum q across LinDA, MaAsLin2, and glmmPQL plus the T0487 direction check | q<0.05 remains null in all 12 analyses; the completed-data MaAsLin2 q=0.089 result is trend-level and changes direction under glmmPQL |
| SF9_succession | DIABIMMUNE 8-taxon developmental succession, base-observed vs imputed, vs Yassour 2016 benchmarks | "completion compresses early succession dynamics; observed grid is gold-standard-faithful" (main e) |
| SF10_delivery | ECAM *Bacteroides/Bifidobacterium/Veillonella* trajectories by delivery mode, observed vs imputed | "observed cesarean deficit closes after 12 mo (Bokulich); completion freezes it as persistent" (main e) |

## Reproduce
```bash
python supp_prepare_data.py                          # -> ST1-16 + supp/data/*.csv
python supp_make_figures.py                          # -> SF1-8 (600 dpi)
python temporal_check/run_temporal_trajectories.py   # -> SF9, SF10, TableS17 (600 dpi)
```
All figures are 600 dpi (`savefig.dpi` in `figstyle.py`).

## Notes / caveats to carry into captions
- Thresholds: ECAM q<0.05, DIABIMMUNE q<0.10 (native per-cohort), stated on each panel.
- SF6 is an *illustration* of modeled-row expansion, not a corrected refit.
  Participant-level bootstrap with re-imputation is recommended because a random
  intercept alone does not propagate tensor-reconstruction uncertainty.
- The DIABIMMUNE solid-food "hits" (SF2/SF3/SF4) are dominated by tiny subgroups
  (e.g. carrot late = 3 infants) and plant-DNA contamination — reported as the
  counterexample, not as biology.
