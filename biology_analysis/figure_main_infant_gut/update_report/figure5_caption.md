# Figure 5 — caption & legend (route B)

**Fig. 5. On real infant-gut cohorts, MOT-(2D)² PCA recovers near-threshold
exposure signals and suppresses technical artifacts, but its full-grid
reconstruction imposes temporal persistence — so the observed grid remains the
reference for early-life exposure biology.**

**(A)** Sparse subject × time × taxon tensors from the DIABIMMUNE and ECAM cohorts
are completed by MOT-(2D)² PCA and analyzed in a fully crossed 2 × 2 design:
preprocessing pipeline (observed `base_raw` retaining zeros/missingness vs
`p2d2pca_imputed`) crossed with inference method spanning the bias–variance spectrum
(LinDA, MaAsLin2).

**(B)** Change in the number of significant taxa after completion (`base_raw` →
`imputed`) for solid-food timing, delivery mode, feeding mode, and antibiotic
exposure, under each inference method. Open and filled points indicate observed and
completed data, respectively; arrows connect the two.

**(C)** Literature-anchored scorecard across both engines and both pipelines:
known biological signals recovered (Cesarean-associated depletion of *Bacteroides*
and *Bifidobacterium*), observed-data benchmarks (formula-associated Clostridiales),
a negative control (antibiotic exposure), and a technical artifact (carrot-associated
plant/host-derived chloroplast and mitochondrial 16S sequences). Cell shade encodes
−log₁₀(q); triangles mark effect direction (▲ enriched, ▼ depleted) at q < 0.10.

**(D)** Effect-size amplification is decoupled from significance. Completion increased
the median absolute effect size ≈3.4-fold under both methods (left, box plots), yet
the number of significant taxa rose under the liberal LinDA (16 → 45) and fell under
the conservative MaAsLin2 (34 → 13) at q < 0.10 for DIABIMMUNE solid-food timing
(right). Because both tests operate on the same completed tensor, the uniform
inflation is a property of the reconstructed data and its conversion into significance
a property of the test. Grey = observed, dark = completed.

**(E)** Temporal-fidelity hinge. *Left and middle:* ECAM *Bacteroides* relative-
abundance trajectory by delivery mode (green = vaginal, vermillion = cesarean) over
0–24 months (dotted line, 12 months). In the observed data (points) the cesarean
deficit is concentrated in the first year and closes in the second
(cesarean/vaginal ratio 0.36 → 0.80, reproducing Bokulich et al.), whereas after
completion (lines) it is frozen as a near-constant, full-axis offset (0.49 → 0.51):
the low-rank factorization renders a participant-level deviation persistent by
construction. *Right:* graded verdict read against the observed grid and the
literature — delivery mode a real but **transient** first-year imprint, antibiotics a
robust genus-level **null**, feeding a maturation-**tempo** modifier rather than a
persistent axis, and individual solid-food timing a **spurious** (contamination /
small-group) non-axis. Verdict color: green = signal confirmed, grey = null,
blue = tempo, vermillion = spurious/artifact.

Significance thresholds were q < 0.10 for DIABIMMUNE and q < 0.05 for ECAM.
Exhaustive per-contrast enumerations, effect-size inflation, prevalence, cross-method
concordance, and the full succession/delivery trajectories are given in Supplementary
Figures S9–S18 and Supplementary Tables S11–S17.
