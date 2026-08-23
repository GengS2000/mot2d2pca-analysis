# HMP2 disease-activity resolution analysis

This directory tests whether equal-grid HMP2 community trajectories contain an ecological signal that precedes exit from a clinically active IBD state.

Run from the workspace root:

```powershell
python biology_analysis/hmp2_activity_resolution/analyze_activity_resolution.py
```

The script uses the existing HMP2 metadata and the raw/completed tensors under `downstream/dimensionality_reduction/output_full_tuning/best_final_run`. It does not refit or alter the completion model.

Primary endpoint:

- Current active state: HBI >= 5 for Crohn's disease or SCCAI >= 5 for ulcerative colitis.
- Next-bin resolution: HBI <= 4 for Crohn's disease or SCCAI <= 2 for ulcerative colitis.
- UC SCCAI 3-4 is excluded from the strict endpoint.
- The independent unit for the primary test is the participant, not the reconstructed participant-time row.

The result is exploratory and reconstruction-conditional. In particular, the selected tensor used `impute_mode=eps_only`; reconstructed biological zeros and missing participant-time cells cannot be distinguished in that artifact.

Manuscript-ready outputs:

- `RESULTS_SECTION_REVISED.md`: revised Results subsection.
- `TableS14_sensitivity_analyses.md`: formatted supplementary table and notes.
- `TableS14_sensitivity_analyses.csv`: editable numerical source for Table S14.
