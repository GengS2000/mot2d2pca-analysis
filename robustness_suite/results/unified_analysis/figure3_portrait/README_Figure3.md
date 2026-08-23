# Figure 3 reproducibility package

This package generates the portrait, A4-proportioned Figure 3 for the
MOT-(2D)^2 PCA parameter-sensitivity and ablation analysis.

## Files

- `Figure3_parameter_sensitivity_ablation.pdf`: vector publication figure.
- `Figure3_parameter_sensitivity_ablation.png`: 600-dpi PNG.
- `Figure3_parameter_sensitivity_ablation.tiff`: 600-dpi losslessly compressed TIFF.
- `plot_figure3_parameter_sensitivity_portrait.py`: plotting program.
- `prepare_figure3_data.py`: source-data preparation program.
- `data/`: audited Stage B/C estimates, the equivalence audit, paired one-visit
  time-point values, a source manifest, a data dictionary, and the exact values
  plotted in panels C-G.

## Reproduce the figure

From this directory, using a Python environment with NumPy, pandas,
Matplotlib, Pillow, and tifffile:

```powershell
python .\plot_figure3_parameter_sensitivity_portrait.py `
  --data-dir .\data `
  --output-stem .\Figure3_parameter_sensitivity_ablation `
  --dpi 600
```

The pooled time-point confidence intervals use 10,000 percentile-bootstrap
resamples with seed `2026071501`.

## Regenerate the delivered data from the robustness suite

```powershell
python .\prepare_figure3_data.py `
  --suite-root D:\penalized_2d2\penalized_2d2_pca\robustness_suite `
  --output-dir .\data
```

Panels A and B explain the temporal roles of sigma and lambda without repeating
the tensor and low-rank workflow already shown in the preceding main figure.
Panel F shows each implementation-equivalent ablation pair once. Panel G shows
time-point masking as one missing visit because its three nominal fraction
labels masked the same number of entries; the 90 paired repeats are pooled.
