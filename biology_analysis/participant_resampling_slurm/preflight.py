#!/usr/bin/env python3
"""Server-side preflight for the participant bootstrap pipeline."""

from pathlib import Path
import importlib
import sys

root = Path(__file__).resolve().parents[1]
for name in ("numpy", "pandas", "xarray"):
    importlib.import_module(name)
sys.path.insert(0, str(root / "ECAM_data"))
sim3 = importlib.import_module("sim3_compare")
if not callable(getattr(sim3, "impute_by_mask", None)):
    raise RuntimeError("sim3_compare.impute_by_mask is unavailable")

required = [
    root / "ECAM_data/ECAM_otu_tensor_export/tensor_base_raw.nc",
    root / "ECAM_data/ECAM_otu_tensor_export/replace_mask.nc",
    root / "DIABIMMUNE_data/diabimmune_p2d2pca_netcdf_export/tensors/tensor_base_raw.nc",
    root / "DIABIMMUNE_data/diabimmune_p2d2pca_netcdf_export/tensors/replace_mask.nc",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
print("PYTHON_PREFLIGHT_OK")
