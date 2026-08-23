from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustness.common import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Validate and cache a configured pseudo-truth tensor.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    registry = load_yaml(ROOT / "configs" / "datasets.yaml")["datasets"]
    spec = registry[args.dataset]
    source = (ROOT.parent / spec["path"]).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Configured dataset does not exist: {source}")
    da = xr.open_dataarray(source).load()
    required = [spec["row_dim"], spec["col_dim"], spec["time_dim"]]
    if set(da.dims) != set(required):
        raise ValueError(f"Expected dimensions {required}, found {list(da.dims)}")
    da = da.transpose(*required)
    target = ROOT / "cache" / "prepared_data" / f"{args.dataset}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(target)
    print(target)


if __name__ == "__main__":
    main()
