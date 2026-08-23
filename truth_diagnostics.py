"""Write comparable temporal diagnostics for a frozen Sim1 or Sim2 truth tensor."""

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from sim1_compare.sim1_methodcompare import summarize_truth_dynamics


def _load_truth(path: Path) -> xr.DataArray:
    try:
        return xr.open_dataarray(path).load()
    except Exception:
        dataset = xr.open_dataset(path)
        if len(dataset.data_vars) != 1:
            raise ValueError(f"Expected one data variable in {path}, found {list(dataset.data_vars)}")
        return dataset[next(iter(dataset.data_vars))].load()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulation", required=True, choices=["sim1", "sim2"])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--ctf-rank", type=int, default=None)
    args = parser.parse_args()

    da = _load_truth(Path(args.truth))
    row = {
        "simulation": args.simulation,
        "candidate": args.candidate,
        "d": args.d,
        "q": args.q,
        "lam": args.lam,
        "sigma": args.sigma,
        "ctf_rank": args.ctf_rank,
        **summarize_truth_dynamics(da, eps=args.eps),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
