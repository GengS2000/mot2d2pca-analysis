#!/bin/bash
#SBATCH --job-name=sim123_check
#SBATCH --output=server_runs/logs/%x_%j.out
#SBATCH --error=server_runs/logs/%x_%j.err
#SBATCH --partition=hebhcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p server_runs/logs

: "${SIM1_TRUTH_PATH:?Export SIM1_TRUTH_PATH}"
: "${SIM2_TRUTH_PATH:?Export SIM2_TRUTH_PATH}"
: "${SIM3_TRUTH_PATH:?Export SIM3_TRUTH_PATH}"
: "${RUN_ROOT:?Export RUN_ROOT}"
test -f "$SIM1_TRUTH_PATH"
test -f "$SIM2_TRUTH_PATH"
test -f "$SIM3_TRUTH_PATH"

source ~/miniconda3/bin/activate
conda activate q2-202402
set -u
python - <<'PY'
from pathlib import Path
import importlib.util
import os
import xarray as xr

missing = []
for name in ("gemelli.factorization", "gemelli.preprocessing"):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is None:
        missing.append(name)
if missing:
    raise RuntimeError("Gemelli CTF comparison dependencies are missing: " + ", ".join(missing))
from gemelli import factorization
from gemelli.preprocessing import tensor_rclr  # noqa: F401
for attr in ("tenals", "construct_tensor", "robust_tensor_power_method"):
    if not hasattr(factorization, attr):
        raise RuntimeError(f"q2-202402 Gemelli lacks required factorization.{attr} API")

truth_info = {}
for env_name in ("SIM1_TRUTH_PATH", "SIM2_TRUTH_PATH", "SIM3_TRUTH_PATH"):
    path = Path(os.environ[env_name])
    obj = xr.open_dataset(path)
    try:
        if not obj.data_vars:
            raise RuntimeError(f"No data variable in {path}")
        da = obj[next(iter(obj.data_vars))]
        if set(da.dims) != {"participant", "taxon", "bin"}:
            raise RuntimeError(f"Unexpected dimensions in {path}: {da.dims}")
        truth_info[env_name] = {
            dim: tuple(map(str, da.coords[dim].values))
            for dim in ("participant", "taxon", "bin")
        }
        print(f"[truth check] {env_name}: shape={da.shape} path={path}")
    finally:
        obj.close()
if truth_info["SIM2_TRUTH_PATH"] != truth_info["SIM3_TRUTH_PATH"]:
    raise RuntimeError("Sim2 and Sim3 frozen truths do not have identical coordinates")
print("[preflight] q2-202402, Gemelli, and all frozen truths are ready")
PY

mkdir -p "$RUN_ROOT/sim1" "$RUN_ROOT/sim2" "$RUN_ROOT/sim3"
