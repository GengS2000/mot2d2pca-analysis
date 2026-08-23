"""External competitor adapter for the paired-mask imputation benchmarks.

The public entry point deliberately accepts the benchmark's existing mask.  It
never samples a new mask and it only writes predictions at ``replace_mask``.
CTF uses gemelli's official rclr + CP/ALS implementation.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import xarray as xr


COMPETITOR_METHODS = frozenset({"ctf"})
COMPETITOR_GROUPS = {
    "ctf": "tensor_factorization",
}


class CompetitorUnavailable(RuntimeError):
    """Raised when an explicitly selected external method cannot run."""


def _validate_inputs(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    required = ("participant", "taxon", "bin")
    if not isinstance(da_input, xr.DataArray):
        raise TypeError("da_input must be an xarray.DataArray")
    if set(da_input.dims) != set(required):
        raise ValueError(f"da_input must have dims {required}, got {da_input.dims}")
    for name, mask in (("replace_mask", replace_mask), ("observed_mask", observed_mask)):
        if not isinstance(mask, xr.DataArray):
            raise TypeError(f"{name} must be an xarray.DataArray")
        try:
            mask = mask.transpose(*da_input.dims)
        except ValueError as exc:
            raise ValueError(f"{name} dims do not match da_input") from exc
        if mask.shape != da_input.shape:
            raise ValueError(f"{name} shape does not match da_input")
        if name == "replace_mask":
            replace_mask = mask
        else:
            observed_mask = mask
    da_input = da_input.transpose(*required)
    replace_mask = replace_mask.transpose(*required).astype(bool)
    observed_mask = observed_mask.transpose(*required).astype(bool)
    if bool(np.any(replace_mask.values & observed_mask.values)):
        raise ValueError("replace_mask and observed_mask overlap")
    return da_input, replace_mask, observed_mask


def _log_to_percent(values: np.ndarray, eps: float) -> np.ndarray:
    out = np.power(10.0, np.asarray(values, dtype=float)) - float(eps)
    out[~np.isfinite(out)] = 0.0
    return np.maximum(out, 0.0)


def _masked_compositional_writeback(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    score_tensor: np.ndarray,
    eps: float,
) -> xr.DataArray:
    """Turn arbitrary per-taxon scores into residual-mass predictions."""
    original = da_input.values.astype(float)
    replace = replace_mask.values.astype(bool)
    percent = _log_to_percent(original, eps=eps)
    out = original.copy()
    for p in range(original.shape[0]):
        for b in range(original.shape[2]):
            target = replace[p, :, b]
            if not np.any(target):
                continue
            scores = np.asarray(score_tensor[p, :, b], dtype=float)[target]
            if not np.all(np.isfinite(scores)):
                raise ValueError(f"non-finite competitor scores at participant={p}, bin={b}")
            scores = scores - float(np.max(scores))
            weights = np.exp(np.clip(scores, -700.0, 0.0))
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
                weights = np.ones_like(weights)
            observed_total = float(percent[p, ~target, b].sum())
            residual = max(0.0, 100.0 - observed_total)
            prediction = residual * weights / float(weights.sum())
            out[p, target, b] = np.log10(prediction + float(eps))
    if not np.array_equal(out[~replace], original[~replace]):
        raise AssertionError("competitor changed an off-mask cell")
    if not np.isfinite(out[replace]).all():
        raise ValueError("competitor produced NaN/Inf at a masked cell")
    result = xr.DataArray(
        out,
        coords=da_input.coords,
        dims=da_input.dims,
        attrs=dict(da_input.attrs),
        name="log_abundance_imputed",
    )
    result.attrs["scale"] = "log10_percent_plus_eps"
    return result


def competitor_status(method: str, params: Optional[Dict] = None) -> Tuple[bool, str]:
    """Return dependency availability and a human-readable reason."""
    method = str(method).lower()
    params = {} if params is None else dict(params)
    if method == "ctf":
        missing = []
        for name in ("gemelli.factorization", "gemelli.preprocessing"):
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ModuleNotFoundError):
                spec = None
            if spec is None:
                missing.append(name)
        if missing:
            return False, "missing Python modules: " + ", ".join(missing)
        return True, "gemelli available"
    return False, f"unknown competitor: {method}"


def competitor_available(method: str, params: Optional[Dict] = None) -> bool:
    return competitor_status(method, params=params)[0]


def available_requested_methods(
    methods: Iterable[str], method_params: Optional[Dict[str, Dict]] = None
) -> Tuple[list, Dict[str, str]]:
    """Filter only unavailable competitors; built-in methods pass through."""
    method_params = {} if method_params is None else method_params
    kept, skipped = [], {}
    for method in methods:
        if method not in COMPETITOR_METHODS:
            kept.append(method)
            continue
        ok, reason = competitor_status(method, method_params.get(method, {}))
        if ok:
            kept.append(method)
        else:
            skipped[method] = reason
    return kept, skipped


def _ctf_adapter(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
    eps: float,
    seed: int,
    params: Dict,
) -> xr.DataArray:
    ok, reason = competitor_status("ctf", params)
    if not ok:
        raise CompetitorUnavailable(reason)

    from gemelli import factorization
    from gemelli.preprocessing import tensor_rclr

    abundance = _log_to_percent(da_input.values, eps=eps)
    observed = observed_mask.values.astype(bool)
    sparse = np.where(observed, abundance, 0.0)
    transformed = np.asarray(tensor_rclr(sparse), dtype=float)
    transformed[~observed] = 0.0
    mask = observed.astype(float)

    requested_rank = int(params.get("rank", 3))
    rank = max(1, min(requested_rank, min(transformed.shape)))
    n_init = max(1, int(params.get("n_initializations", 10)))
    max_als = max(1, int(params.get("max_iterations_als", 50)))
    max_rtpm = max(1, int(params.get("max_iterations_rptm", 50)))

    # gemelli's public tenals currently fixes RTPM's random_state at 42.  The
    # temporary wrapper preserves the official implementation while threading
    # the benchmark seed into that one exposed initialization point.
    original_rtpm = factorization.robust_tensor_power_method

    def seeded_rtpm(tensor, n_components, n_initializations=50,
                    max_rtpm_iterations=50, tol_rtpm=1e-10, random_state=42):
        return original_rtpm(
            tensor,
            n_components,
            n_initializations=n_initializations,
            max_rtpm_iterations=max_rtpm_iterations,
            tol_rtpm=tol_rtpm,
            random_state=int(seed),
        )

    factorization.robust_tensor_power_method = seeded_rtpm
    try:
        loadings, eigvals, _ = factorization.tenals(
            transformed,
            mask,
            n_components=rank,
            n_initializations=n_init,
            max_als_iterations=max_als,
            max_rtpm_iterations=max_rtpm,
            tol_als=float(params.get("tol_als", 1e-8)),
            tol_rtpm=float(params.get("tol_rtpm", 1e-8)),
            fillna=float(params.get("fillna", 1.0)),
        )
    finally:
        factorization.robust_tensor_power_method = original_rtpm

    # tenals returns its component weights as a diagonal matrix, while
    # construct_tensor expects the corresponding one-dimensional vector.
    component_weights = np.diag(eigvals) if np.ndim(eigvals) == 2 else np.asarray(eigvals)
    reconstructed_rclr = np.asarray(
        factorization.construct_tensor(component_weights, loadings), dtype=float
    )
    return _masked_compositional_writeback(
        da_input, replace_mask, reconstructed_rclr, eps=eps
    )


def impute_competitor(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
    zero_value: float,
    method: str,
    seed: int,
    params: Optional[Dict] = None,
    n_threads: int = 1,
) -> xr.DataArray:
    """Impute with an external method under the paired-mask invariants."""
    da_input, replace_mask, observed_mask = _validate_inputs(
        da_input, replace_mask, observed_mask
    )
    method = str(method).lower()
    params = {} if params is None else dict(params)
    eps = float(params.get("eps", 10.0 ** float(zero_value)))
    if eps <= 0:
        raise ValueError("eps must be positive")
    if method == "ctf":
        result = _ctf_adapter(
            da_input, replace_mask, observed_mask, eps=eps, seed=int(seed), params=params
        )
    else:
        raise ValueError(f"Unsupported competitor method: {method}")

    result = result.transpose(*da_input.dims)
    off_mask = ~replace_mask.values.astype(bool)
    if not np.array_equal(result.values[off_mask], da_input.values[off_mask]):
        raise AssertionError(f"{method} modified off-mask cells")
    if not np.isfinite(result.values[~off_mask]).all():
        raise ValueError(f"{method} returned NaN/Inf on masked cells")
    result.attrs["competitor_method"] = method
    result.attrs["competitor_seed"] = int(seed)
    return result
