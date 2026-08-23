from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from .penalized_2d2_pca import _bin_to_time_value, build_observed_mask, penalized_2d2_pca
from .preprocessing import EXPECTED_DIMS, validate_tensor


def _global_fallback(arr: np.ndarray, obs: np.ndarray, zero_value: float, reducer: str = "mean") -> float:
    vals = arr[obs & np.isfinite(arr)]
    if vals.size == 0:
        return float(zero_value)
    return float(np.nanmedian(vals) if reducer == "median" else np.nanmean(vals))


def _fill_last(times, obs_idx, obs_vals, target_time, fallback):
    if obs_idx.size == 0:
        return float(fallback)
    obs_times = times[obs_idx]
    left = np.flatnonzero(obs_times <= target_time)
    if left.size:
        return float(obs_vals[left[-1]])
    return float(obs_vals[0])


def _fill_weighted(times, obs_idx, obs_vals, target_time, fallback, sigma=4.0):
    if obs_idx.size == 0:
        return float(fallback)
    dist = np.abs(times[obs_idx] - float(target_time))
    w = np.exp(-(dist**2) / (2.0 * max(float(sigma), 1e-8) ** 2))
    return float(np.average(obs_vals, weights=w)) if np.any(w > 0) else float(np.mean(obs_vals))


def _fill_knn(times, obs_idx, obs_vals, target_time, fallback, k=3, sigma=4.0):
    if obs_idx.size == 0:
        return float(fallback)
    dist = np.abs(times[obs_idx] - float(target_time))
    order = np.argsort(dist)[: max(int(k), 1)]
    return _fill_weighted(times[obs_idx][order], np.arange(order.size), obs_vals[order], target_time, fallback, sigma=sigma)


def _fill_spline(times, obs_idx, obs_vals, target_time, fallback, min_points=4):
    if obs_idx.size == 0:
        return float(fallback)
    obs_times = times[obs_idx]
    if obs_idx.size >= max(int(min_points), 4):
        try:
            from scipy.interpolate import CubicSpline

            pred = float(CubicSpline(obs_times, obs_vals, extrapolate=False)(target_time))
            if np.isfinite(pred):
                return pred
        except Exception:
            pass
    if obs_idx.size >= 2:
        return float(np.interp(target_time, obs_times, obs_vals))
    return float(obs_vals[0])


def impute_penalized_2d2_pca(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
    d: int,
    q: int,
    lam: float,
    sigma: float,
    T0: int,
    T: int,
    seed: int,
    clip_to_observed_range: bool = True,
    verbose: bool = False,
) -> Tuple[xr.DataArray, xr.DataArray, Dict]:
    """Run penalized 2D2-PCA and replace only requested tensor cells."""
    validate_tensor(da_input, "da_input")
    validate_tensor(replace_mask, "replace_mask")
    validate_tensor(observed_mask, "observed_mask")
    x, z, c, meta = penalized_2d2_pca(
        da=da_input,
        time=da_input.coords["bin"].values,
        d=d,
        q=q,
        lam=lam,
        sigma=sigma,
        T0=T0,
        T=T,
        seed=seed,
        observed_mask=observed_mask,
        verbose=verbose,
    )
    recon = np.empty_like(da_input.values.astype(float))
    for ti, _b in enumerate(meta["bin"]):
        recon[:, :, ti] = meta["Abar"] + z @ c[ti] @ x.T
    if clip_to_observed_range:
        obs_vals = da_input.values[observed_mask.values.astype(bool)]
        obs_vals = obs_vals[np.isfinite(obs_vals)]
        if obs_vals.size:
            recon = np.clip(recon, float(np.min(obs_vals)), float(np.max(obs_vals)))
    out = da_input.values.astype(float).copy()
    mk = replace_mask.values.astype(bool)
    out[mk] = recon[mk]
    da_recon = xr.DataArray(recon, dims=EXPECTED_DIMS, coords=da_input.coords, name="log_profile_reconstructed")
    da_imputed = xr.DataArray(out, dims=EXPECTED_DIMS, coords=da_input.coords, name="log_profile_imputed")
    model_meta = {"method": "penalized_2d2_pca", "d": int(d), "q": int(q), "lam": float(lam), "sigma": float(sigma), "T0": int(T0), "T": int(T), "seed": int(seed), "fit_meta": {k: v for k, v in meta.items() if k not in {"Abar", "W"}}}
    return da_imputed, da_recon, model_meta


def impute_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: Optional[xr.DataArray],
    method: str,
    method_params: Optional[Dict] = None,
    pca_params: Optional[Dict] = None,
    zero_value: float = -3.0,
    seed: int = 0,
) -> xr.DataArray:
    """Uniform dispatcher for PCA and baseline imputation methods."""
    validate_tensor(da_input, "da_input")
    validate_tensor(replace_mask, "replace_mask")
    method = str(method)
    method_params = {} if method_params is None else dict(method_params)
    if observed_mask is None:
        observed_mask = build_observed_mask(da_input, zero_value=zero_value, mode="positive_only_except_replace", replace_mask=replace_mask)
    else:
        validate_tensor(observed_mask, "observed_mask")

    if method == "penalized_2d2_pca":
        if pca_params is None:
            raise ValueError("pca_params is required for penalized_2d2_pca")
        da_imp, _da_recon, _meta = impute_penalized_2d2_pca(
            da_input=da_input,
            replace_mask=replace_mask,
            observed_mask=observed_mask,
            seed=seed,
            **pca_params,
        )
        return da_imp
    if method in {"no_imputation"}:
        return da_input.copy(deep=True)
    if method in {"equal_zero", "pseudo_count", "equal"}:
        out = da_input.values.astype(float).copy()
        out[replace_mask.values.astype(bool)] = float(method_params.get("fill_value", zero_value))
        return xr.DataArray(out, dims=EXPECTED_DIMS, coords=da_input.coords, name="log_profile_imputed")
    if method not in {"mean", "median", "last", "weighted_time_average", "temporal_knn", "spline"}:
        raise ValueError(f"Unsupported imputation method: {method}")

    arr = da_input.values.astype(float)
    mask = replace_mask.values.astype(bool)
    obs = observed_mask.values.astype(bool)
    times = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
    out = arr.copy()
    global_mean = _global_fallback(arr, obs, zero_value, reducer="mean")
    global_median = _global_fallback(arr, obs, zero_value, reducer="median")
    n_p, n_j, _n_t = arr.shape
    for p in range(n_p):
        for j in range(n_j):
            mk = mask[p, j, :]
            if not np.any(mk):
                continue
            obs_idx = np.flatnonzero(obs[p, j, :] & ~mk & np.isfinite(arr[p, j, :]))
            obs_vals = arr[p, j, obs_idx]
            fallback = global_median if method == "median" else global_mean
            if obs_vals.size:
                fallback = float(np.nanmedian(obs_vals) if method == "median" else np.nanmean(obs_vals))
            for t_idx in np.flatnonzero(mk):
                target_time = times[t_idx]
                if method == "mean":
                    out[p, j, t_idx] = fallback
                elif method == "median":
                    out[p, j, t_idx] = fallback
                elif method == "last":
                    out[p, j, t_idx] = _fill_last(times, obs_idx, obs_vals, target_time, fallback)
                elif method == "weighted_time_average":
                    out[p, j, t_idx] = _fill_weighted(times, obs_idx, obs_vals, target_time, fallback, sigma=method_params.get("sigma", 4.0))
                elif method == "temporal_knn":
                    out[p, j, t_idx] = _fill_knn(times, obs_idx, obs_vals, target_time, fallback, k=method_params.get("k", 3), sigma=method_params.get("kernel_sigma", 4.0))
                elif method == "spline":
                    out[p, j, t_idx] = _fill_spline(times, obs_idx, obs_vals, target_time, fallback, min_points=method_params.get("min_points", 4))
    return xr.DataArray(out, dims=EXPECTED_DIMS, coords=da_input.coords, name="log_profile_imputed")

