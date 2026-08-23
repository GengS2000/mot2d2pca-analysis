import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Lasso, Ridge


METRIC_COLS = [
    "mse_before",
    "mse_after",
    "mse_improve",
    "pearson_before",
    "pearson_after",
    "pearson_improve",
    "wd_before",
    "wd_after",
    "wd_improve",
]

DEFAULT_METHODS = [
    "penalized_2d2_pca",
    "mean",
    "median",
    "last",
    "weighted_time_average",
    "temporal_knn",
    "spline",
    "dbn_dense",
    "dbn_sparse",
    "glv",
    "equal",
]

METHOD_GROUP_MAP = {
    "penalized_2d2_pca": "matrix_factorization",
    "mean": "naive",
    "median": "naive",
    "equal": "naive",
    "last": "local_time",
    "weighted_time_average": "local_time",
    "temporal_knn": "local_time",
    "spline": "local_time",
    "dbn_dense": "dynamic",
    "dbn_sparse": "dynamic",
    "glv": "dynamic",
}


def _validate_da(da: xr.DataArray, name: str = "data_array") -> xr.DataArray:
    """Validate a tensor-like xarray DataArray input."""
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    missing = [dim for dim in ("participant", "taxon", "bin") if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dims: {missing}")
    return da


def _method_group(method: str) -> str:
    """Return a stable method group label."""
    return METHOD_GROUP_MAP.get(method, "other")


def _format_frac(frac: float) -> str:
    """Format a masking fraction for filenames."""
    return f"{float(frac):g}"


def _summary_real(values: pd.Series) -> pd.Series:
    """Summarize a real-valued metric across repeats."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    mean = float(np.mean(arr)) if n > 0 else np.nan
    sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    sem = float(sd / np.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan
    ci95_low = float(mean - 1.96 * sem) if np.isfinite(mean) and np.isfinite(sem) else np.nan
    ci95_high = float(mean + 1.96 * sem) if np.isfinite(mean) and np.isfinite(sem) else np.nan
    return pd.Series(
        {
            "n": n,
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
        }
    )


def week_to_bin(week: int) -> Optional[str]:
    if week == 0:
        return "0"
    if 1 <= week <= 36:
        start = ((week - 1) // 4) * 4 + 1
        end = start + 3
        return f"{start}-{end}"
    return None


def _bin_to_time_value(x) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return float(s)


def _bin_key(x):
    return _bin_to_time_value(x)


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(mat, mode="reduced")
    return q


def _top_eigvecs(mat: np.ndarray, k: int) -> np.ndarray:
    w, v = np.linalg.eigh(mat)
    idx = np.argsort(w)[::-1][:k]
    return v[:, idx]


def _build_time_weights(t: np.ndarray, sigma: float) -> np.ndarray:
    dt2 = (t[:, None] - t[None, :]) ** 2
    w = np.exp(-dt2 / (2.0 * sigma**2))
    np.fill_diagonal(w, 0.0)
    return w


def build_observed_mask(
    da_input: xr.DataArray,
    zero_value: float,
    mode: str,
    replace_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    vals = da_input.values
    finite = np.isfinite(vals)

    if mode == "positive_only":
        obs = finite & (vals > zero_value)
    elif mode == "positive_only_except_replace":
        if replace_mask is None:
            raise ValueError("replace_mask is required when mode='positive_only_except_replace'")
        obs = finite & (vals > zero_value) & (~replace_mask.values.astype(bool))
    elif mode == "all_finite_except_replace":
        if replace_mask is None:
            raise ValueError("replace_mask is required when mode='all_finite_except_replace'")
        obs = finite & (~replace_mask.values.astype(bool))
    elif mode == "all_finite":
        obs = finite
    else:
        raise ValueError(f"Unsupported observed mode: {mode}")

    return xr.DataArray(obs, coords=da_input.coords, dims=da_input.dims, name="observed_mask")


def penalized_2d2_pca(
    da: xr.DataArray,
    time=None,
    d: int = 1,
    q: int = 1,
    lam: float = 0.1,
    sigma: float = 4.0,
    t0: int = 30,
    t: int = 30,
    seed: int = 0,
    observed_mask: Optional[xr.DataArray] = None,
    verbose: bool = False,
):
    rng = np.random.default_rng(seed)

    if time is None:
        time = da.coords["bin"].values
    time = np.array(time)
    m_time = len(time)

    a_list, m_list = [], []
    for w in time:
        ai = da.sel(bin=w).values.astype(float)
        if observed_mask is None:
            mi = np.isfinite(ai)
        else:
            mi = observed_mask.sel(bin=w).values.astype(bool)
        a_list.append(ai)
        m_list.append(mi)

    a = np.stack(a_list, axis=0)
    obs = np.stack(m_list, axis=0)
    _, m, n = a.shape

    tvals = np.array([_bin_to_time_value(x) for x in time], dtype=float)
    w_time = _build_time_weights(tvals, sigma=sigma)

    abar = np.where(obs, a, 0.0).sum(axis=0) / np.maximum(obs.sum(axis=0), 1.0)
    ac = np.where(obs, a - abar[None, :, :], 0.0)

    z = _orthonormalize(rng.standard_normal((m, q)))

    if verbose:
        print(f"[Phase 1] lambda=0, iter={t0}, M={m_time}, m={m}, n={n}, q={q}, d={d}")
    for _ in range(t0):
        zzt = z @ z.T
        astar0 = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astar0 += bi.T @ zzt @ bi
        astar0 /= m_time
        x = _top_eigvecs(astar0, d)

        xxt = x @ x.T
        astarstar0 = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar0 += bi @ xxt @ bi.T
        astarstar0 /= m_time
        z = _top_eigvecs(astarstar0, q)

    if verbose:
        print(f"[Phase 2] lambda={lam}, sigma={sigma}, iter={t}")
    for _ in range(t):
        zzt = z @ z.T
        astar = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astar += bi.T @ zzt @ bi
        astar /= m_time

        pen = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            for j in range(m_time):
                wij = w_time[i, j]
                if wij == 0.0:
                    continue
                oij = obs[i] & obs[j]
                if not np.any(oij):
                    continue
                dij = np.where(oij, a[i] - a[j], 0.0)
                pen += wij * (dij.T @ zzt @ dij)
        astar = astar - lam * pen
        x = _top_eigvecs(astar, d)

        xxt = x @ x.T
        astarstar = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar += bi @ xxt @ bi.T
        astarstar /= m_time

        pen2 = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            for j in range(m_time):
                wij = w_time[i, j]
                if wij == 0.0:
                    continue
                oij = obs[i] & obs[j]
                if not np.any(oij):
                    continue
                dij = np.where(oij, a[i] - a[j], 0.0)
                pen2 += wij * (dij @ xxt @ dij.T)
        astarstar = astarstar - lam * pen2
        z = _top_eigvecs(astarstar, q)

    c = np.zeros((m_time, q, d), dtype=float)
    for i in range(m_time):
        ai_eff = np.where(obs[i], a[i], abar)
        c[i] = z.T @ ai_eff @ x

    meta = {"bin": time, "t": tvals, "W": w_time, "Abar": abar}
    return x, z, c, meta


def reconstruct_full_tensor_from_p2d2_fit(
    x: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    meta: Dict,
    template_da: xr.DataArray,
) -> xr.DataArray:
    abar = meta["Abar"]
    bins = np.array(meta["bin"])
    full_vals = np.empty(template_da.shape, dtype=float)

    for ti, b in enumerate(bins):
        full_vals[..., ti] = abar + (z @ c[ti] @ x.T)

    da_full = template_da.copy(deep=True)
    da_full.values[:] = full_vals
    return da_full


def build_full_p2d2_truth(
    da_base: xr.DataArray,
    zero_value: float,
    pca_params: Dict,
    model_seed: int,
    truth_observed_mode: str = "positive_only",
    verbose: bool = False,
) -> xr.DataArray:
    """
    Fit penalized_2d2_pca on the truth-stage observed set and return the full tensor reconstruction.
    """
    observed_mask = build_observed_mask(
        da_input=da_base,
        zero_value=zero_value,
        mode=truth_observed_mode,
        replace_mask=None,
    )
    bins_use = np.array(da_base.coords["bin"].values)
    x, z, c, meta = penalized_2d2_pca(
        da=da_base,
        time=bins_use,
        d=pca_params["d"],
        q=pca_params["q"],
        lam=pca_params["lam"],
        sigma=pca_params["sigma"],
        t0=pca_params["T0"],
        t=pca_params["T"],
        seed=model_seed,
        observed_mask=observed_mask,
        verbose=verbose,
    )
    da_truth = reconstruct_full_tensor_from_p2d2_fit(
        x=x,
        z=z,
        c=c,
        meta=meta,
        template_da=da_base,
    )
    da_truth.name = "da_truth_sim1"
    return da_truth


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return np.nan
    return float(wasserstein_distance(x, y))


def evaluate_per_bin(
    da_true: xr.DataArray,
    da_before: xr.DataArray,
    da_after: xr.DataArray,
    bins: Iterable,
    eval_mask: Optional[xr.DataArray] = None,
    bin_dim: str = "bin",
) -> pd.DataFrame:
    rows = []
    for b in bins:
        yt = da_true.sel({bin_dim: b}).values.ravel()
        yb = da_before.sel({bin_dim: b}).values.ravel()
        ya = da_after.sel({bin_dim: b}).values.ravel()
        finite = np.isfinite(yt) & np.isfinite(yb) & np.isfinite(ya)
        if eval_mask is not None:
            mk = eval_mask.sel({bin_dim: b}).values.ravel().astype(bool)
            use = finite & mk
        else:
            use = finite

        yt_u, yb_u, ya_u = yt[use], yb[use], ya[use]
        n_eval = int(use.sum())
        if n_eval == 0:
            mse_before = mse_after = np.nan
            pear_before = pear_after = np.nan
            wd_before = wd_after = np.nan
        else:
            mse_before = float(np.mean((yb_u - yt_u) ** 2))
            mse_after = float(np.mean((ya_u - yt_u) ** 2))
            pear_before = _safe_pearson(yb_u, yt_u)
            pear_after = _safe_pearson(ya_u, yt_u)
            wd_before = _safe_wasserstein(yb_u, yt_u)
            wd_after = _safe_wasserstein(ya_u, yt_u)

        rows.append(
            {
                "bin": b,
                "n_eval": n_eval,
                "mse_before": mse_before,
                "mse_after": mse_after,
                "mse_improve": mse_before - mse_after,
                "pearson_before": pear_before,
                "pearson_after": pear_after,
                "pearson_improve": pear_after - pear_before,
                "wd_before": wd_before,
                "wd_after": wd_after,
                "wd_improve": wd_before - wd_after,
            }
        )
    return pd.DataFrame(rows)


def _weighted_avg(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.average(v[m], weights=w[m]))


def build_weighted_summary(df_per_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, g in df_per_bin.groupby("scope"):
        w = g["n_eval"].to_numpy(dtype=float)
        row = {"scope": scope, "n_eval": int(np.nansum(w))}
        for c in METRIC_COLS:
            row[c] = _weighted_avg(g[c], w)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scope").reset_index(drop=True)


def make_random_mask(
    da_ref: xr.DataArray,
    bins,
    frac: float,
    seed: int,
    zero_value: float,
    eligible_mode: str = "all_finite",
) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    da_sub = da_ref.sel(bin=bins)
    vals = da_sub.values
    finite = np.isfinite(vals)

    if eligible_mode == "positive_only":
        eligible = finite & (vals > zero_value)
    elif eligible_mode == "all_finite":
        eligible = finite
    else:
        raise ValueError(f"Unsupported eligible_mode: {eligible_mode}")

    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        mask = np.zeros(vals.shape, dtype=bool)
    else:
        n_pick = int(np.round(frac * idx.size))
        if frac > 0 and n_pick == 0:
            n_pick = 1
        n_pick = min(n_pick, idx.size)
        pick = rng.choice(idx, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)
        mask = np.zeros(vals.size, dtype=bool)
        mask[pick] = True
        mask = mask.reshape(vals.shape)

    return xr.DataArray(mask, coords=da_sub.coords, dims=da_sub.dims, name="rand_mask_to_impute")


def _trajectory_mean(values: np.ndarray, observed: np.ndarray, zero_value: float) -> float:
    obs_vals = values[observed]
    if obs_vals.size > 0:
        return float(np.mean(obs_vals))
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size > 0:
        return float(np.mean(finite_vals))
    return float(zero_value)


def _trajectory_median(values: np.ndarray, observed: np.ndarray, zero_value: float) -> float:
    obs_vals = values[observed]
    if obs_vals.size > 0:
        return float(np.median(obs_vals))
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size > 0:
        return float(np.median(finite_vals))
    return float(zero_value)


def _find_prev_next_indices(observed: np.ndarray, idx: int) -> Tuple[Optional[int], Optional[int]]:
    prev_candidates = np.flatnonzero(observed[:idx])
    next_candidates = np.flatnonzero(observed[idx + 1 :])
    prev_idx = int(prev_candidates[-1]) if prev_candidates.size > 0 else None
    next_idx = int(next_candidates[0] + idx + 1) if next_candidates.size > 0 else None
    return prev_idx, next_idx


def _fill_last(values: np.ndarray, observed: np.ndarray, idx: int, mean_value: float) -> float:
    prev_idx, next_idx = _find_prev_next_indices(observed, idx)
    if prev_idx is not None:
        return float(values[prev_idx])
    if next_idx is not None:
        return float(values[next_idx])
    return float(mean_value)


def _fill_linear(values: np.ndarray, observed: np.ndarray, times: np.ndarray, idx: int, mean_value: float) -> float:
    prev_idx, next_idx = _find_prev_next_indices(observed, idx)
    if prev_idx is not None and next_idx is not None:
        left_t = times[prev_idx]
        right_t = times[next_idx]
        if right_t == left_t:
            return float(0.5 * (values[prev_idx] + values[next_idx]))
        left_dist = times[idx] - left_t
        right_dist = right_t - times[idx]
        return float((right_dist * values[prev_idx] + left_dist * values[next_idx]) / (left_dist + right_dist))
    return _fill_last(values, observed, idx, mean_value)


def _fill_weighted_time_average(
    values: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    idx: int,
    mean_value: float,
) -> float:
    prev_idx, next_idx = _find_prev_next_indices(observed, idx)
    if prev_idx is not None and next_idx is not None:
        left_dist = max(times[idx] - times[prev_idx], 1e-12)
        right_dist = max(times[next_idx] - times[idx], 1e-12)
        w_left = 1.0 / left_dist
        w_right = 1.0 / right_dist
        return float((w_left * values[prev_idx] + w_right * values[next_idx]) / (w_left + w_right))
    return _fill_last(values, observed, idx, mean_value)


def _fill_temporal_knn(
    values: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    idx: int,
    mean_value: float,
    k: int,
    bandwidth: float,
) -> float:
    obs_idx = np.flatnonzero(observed)
    if obs_idx.size == 0:
        return float(mean_value)
    dists = np.abs(times[obs_idx] - times[idx])
    order = np.argsort(dists)[: max(1, min(k, obs_idx.size))]
    nbr_idx = obs_idx[order]
    nbr_dists = dists[order]
    bw = max(float(bandwidth), 1e-12)
    weights = np.exp(-0.5 * (nbr_dists / bw) ** 2)
    if not np.any(weights > 0):
        weights = np.ones_like(weights, dtype=float)
    return float(np.average(values[nbr_idx], weights=weights))


def _impute_local_trajectory(
    values: np.ndarray,
    replace: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    method: str,
    method_params: Dict,
    zero_value: float,
) -> np.ndarray:
    out = values.copy()
    mean_value = _trajectory_mean(values, observed, zero_value)
    median_value = _trajectory_median(values, observed, zero_value)
    spline_min_points = int(method_params.get("min_spline_points", 4))

    spline_model = None
    obs_idx = np.flatnonzero(observed)
    if method == "spline" and obs_idx.size >= spline_min_points:
        try:
            spline_model = CubicSpline(times[obs_idx], values[obs_idx], extrapolate=False)
        except Exception:
            spline_model = None

    for idx in np.flatnonzero(replace):
        if method == "mean":
            out[idx] = mean_value
        elif method == "median":
            out[idx] = median_value
        elif method == "last":
            out[idx] = _fill_last(values, observed, idx, mean_value)
        elif method == "weighted_time_average":
            out[idx] = _fill_weighted_time_average(values, observed, times, idx, mean_value)
        elif method == "temporal_knn":
            out[idx] = _fill_temporal_knn(
                values,
                observed,
                times,
                idx,
                mean_value,
                k=int(method_params.get("k", 3)),
                bandwidth=float(method_params.get("bandwidth", 4.0)),
            )
        elif method == "spline":
            pred = np.nan
            if spline_model is not None:
                pred = float(spline_model(times[idx]))
            if np.isfinite(pred):
                out[idx] = pred
            else:
                out[idx] = _fill_linear(values, observed, times, idx, mean_value)
        elif method == "equal":
            out[idx] = float(method_params.get("equal_value", 0.0))
        else:
            raise ValueError(f"Unsupported local method: {method}")
    return out


def _mean_initialize_array(
    values: np.ndarray,
    observed_mask: np.ndarray,
    zero_value: float,
) -> np.ndarray:
    initialized = values.copy()
    n_participants, n_taxa, _ = initialized.shape
    for i in range(n_participants):
        for j in range(n_taxa):
            obs = observed_mask[i, j, :]
            mean_value = _trajectory_mean(initialized[i, j, :], obs, zero_value)
            initialized[i, j, ~obs] = mean_value
    return initialized


def _fit_transition_models(
    init_array: np.ndarray,
    bins: np.ndarray,
    method: str,
    method_params: Dict,
) -> List[Optional[object]]:
    tensor = np.transpose(init_array, (0, 2, 1))
    times = np.array([_bin_to_time_value(x) for x in bins], dtype=float)
    x_rows = []
    y_rows = []

    for p in range(tensor.shape[0]):
        for b in range(1, tensor.shape[1]):
            dt = times[b] - times[b - 1]
            if dt <= 0:
                continue
            prev_state = tensor[p, b - 1, :]
            curr_state = tensor[p, b, :]
            if not (np.all(np.isfinite(prev_state)) and np.all(np.isfinite(curr_state))):
                continue
            x_rows.append(np.concatenate([prev_state, [dt]]))
            y_rows.append(curr_state)

    n_taxa = tensor.shape[2]
    if len(x_rows) == 0:
        return [None] * n_taxa

    X = np.vstack(x_rows)
    Y = np.vstack(y_rows)
    models: List[Optional[object]] = []
    alpha = float(method_params.get("alpha", 1e-3))

    for j in range(n_taxa):
        try:
            if method == "dbn_dense":
                model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
                model.fit(X, Y[:, j])
            elif method == "dbn_sparse":
                model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000, random_state=0)
                model.fit(X, Y[:, j])
            else:
                raise ValueError(f"Unsupported transition model method: {method}")
            models.append(model)
        except Exception:
            models.append(None)
    return models


def _apply_dbn_refinement(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
    bins: np.ndarray,
    method: str,
    method_params: Dict,
    zero_value: float,
) -> xr.DataArray:
    values = da_input.values.astype(float)
    replace = replace_mask.values.astype(bool)
    observed = observed_mask.values.astype(bool)
    init_array = _mean_initialize_array(values, observed, zero_value)
    models = _fit_transition_models(init_array, bins, method, method_params)
    refined = init_array.copy()
    times = np.array([_bin_to_time_value(x) for x in bins], dtype=float)

    for b in range(1, len(bins)):
        dt = times[b] - times[b - 1]
        if dt <= 0:
            continue
        for p in range(values.shape[0]):
            prev_state = init_array[p, :, b - 1]
            features = np.concatenate([prev_state, [dt]])[None, :]
            for j in np.flatnonzero(replace[p, :, b]):
                model = models[j]
                if model is None:
                    continue
                try:
                    pred = float(model.predict(features)[0])
                except Exception:
                    pred = np.nan
                if np.isfinite(pred):
                    refined[p, j, b] = pred

    out = da_input.copy(deep=True)
    out_vals = out.values.copy()
    out_vals[replace] = refined[replace]
    out.values[:] = out_vals
    return out


def _fit_glv_models(
    init_array: np.ndarray,
    bins: np.ndarray,
    method_params: Dict,
) -> List[Optional[object]]:
    tensor = np.transpose(init_array, (0, 2, 1))
    times = np.array([_bin_to_time_value(x) for x in bins], dtype=float)
    x_rows = []
    y_rows = []

    for p in range(tensor.shape[0]):
        for b in range(1, tensor.shape[1]):
            dt = times[b] - times[b - 1]
            if dt <= 0:
                continue
            prev_state = tensor[p, b - 1, :]
            curr_state = tensor[p, b, :]
            if not (np.all(np.isfinite(prev_state)) and np.all(np.isfinite(curr_state))):
                continue
            x_rows.append(prev_state)
            y_rows.append((curr_state - prev_state) / dt)

    n_taxa = tensor.shape[2]
    if len(x_rows) == 0:
        return [None] * n_taxa

    X = np.vstack(x_rows)
    Y = np.vstack(y_rows)
    alpha = float(method_params.get("alpha", 1e-3))
    models: List[Optional[object]] = []

    for j in range(n_taxa):
        try:
            model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
            model.fit(X, Y[:, j])
            models.append(model)
        except Exception:
            models.append(None)
    return models


def _apply_glv_refinement(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
    bins: np.ndarray,
    method_params: Dict,
    zero_value: float,
) -> xr.DataArray:
    # This is a gLV-style discrete-time refinement on the current log tensor.
    # It is intentionally used as a practical benchmark approximation, not as a
    # strict abundance-scale ecological gLV model.
    values = da_input.values.astype(float)
    replace = replace_mask.values.astype(bool)
    observed = observed_mask.values.astype(bool)
    init_array = _mean_initialize_array(values, observed, zero_value)
    models = _fit_glv_models(init_array, bins, method_params)
    refined = init_array.copy()
    times = np.array([_bin_to_time_value(x) for x in bins], dtype=float)

    for b in range(1, len(bins)):
        dt = times[b] - times[b - 1]
        if dt <= 0:
            continue
        for p in range(values.shape[0]):
            prev_state = init_array[p, :, b - 1]
            features = prev_state[None, :]
            for j in np.flatnonzero(replace[p, :, b]):
                model = models[j]
                if model is None:
                    continue
                try:
                    slope = float(model.predict(features)[0])
                except Exception:
                    slope = np.nan
                if np.isfinite(slope):
                    refined[p, j, b] = prev_state[j] + dt * slope

    out = da_input.copy(deep=True)
    out_vals = out.values.copy()
    out_vals[replace] = refined[replace]
    out.values[:] = out_vals
    return out


def _impute_local_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
    method: str,
    method_params: Dict,
    zero_value: float,
) -> xr.DataArray:
    values = da_input.values.astype(float)
    replace = replace_mask.values.astype(bool)
    observed = observed_mask.values.astype(bool)
    out_vals = values.copy()
    times = np.array([_bin_to_time_value(x) for x in da_input.coords["bin"].values], dtype=float)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.any(replace[i, j, :]):
                continue
            out_vals[i, j, :] = _impute_local_trajectory(
                values=values[i, j, :],
                replace=replace[i, j, :],
                observed=observed[i, j, :],
                times=times,
                method=method,
                method_params=method_params,
                zero_value=zero_value,
            )

    out = da_input.copy(deep=True)
    out.values[:] = np.where(replace, out_vals, values)
    return out


def impute_by_mask(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    method: str,
    method_params: Optional[Dict] = None,
    pca_params: Optional[Dict] = None,
    model_seed: int = 0,
    zero_value: float = 0.0,
    observed_mode: str = "positive_only_except_replace",
    verbose: bool = False,
) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Unified second-stage imputation dispatcher.

    Experimental fallback policy:
    - `spline`: if available observed points < 4, fallback to linear interpolation.
    - `linear` fallback inside spline: if two-sided interpolation is unavailable, fallback to previous/next (`last` rule).
    - `temporal_knn`: if fewer than K neighbors exist, use available neighbors; if none exist, fallback to trajectory mean.
    - `dbn_dense`, `dbn_sparse`, and `glv`: always start from mean initialization, then refine only masked cells with the fitted dynamics model.
    - `equal`: always available with a fixed log-scale constant, defaulting to 0.0 unless overridden.
    """
    bins_use = np.array(da_input.coords["bin"].values)
    method_params = {} if method_params is None else dict(method_params)

    observed_mask = build_observed_mask(
        da_input=da_input,
        zero_value=zero_value,
        mode=observed_mode,
        replace_mask=replace_mask,
    )

    bins_trained = np.array(bins_use)

    if method == "penalized_2d2_pca":
        if pca_params is None:
            raise ValueError("pca_params is required for method='penalized_2d2_pca'")
        x, z, c, meta = penalized_2d2_pca(
            da=da_input,
            time=bins_use,
            d=pca_params["d"],
            q=pca_params["q"],
            lam=pca_params["lam"],
            sigma=pca_params["sigma"],
            t0=pca_params["T0"],
            t=pca_params["T"],
            seed=model_seed,
            observed_mask=observed_mask,
            verbose=verbose,
        )

        bins_trained = np.array(meta["bin"])
        abar = meta["Abar"]
        da_imp = da_input.copy(deep=True)

        for ti, b in enumerate(bins_trained):
            ai = da_imp.sel(bin=b).values
            ahat = abar + (z @ c[ti] @ x.T)
            mk = replace_mask.sel(bin=b).values.astype(bool)
            if np.any(mk):
                ai_new = ai.copy()
                ai_new[mk] = ahat[mk]
                da_imp.loc[dict(bin=b)] = ai_new
    elif method in {"mean", "median", "last", "weighted_time_average", "temporal_knn", "spline", "equal"}:
        da_imp = _impute_local_method(
            da_input=da_input,
            replace_mask=replace_mask,
            observed_mask=observed_mask,
            method=method,
            method_params=method_params,
            zero_value=zero_value,
        )
    elif method in {"dbn_dense", "dbn_sparse"}:
        da_imp = _apply_dbn_refinement(
            da_input=da_input,
            replace_mask=replace_mask,
            observed_mask=observed_mask,
            bins=bins_use,
            method=method,
            method_params=method_params,
            zero_value=zero_value,
        )
    elif method == "glv":
        da_imp = _apply_glv_refinement(
            da_input=da_input,
            replace_mask=replace_mask,
            observed_mask=observed_mask,
            bins=bins_use,
            method_params=method_params,
            zero_value=zero_value,
        )
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    da_imp.name = "log_abundance_imputed"
    return da_imp, bins_trained


def run_one_replication(
    da_truth: xr.DataArray,
    simulation: str,
    frac: float,
    replicate: int,
    mask_seed: int,
    model_seed: int,
    zero_value: float,
    pca_params: Dict,
    methods: List[str],
    impute_method_params: Dict[str, Dict],
    second_observed_mode: str = "positive_only_except_replace",
    mask_eligible_mode: str = "all_finite",
    outdir: Optional[Path] = None,
    eps: float = 1e-3,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bins_use = np.array(da_truth.coords["bin"].values)

    da_rand_mask = make_random_mask(
        da_truth,
        bins_use,
        frac=frac,
        seed=mask_seed,
        zero_value=zero_value,
        eligible_mode=mask_eligible_mode,
    )

    da_before = da_truth.copy(deep=True)
    vals = da_before.sel(bin=bins_use).values.copy()
    vals[da_rand_mask.values.astype(bool)] = zero_value
    da_before.loc[dict(bin=bins_use)] = vals
    da_before.name = "log_abundance_zero_inflated"

    per_bin_tables = []
    summary_tables = []
    taxon_marginal_stats_tables = []
    taxon_marginal_eval_tables = []
    taxon_marginal_method_tables = []
    taxon_bin_stats_tables = []
    taxon_trajectory_eval_tables = []
    taxon_trajectory_detail_tables = []
    taxon_trajectory_method_tables = []
    bin_distribution_tables = []
    da_truth_abund = to_abundance_space(da_truth, eps=eps, name="truth_abundance_space")
    da_before_abund = to_abundance_space(da_before, eps=eps, name="before_abundance_space")

    for method in methods:
        da_after, bins_trained = impute_by_mask(
            da_input=da_before,
            replace_mask=da_rand_mask,
            method=method,
            method_params=impute_method_params.get(method, {}),
            pca_params=pca_params,
            model_seed=model_seed,
            zero_value=zero_value,
            observed_mode=second_observed_mode,
            verbose=verbose,
        )

        df_masked = evaluate_per_bin(
            da_true=da_truth,
            da_before=da_before,
            da_after=da_after,
            bins=bins_trained,
            eval_mask=da_rand_mask,
            bin_dim="bin",
        )
        df_masked["scope"] = "MaskedCells"

        df_all = evaluate_per_bin(
            da_true=da_truth,
            da_before=da_before,
            da_after=da_after,
            bins=bins_trained,
            eval_mask=None,
            bin_dim="bin",
        )
        df_all["scope"] = "AllCells"

        method_group = METHOD_GROUP_MAP.get(method, "other")
        init_method = "mean" if method in {"dbn_dense", "dbn_sparse", "glv"} else method

        df_per_bin = pd.concat([df_masked, df_all], ignore_index=True)
        df_per_bin["bin_order"] = df_per_bin["bin"].map(_bin_key)
        df_per_bin = df_per_bin.sort_values(["scope", "bin_order"]).drop(columns=["bin_order"])
        df_per_bin["simulation"] = simulation
        df_per_bin["frac"] = float(frac)
        df_per_bin["replicate"] = int(replicate)
        df_per_bin["mask_seed"] = int(mask_seed)
        df_per_bin["model_seed"] = int(model_seed)
        df_per_bin["method"] = method
        df_per_bin["init_method"] = init_method
        df_per_bin["method_group"] = method_group
        per_bin_tables.append(df_per_bin)

        df_summary = build_weighted_summary(df_per_bin)
        df_summary["simulation"] = simulation
        df_summary["frac"] = float(frac)
        df_summary["replicate"] = int(replicate)
        df_summary["mask_seed"] = int(mask_seed)
        df_summary["model_seed"] = int(model_seed)
        df_summary["method"] = method
        df_summary["init_method"] = init_method
        df_summary["method_group"] = method_group
        summary_tables.append(df_summary)

        da_after_abund = to_abundance_space(da_after, eps=eps, name=f"{method}_after_abundance_space")

        df_taxon_stats = pd.concat(
            [
                compute_taxon_marginal_stats(
                    da_abund=da_truth_abund,
                    frac=frac,
                    repeat=replicate,
                    method=method,
                    dataset_role="truth",
                    mask_seed=mask_seed,
                    algo_seed=model_seed,
                ),
                compute_taxon_marginal_stats(
                    da_abund=da_before_abund,
                    frac=frac,
                    repeat=replicate,
                    method=method,
                    dataset_role="before",
                    mask_seed=mask_seed,
                    algo_seed=model_seed,
                ),
                compute_taxon_marginal_stats(
                    da_abund=da_after_abund,
                    frac=frac,
                    repeat=replicate,
                    method=method,
                    dataset_role="after",
                    mask_seed=mask_seed,
                    algo_seed=model_seed,
                ),
            ],
            ignore_index=True,
        )
        df_taxon_eval, df_taxon_method = evaluate_taxon_marginal_preservation(
            df_taxon_stats=df_taxon_stats,
            frac=frac,
            repeat=replicate,
            method=method,
            mask_seed=mask_seed,
            algo_seed=model_seed,
        )
        taxon_marginal_stats_tables.append(df_taxon_stats)
        taxon_marginal_eval_tables.append(df_taxon_eval)
        taxon_marginal_method_tables.append(df_taxon_method)

        df_taxon_bin_stats = pd.concat(
            [
                compute_taxon_bin_stats(
                    da_abund=da_truth_abund,
                    frac=frac,
                    repeat=replicate,
                    method=method,
                    dataset_role="truth",
                    mask_seed=mask_seed,
                    algo_seed=model_seed,
                ),
                compute_taxon_bin_stats(
                    da_abund=da_before_abund,
                    frac=frac,
                    repeat=replicate,
                    method=method,
                    dataset_role="before",
                    mask_seed=mask_seed,
                    algo_seed=model_seed,
                ),
                compute_taxon_bin_stats(
                    da_abund=da_after_abund,
                    frac=frac,
                    repeat=replicate,
                    method=method,
                    dataset_role="after",
                    mask_seed=mask_seed,
                    algo_seed=model_seed,
                ),
            ],
            ignore_index=True,
        )
        (
            df_taxon_traj_eval,
            df_taxon_traj_detail,
            df_taxon_traj_method,
        ) = evaluate_taxon_trajectory_preservation(
            df_taxon_bin_stats=df_taxon_bin_stats,
            da_truth_abund=da_truth_abund,
            da_before_abund=da_before_abund,
            da_after_abund=da_after_abund,
            frac=frac,
            repeat=replicate,
            method=method,
            mask_seed=mask_seed,
            algo_seed=model_seed,
        )
        taxon_bin_stats_tables.append(df_taxon_bin_stats)
        taxon_trajectory_eval_tables.append(df_taxon_traj_eval)
        taxon_trajectory_detail_tables.append(df_taxon_traj_detail)
        taxon_trajectory_method_tables.append(df_taxon_traj_method)

        df_bin_distribution = evaluate_bin_distribution_preservation(
            da_truth_abund=da_truth_abund,
            da_before_abund=da_before_abund,
            da_after_abund=da_after_abund,
            frac=frac,
            repeat=replicate,
            method=method,
            mask_seed=mask_seed,
            algo_seed=model_seed,
        )
        bin_distribution_tables.append(df_bin_distribution)

    df_per_bin_all = pd.concat(per_bin_tables, ignore_index=True)
    df_summary_all = pd.concat(summary_tables, ignore_index=True)
    ordered = [
        "simulation",
        "frac",
        "replicate",
        "scope",
        "method",
        "init_method",
        "method_group",
        "mask_seed",
        "model_seed",
        "n_eval",
    ] + METRIC_COLS
    df_summary_all = df_summary_all[ordered]

    if outdir is not None:
        frac_str = _format_frac(frac)
        df_taxon_marginal_stats_all = pd.concat(taxon_marginal_stats_tables, ignore_index=True)
        df_taxon_marginal_eval_all = pd.concat(taxon_marginal_eval_tables, ignore_index=True)
        df_taxon_marginal_method_all = pd.concat(taxon_marginal_method_tables, ignore_index=True)
        df_taxon_bin_stats_all = pd.concat(taxon_bin_stats_tables, ignore_index=True)
        df_taxon_trajectory_eval_all = pd.concat(taxon_trajectory_eval_tables, ignore_index=True)
        df_taxon_trajectory_detail_all = pd.concat(taxon_trajectory_detail_tables, ignore_index=True)
        df_taxon_trajectory_method_all = pd.concat(taxon_trajectory_method_tables, ignore_index=True)
        df_bin_distribution_all = pd.concat(bin_distribution_tables, ignore_index=True)
        df_abundance_eval_summary = build_abundance_eval_repeat_summary(
            df_taxon_marginal_method=df_taxon_marginal_method_all,
            df_taxon_traj_method=df_taxon_trajectory_method_all,
            df_bin_distribution=df_bin_distribution_all,
        )

        df_taxon_marginal_stats_all.to_csv(
            outdir / f"taxon_marginal_stats_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
        df_taxon_marginal_eval_all.to_csv(
            outdir / f"taxon_marginal_eval_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
        df_taxon_bin_stats_all.to_csv(
            outdir / f"taxon_bin_stats_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
        df_taxon_trajectory_eval_all.to_csv(
            outdir / f"taxon_trajectory_eval_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
        df_taxon_trajectory_detail_all.to_csv(
            outdir / f"taxon_trajectory_detail_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
        df_bin_distribution_all.to_csv(
            outdir / f"bin_distribution_eval_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
        df_abundance_eval_summary.to_csv(
            outdir / f"abundance_eval_summary_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
    return df_per_bin_all, df_summary_all


def run_repeated_experiments(
    da_truth: xr.DataArray,
    simulation: str,
    frac: float,
    n_repeats: int,
    seed_base: int,
    zero_value: float,
    pca_params: Dict,
    methods: List[str],
    impute_method_params: Dict[str, Dict],
    second_observed_mode: str = "positive_only_except_replace",
    mask_eligible_mode: str = "all_finite",
    collect_per_bin: bool = True,
    outdir: Optional[Path] = None,
    eps: float = 1e-3,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    rep_rows = []
    bin_rows = []
    for replicate in range(n_repeats):
        mask_seed = seed_base + replicate
        model_seed = seed_base + 10000 + replicate
        df_per_bin, df_summary = run_one_replication(
            da_truth=da_truth,
            simulation=simulation,
            frac=frac,
            replicate=replicate,
            mask_seed=mask_seed,
            model_seed=model_seed,
            zero_value=zero_value,
            pca_params=pca_params,
            methods=methods,
            impute_method_params=impute_method_params,
            second_observed_mode=second_observed_mode,
            mask_eligible_mode=mask_eligible_mode,
            outdir=outdir,
            eps=eps,
            verbose=verbose,
        )
        rep_rows.append(df_summary)
        if collect_per_bin:
            bin_rows.append(df_per_bin)

    df_reps = pd.concat(rep_rows, ignore_index=True)
    df_bins = pd.concat(bin_rows, ignore_index=True) if collect_per_bin else None
    return df_reps, df_bins


def summarize_replicates(df_replicates: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["simulation", "frac", "scope", "method", "replicate"]
    if "init_method" in df_replicates.columns:
        id_cols.append("init_method")
    if "method_group" in df_replicates.columns:
        id_cols.append("method_group")
    long_df = df_replicates[id_cols + METRIC_COLS].melt(
        id_vars=id_cols,
        value_vars=METRIC_COLS,
        var_name="metric",
        value_name="value",
    )
    group_cols = ["simulation", "frac", "scope", "method", "metric"]
    if "init_method" in long_df.columns:
        group_cols.insert(4, "init_method")
    if "method_group" in long_df.columns:
        insert_at = 5 if "init_method" in long_df.columns else 4
        group_cols.insert(insert_at, "method_group")
    g = long_df.groupby(group_cols, as_index=False)
    summary = g.agg(
        mean=("value", "mean"),
        sd=("value", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
        n_repeats=("value", lambda x: int(np.sum(np.isfinite(x)))),
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n_repeats"].replace(0, np.nan))
    summary["ci_low"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci_high"] = summary["mean"] + 1.96 * summary["se"]
    return summary.sort_values(group_cols).reset_index(drop=True)


def load_base_tensor_with_missing_mask(tax_path: str, meta_path: str, eps: float) -> Tuple[xr.DataArray, xr.DataArray]:
    tax = pd.read_csv(tax_path, sep="\t")
    tax = tax.rename(columns={tax.columns[0]: "taxon"}).set_index("taxon")
    meta = pd.read_excel(meta_path)

    tax.columns = tax.columns.str.replace(r"_profile$", "", regex=True)
    meta_mgx = (
        meta.loc[meta["data_type"].eq("metagenomics"), ["External ID", "Participant ID", "week_num"]]
        .dropna(subset=["External ID", "Participant ID", "week_num"])
        .rename(columns={"External ID": "external_id", "Participant ID": "participant", "week_num": "week"})
    )
    meta_mgx["week"] = meta_mgx["week"].astype(int)
    meta_mgx["participant"] = meta_mgx["participant"].astype(str)

    idx = tax.index.astype(str)
    has_species = idx.str.contains("s__", na=False)
    not_euk = ~idx.str.contains("k__Eukaryota", na=False)
    has_unclassified = idx.str.contains("unclassified", na=False)
    has_cag = idx.str.contains(r"CAG[_-]?\d+", regex=True, na=False)
    bad_unclassified = has_unclassified & (~has_cag)
    tax = tax.loc[has_species & not_euk & (~bad_unclassified)].copy()

    common = sorted(set(tax.columns).intersection(set(meta_mgx["external_id"])))
    tax = tax[common]
    meta_mgx = meta_mgx.drop_duplicates("external_id").set_index("external_id").loc[common]

    x = tax.T
    x.index.name = "external_id"
    x = x.join(meta_mgx, how="left")
    taxa = tax.index.astype(str).tolist()

    x_taxa = x[taxa].copy()
    x_taxa["participant"] = x["participant"].astype(str)
    x_taxa["week"] = x["week"].astype(int)
    x_taxa["bin"] = x_taxa["week"].map(week_to_bin)
    x_taxa = x_taxa.dropna(subset=["bin"])
    x_taxa = x_taxa.set_index(["participant", "bin"])[taxa]
    x_agg = x_taxa.groupby(level=["participant", "bin"]).mean()

    participants = sorted(x_agg.index.get_level_values("participant").unique().tolist())
    bins = ["0"] + [f"{s}-{s+3}" for s in range(1, 37, 4)]
    full_index = pd.MultiIndex.from_product([participants, bins], names=["participant", "bin"])
    x_full = x_agg.reindex(full_index)

    init_missing_mask_df = x_full.isna()
    x_full_0 = x_full.fillna(0.0)
    x_full_log = np.log10(x_full_0 + eps)

    arr = x_full_log.to_numpy().reshape(len(participants), len(bins), len(taxa))
    tensor = np.transpose(arr, (0, 2, 1))
    da = xr.DataArray(
        tensor,
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="abundance_binned_log",
    )

    mask_arr = init_missing_mask_df.to_numpy().reshape(len(participants), len(bins), len(taxa))
    mask_tensor = np.transpose(mask_arr, (0, 2, 1))
    init_missing_mask = xr.DataArray(
        mask_tensor.astype(bool),
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="initial_missing_mask",
    )
    return da, init_missing_mask


def prepare_truth_data_for_sim1(
    da_base: xr.DataArray,
    init_missing_mask: xr.DataArray,
    zero_value: float,
    pca_params: Dict,
    seed_base: int,
    truth_observed_mode: str = "positive_only",
    verbose: bool = False,
) -> xr.DataArray:
    del init_missing_mask
    da_truth = build_full_p2d2_truth(
        da_base=da_base.copy(deep=True),
        zero_value=zero_value,
        pca_params=pca_params,
        model_seed=seed_base + 500000,
        truth_observed_mode=truth_observed_mode,
        verbose=verbose,
    )
    return da_truth


def _safe_output_path(outdir: Path, name: str) -> Path:
    target = outdir / name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outdir / f"{stem}_{ts}{suffix}"


def run_sim1_pipeline(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eps = args.eps
    zero_value = np.log10(eps)
    pca_params = {
        "d": args.d,
        "q": args.q,
        "lam": args.lam,
        "sigma": args.sigma,
        "T0": args.t0,
        "T": args.t,
    }
    methods = list(dict.fromkeys(args.methods))
    unsupported = sorted(set(methods) - set(DEFAULT_METHODS))
    if unsupported:
        raise ValueError(f"Unsupported methods requested: {unsupported}")
    impute_method_params = {
        "mean": {},
        "median": {},
        "last": {},
        "weighted_time_average": {},
        "temporal_knn": {
            "k": args.temporal_knn_k,
            "bandwidth": args.temporal_knn_bandwidth,
        },
        "spline": {
            "min_spline_points": args.spline_min_points,
        },
        "equal": {
            "equal_value": args.equal_value,
        },
        "dbn_dense": {
            "alpha": args.dbn_dense_alpha,
        },
        "dbn_sparse": {
            "alpha": args.dbn_sparse_alpha,
        },
        "glv": {
            "alpha": args.glv_alpha,
        },
    }

    da_base, init_missing_mask = load_base_tensor_with_missing_mask(args.tax_path, args.meta_path, eps=eps)

    da_truth = prepare_truth_data_for_sim1(
        da_base=da_base,
        init_missing_mask=init_missing_mask,
        zero_value=zero_value,
        pca_params=pca_params,
        seed_base=args.seed_base,
        truth_observed_mode=args.truth_observed_mode,
        verbose=args.verbose,
    )

    rep_tables = []
    bin_tables = []
    for frac in args.fracs:
        frac_seed_base = args.seed_base + int(round(frac * 1_000_000))
        df_reps, df_bins = run_repeated_experiments(
            da_truth=da_truth,
            simulation="sim1",
            frac=float(frac),
            n_repeats=args.n_repeats,
            seed_base=frac_seed_base,
            zero_value=zero_value,
            pca_params=pca_params,
            methods=methods,
            impute_method_params=impute_method_params,
            second_observed_mode=args.second_observed_mode,
            mask_eligible_mode=args.mask_eligible_mode,
            collect_per_bin=True,
            outdir=outdir,
            eps=eps,
            verbose=args.verbose,
        )
        rep_tables.append(df_reps)
        if df_bins is not None:
            bin_tables.append(df_bins)
        merge_abundance_eval_outputs(outdir=outdir)

    all_reps = pd.concat(rep_tables, ignore_index=True)
    all_reps = all_reps.sort_values(["simulation", "frac", "replicate", "scope", "method"]).reset_index(drop=True)
    summary = summarize_replicates(all_reps)

    reps_path = _safe_output_path(outdir, "all_replicates_metrics_sim1.csv")
    summary_path = _safe_output_path(outdir, "summary_with_errorbar_sim1.csv")
    bins_path = _safe_output_path(outdir, "per_bin_metrics_sim1.csv")
    truth_path = _safe_output_path(outdir, "truth_tensor_sim1.nc")

    all_reps.to_csv(reps_path, index=False)
    summary.to_csv(summary_path, index=False)
    if len(bin_tables) > 0:
        all_bins = pd.concat(bin_tables, ignore_index=True)
        all_bins = all_bins.sort_values(["simulation", "frac", "replicate", "scope", "method", "bin"]).reset_index(drop=True)
        all_bins.to_csv(bins_path, index=False)
    da_truth.to_netcdf(truth_path)

    print(f"Saved replicate-level table: {reps_path}")
    print(f"Saved summary table: {summary_path}")
    if len(bin_tables) > 0:
        print(f"Saved per-bin table: {bins_path}")
    print(f"Saved truth tensor: {truth_path}")
    merge_abundance_eval_outputs(outdir=outdir)


# =====================
# Abundance-space distribution / trajectory evaluation
# =====================
def to_abundance_space(
    da_log: xr.DataArray,
    eps: float = 1e-3,
    name: str = "abundance_space",
) -> xr.DataArray:
    """Invert log10(percent + eps) values back to abundance space."""
    da_log = _validate_da(da_log, name=name)
    vals = np.power(10.0, da_log.values.astype(float)) - float(eps)
    vals = np.where(vals < 0.0, 0.0, vals)
    da_abund = xr.DataArray(
        vals,
        coords=da_log.coords,
        dims=da_log.dims,
        name=name,
        attrs=dict(da_log.attrs),
    )
    da_abund.attrs["eps"] = float(eps)
    da_abund.attrs["source_scale"] = "log10(percent+eps)"
    return da_abund


def _empty_df(columns: List[str]) -> pd.DataFrame:
    """Create an empty DataFrame with a fixed schema."""
    return pd.DataFrame(columns=columns)


def _mean_euclid_2d(
    x_true_mean: np.ndarray,
    x_true_sd: np.ndarray,
    x_pred_mean: np.ndarray,
    x_pred_sd: np.ndarray,
) -> float:
    """Compute the average 2D Euclidean distance after finite filtering."""
    xtm = np.asarray(x_true_mean, dtype=float)
    xts = np.asarray(x_true_sd, dtype=float)
    xpm = np.asarray(x_pred_mean, dtype=float)
    xps = np.asarray(x_pred_sd, dtype=float)
    mask = np.isfinite(xtm) & np.isfinite(xts) & np.isfinite(xpm) & np.isfinite(xps)
    if not np.any(mask):
        return np.nan
    dist = np.sqrt((xpm[mask] - xtm[mask]) ** 2 + (xps[mask] - xts[mask]) ** 2)
    return float(np.mean(dist)) if dist.size > 0 else np.nan


def _collect_aligned_role_stats(
    df_stats: pd.DataFrame,
    key_cols: List[str],
    value_cols: List[str],
) -> pd.DataFrame:
    """Align truth/before/after statistics into a single wide table."""
    parts = []
    for role in ["truth", "before", "after"]:
        role_df = df_stats.loc[df_stats["dataset_role"].eq(role), key_cols + value_cols].copy()
        role_df = role_df.rename(columns={col: f"{col}_{role}" for col in value_cols})
        parts.append(role_df)
    aligned = parts[0]
    for role_df in parts[1:]:
        aligned = aligned.merge(role_df, on=key_cols, how="outer")
    return aligned


def _distribution_wd_from_series(x_true: pd.Series, x_pred: pd.Series) -> Tuple[float, int]:
    """Compute 1D Wasserstein distance after finite filtering on aligned values."""
    a = pd.to_numeric(x_true, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(x_pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    n_used = int(mask.sum())
    if n_used == 0:
        return np.nan, 0
    return _safe_wasserstein(a[mask], b[mask]), n_used


def _mean_l2_distance(x_true: np.ndarray, x_pred: np.ndarray) -> Tuple[float, int]:
    """Compute L2 distance over aligned finite entries."""
    a = np.asarray(x_true, dtype=float)
    b = np.asarray(x_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    n_used = int(mask.sum())
    if n_used == 0:
        return np.nan, 0
    return float(np.linalg.norm(b[mask] - a[mask])), n_used


def _bin_taxon_stats_from_abundance(da_abund: xr.DataArray, bin_label) -> pd.DataFrame:
    """Compute per-taxon participant-wise abundance summaries for one bin."""
    da_abund = _validate_da(da_abund, name="bin_abundance_space")
    da_bin = da_abund.sel(bin=bin_label)
    rows = []
    for taxon in da_abund.coords["taxon"].values:
        vals = da_bin.sel(taxon=taxon).values.astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        n_obs = int(vals.size)
        abund_mean_bin = float(np.mean(vals)) if n_obs > 0 else np.nan
        abund_sd_bin = float(np.std(vals, ddof=1)) if n_obs >= 2 else np.nan
        if np.isfinite(abund_sd_bin) and abund_sd_bin > 0:
            abund_mean_over_sd_bin = float(abund_mean_bin / abund_sd_bin)
        else:
            abund_mean_over_sd_bin = np.nan
        rows.append(
            {
                "taxon": taxon,
                "n_obs": n_obs,
                "abund_mean_bin": abund_mean_bin,
                "abund_sd_bin": abund_sd_bin,
                "abund_mean_over_sd_bin": abund_mean_over_sd_bin,
            }
        )
    return pd.DataFrame(rows)


def compute_taxon_marginal_stats(
    da_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    dataset_role: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute participant-bin marginal abundance summaries for each taxon."""
    da_abund = _validate_da(da_abund, name="da_abund")
    da_abund = da_abund.transpose("participant", "taxon", "bin")
    rows = []
    for taxon in da_abund.coords["taxon"].values:
        vals = da_abund.sel(taxon=taxon).values.astype(float).ravel()
        vals = vals[np.isfinite(vals)]
        n_obs = int(vals.size)
        abund_mean = float(np.mean(vals)) if n_obs > 0 else np.nan
        abund_sd = float(np.std(vals, ddof=1)) if n_obs >= 2 else np.nan
        if np.isfinite(abund_sd) and abund_sd > 0:
            abund_mean_over_sd = float(abund_mean / abund_sd)
        else:
            abund_mean_over_sd = np.nan
        rows.append(
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": mask_seed,
                "algo_seed": algo_seed,
                "dataset_role": dataset_role,
                "taxon": taxon,
                "n_obs": n_obs,
                "abund_mean": abund_mean,
                "abund_sd": abund_sd,
                "abund_mean_over_sd": abund_mean_over_sd,
            }
        )
    return pd.DataFrame(rows)


def evaluate_taxon_marginal_preservation(
    df_taxon_stats: pd.DataFrame,
    frac: float,
    repeat: int,
    method: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate taxon-level marginal distribution preservation."""
    detail_cols = [
        "frac",
        "repeat",
        "method",
        "method_group",
        "mask_seed",
        "algo_seed",
        "taxon",
        "delta_mean_before",
        "delta_mean_after",
        "delta_sd_before",
        "delta_sd_after",
        "delta_mean_over_sd_before",
        "delta_mean_over_sd_after",
    ]
    method_cols = [
        "frac",
        "repeat",
        "method",
        "method_group",
        "mask_seed",
        "algo_seed",
        "scope",
        "n_taxa_used",
        "marginal_ratio_wd_before",
        "marginal_ratio_wd_after",
        "marginal_ratio_wd_improve",
        "marginal_avg_euclid_before",
        "marginal_avg_euclid_after",
        "marginal_avg_euclid_improve",
    ]
    if df_taxon_stats.empty:
        return _empty_df(detail_cols), _empty_df(method_cols)

    wide = _collect_aligned_role_stats(df_taxon_stats, ["taxon"], ["abund_mean", "abund_sd", "abund_mean_over_sd"])
    if wide.empty:
        return _empty_df(detail_cols), _empty_df(method_cols)

    detail = pd.DataFrame(
        {
            "frac": float(frac),
            "repeat": int(repeat),
            "method": method,
            "method_group": _method_group(method),
            "mask_seed": mask_seed,
            "algo_seed": algo_seed,
            "taxon": wide["taxon"],
            "delta_mean_before": wide["abund_mean_before"] - wide["abund_mean_truth"],
            "delta_mean_after": wide["abund_mean_after"] - wide["abund_mean_truth"],
            "delta_sd_before": wide["abund_sd_before"] - wide["abund_sd_truth"],
            "delta_sd_after": wide["abund_sd_after"] - wide["abund_sd_truth"],
            "delta_mean_over_sd_before": wide["abund_mean_over_sd_before"] - wide["abund_mean_over_sd_truth"],
            "delta_mean_over_sd_after": wide["abund_mean_over_sd_after"] - wide["abund_mean_over_sd_truth"],
        }
    )

    marginal_ratio_wd_before, n_ratio_before = _distribution_wd_from_series(
        wide["abund_mean_over_sd_truth"], wide["abund_mean_over_sd_before"]
    )
    marginal_ratio_wd_after, n_ratio_after = _distribution_wd_from_series(
        wide["abund_mean_over_sd_truth"], wide["abund_mean_over_sd_after"]
    )
    marginal_avg_euclid_before = _mean_euclid_2d(
        wide["abund_mean_truth"].to_numpy(dtype=float),
        wide["abund_sd_truth"].to_numpy(dtype=float),
        wide["abund_mean_before"].to_numpy(dtype=float),
        wide["abund_sd_before"].to_numpy(dtype=float),
    )
    marginal_avg_euclid_after = _mean_euclid_2d(
        wide["abund_mean_truth"].to_numpy(dtype=float),
        wide["abund_sd_truth"].to_numpy(dtype=float),
        wide["abund_mean_after"].to_numpy(dtype=float),
        wide["abund_sd_after"].to_numpy(dtype=float),
    )
    euclid_before_mask = (
        np.isfinite(wide["abund_mean_truth"].to_numpy(dtype=float))
        & np.isfinite(wide["abund_sd_truth"].to_numpy(dtype=float))
        & np.isfinite(wide["abund_mean_before"].to_numpy(dtype=float))
        & np.isfinite(wide["abund_sd_before"].to_numpy(dtype=float))
    )
    euclid_after_mask = (
        np.isfinite(wide["abund_mean_truth"].to_numpy(dtype=float))
        & np.isfinite(wide["abund_sd_truth"].to_numpy(dtype=float))
        & np.isfinite(wide["abund_mean_after"].to_numpy(dtype=float))
        & np.isfinite(wide["abund_sd_after"].to_numpy(dtype=float))
    )
    method_summary = pd.DataFrame(
        [
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": mask_seed,
                "algo_seed": algo_seed,
                "scope": "TaxonMarginal",
                "n_taxa_used": int(max(n_ratio_before, n_ratio_after, euclid_before_mask.sum(), euclid_after_mask.sum())),
                "marginal_ratio_wd_before": marginal_ratio_wd_before,
                "marginal_ratio_wd_after": marginal_ratio_wd_after,
                "marginal_ratio_wd_improve": marginal_ratio_wd_before - marginal_ratio_wd_after,
                "marginal_avg_euclid_before": marginal_avg_euclid_before,
                "marginal_avg_euclid_after": marginal_avg_euclid_after,
                "marginal_avg_euclid_improve": marginal_avg_euclid_before - marginal_avg_euclid_after,
            }
        ]
    )
    return detail.reindex(columns=detail_cols), method_summary.reindex(columns=method_cols)


def compute_taxon_bin_stats(
    da_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    dataset_role: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute per-bin abundance summaries for each taxon."""
    da_abund = _validate_da(da_abund, name="da_abund")
    da_abund = da_abund.transpose("participant", "taxon", "bin")
    rows = []
    for taxon in da_abund.coords["taxon"].values:
        for bin_label in da_abund.coords["bin"].values:
            vals = da_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            vals = vals[np.isfinite(vals)]
            n_obs = int(vals.size)
            abund_mean_bin = float(np.mean(vals)) if n_obs > 0 else np.nan
            abund_sd_bin = float(np.std(vals, ddof=1)) if n_obs >= 2 else np.nan
            if np.isfinite(abund_sd_bin) and abund_sd_bin > 0:
                abund_mean_over_sd_bin = float(abund_mean_bin / abund_sd_bin)
            else:
                abund_mean_over_sd_bin = np.nan
            rows.append(
                {
                    "frac": float(frac),
                    "repeat": int(repeat),
                    "method": method,
                    "method_group": _method_group(method),
                    "mask_seed": mask_seed,
                    "algo_seed": algo_seed,
                    "dataset_role": dataset_role,
                    "taxon": taxon,
                    "bin": bin_label,
                    "n_obs": n_obs,
                    "abund_mean_bin": abund_mean_bin,
                    "abund_sd_bin": abund_sd_bin,
                    "abund_mean_over_sd_bin": abund_mean_over_sd_bin,
                }
            )
    return pd.DataFrame(rows)


def evaluate_taxon_trajectory_preservation(
    df_taxon_bin_stats: pd.DataFrame,
    da_truth_abund: xr.DataArray,
    da_before_abund: xr.DataArray,
    da_after_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate per-taxon trajectory preservation across bins."""
    eval_cols = [
        "frac",
        "repeat",
        "method",
        "method_group",
        "mask_seed",
        "algo_seed",
        "taxon",
        "traj_mean_l2_before",
        "traj_mean_l2_after",
        "traj_sd_l2_before",
        "traj_sd_l2_after",
        "traj_mean_sd_joint_euclid_before",
        "traj_mean_sd_joint_euclid_after",
        "taxon_avg_bin_wd_before",
        "taxon_avg_bin_wd_after",
        "traj_mean_l2_improve",
        "traj_sd_l2_improve",
        "traj_mean_sd_joint_euclid_improve",
        "taxon_avg_bin_wd_improve",
    ]
    detail_cols = [
        "frac",
        "repeat",
        "method",
        "method_group",
        "mask_seed",
        "algo_seed",
        "taxon",
        "bin",
        "wd_before_bin",
        "wd_after_bin",
        "euclid_mean_sd_before_bin",
        "euclid_mean_sd_after_bin",
    ]
    method_cols = [
        "frac",
        "repeat",
        "method",
        "method_group",
        "mask_seed",
        "algo_seed",
        "scope",
        "n_taxa_used",
        "traj_mean_l2_before",
        "traj_mean_l2_after",
        "traj_sd_l2_before",
        "traj_sd_l2_after",
        "traj_mean_sd_joint_euclid_before",
        "traj_mean_sd_joint_euclid_after",
        "taxon_avg_bin_wd_before",
        "taxon_avg_bin_wd_after",
        "traj_mean_l2_improve",
        "traj_sd_l2_improve",
        "traj_mean_sd_joint_euclid_improve",
        "taxon_avg_bin_wd_improve",
    ]
    if df_taxon_bin_stats.empty:
        return _empty_df(eval_cols), _empty_df(detail_cols), _empty_df(method_cols)

    bin_order = list(da_truth_abund.coords["bin"].values)
    wide = _collect_aligned_role_stats(
        df_taxon_bin_stats,
        ["taxon", "bin"],
        ["abund_mean_bin", "abund_sd_bin", "abund_mean_over_sd_bin"],
    )
    detail_rows = []
    eval_rows = []
    for taxon in da_truth_abund.coords["taxon"].values:
        taxon_wide = wide.loc[wide["taxon"].eq(taxon)].copy()
        if taxon_wide.empty:
            continue
        taxon_wide["bin_order"] = taxon_wide["bin"].map({b: i for i, b in enumerate(bin_order)})
        taxon_wide = taxon_wide.sort_values("bin_order")

        mean_before, _ = _mean_l2_distance(
            taxon_wide["abund_mean_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_mean_bin_before"].to_numpy(dtype=float),
        )
        mean_after, _ = _mean_l2_distance(
            taxon_wide["abund_mean_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_mean_bin_after"].to_numpy(dtype=float),
        )
        sd_before, _ = _mean_l2_distance(
            taxon_wide["abund_sd_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_sd_bin_before"].to_numpy(dtype=float),
        )
        sd_after, _ = _mean_l2_distance(
            taxon_wide["abund_sd_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_sd_bin_after"].to_numpy(dtype=float),
        )
        joint_before = _mean_euclid_2d(
            taxon_wide["abund_mean_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_sd_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_mean_bin_before"].to_numpy(dtype=float),
            taxon_wide["abund_sd_bin_before"].to_numpy(dtype=float),
        )
        joint_after = _mean_euclid_2d(
            taxon_wide["abund_mean_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_sd_bin_truth"].to_numpy(dtype=float),
            taxon_wide["abund_mean_bin_after"].to_numpy(dtype=float),
            taxon_wide["abund_sd_bin_after"].to_numpy(dtype=float),
        )

        wd_before_vals = []
        wd_after_vals = []
        for bin_label in bin_order:
            truth_vec = da_truth_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            before_vec = da_before_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            after_vec = da_after_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            truth_vec = truth_vec[np.isfinite(truth_vec)]
            before_vec = before_vec[np.isfinite(before_vec)]
            after_vec = after_vec[np.isfinite(after_vec)]

            wd_before_bin = _safe_wasserstein(truth_vec, before_vec)
            wd_after_bin = _safe_wasserstein(truth_vec, after_vec)
            wd_before_vals.append(wd_before_bin)
            wd_after_vals.append(wd_after_bin)

            stat_row = taxon_wide.loc[taxon_wide["bin"].eq(bin_label)]
            if stat_row.empty:
                euclid_before_bin = np.nan
                euclid_after_bin = np.nan
            else:
                euclid_before_bin = _mean_euclid_2d(
                    stat_row["abund_mean_bin_truth"].to_numpy(dtype=float),
                    stat_row["abund_sd_bin_truth"].to_numpy(dtype=float),
                    stat_row["abund_mean_bin_before"].to_numpy(dtype=float),
                    stat_row["abund_sd_bin_before"].to_numpy(dtype=float),
                )
                euclid_after_bin = _mean_euclid_2d(
                    stat_row["abund_mean_bin_truth"].to_numpy(dtype=float),
                    stat_row["abund_sd_bin_truth"].to_numpy(dtype=float),
                    stat_row["abund_mean_bin_after"].to_numpy(dtype=float),
                    stat_row["abund_sd_bin_after"].to_numpy(dtype=float),
                )
            detail_rows.append(
                {
                    "frac": float(frac),
                    "repeat": int(repeat),
                    "method": method,
                    "method_group": _method_group(method),
                    "mask_seed": mask_seed,
                    "algo_seed": algo_seed,
                    "taxon": taxon,
                    "bin": bin_label,
                    "wd_before_bin": wd_before_bin,
                    "wd_after_bin": wd_after_bin,
                    "euclid_mean_sd_before_bin": euclid_before_bin,
                    "euclid_mean_sd_after_bin": euclid_after_bin,
                }
            )

        wd_before_arr = np.asarray(wd_before_vals, dtype=float)
        wd_after_arr = np.asarray(wd_after_vals, dtype=float)
        taxon_avg_bin_wd_before = float(np.mean(wd_before_arr[np.isfinite(wd_before_arr)])) if np.any(
            np.isfinite(wd_before_arr)
        ) else np.nan
        taxon_avg_bin_wd_after = float(np.mean(wd_after_arr[np.isfinite(wd_after_arr)])) if np.any(
            np.isfinite(wd_after_arr)
        ) else np.nan
        eval_rows.append(
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": mask_seed,
                "algo_seed": algo_seed,
                "taxon": taxon,
                "traj_mean_l2_before": mean_before,
                "traj_mean_l2_after": mean_after,
                "traj_sd_l2_before": sd_before,
                "traj_sd_l2_after": sd_after,
                "traj_mean_sd_joint_euclid_before": joint_before,
                "traj_mean_sd_joint_euclid_after": joint_after,
                "taxon_avg_bin_wd_before": taxon_avg_bin_wd_before,
                "taxon_avg_bin_wd_after": taxon_avg_bin_wd_after,
                "traj_mean_l2_improve": mean_before - mean_after,
                "traj_sd_l2_improve": sd_before - sd_after,
                "traj_mean_sd_joint_euclid_improve": joint_before - joint_after,
                "taxon_avg_bin_wd_improve": taxon_avg_bin_wd_before - taxon_avg_bin_wd_after,
            }
        )

    df_eval = pd.DataFrame(eval_rows).reindex(columns=eval_cols)
    df_detail = pd.DataFrame(detail_rows).reindex(columns=detail_cols)
    if df_eval.empty:
        return df_eval, df_detail, _empty_df(method_cols)

    metric_cols = [
        c
        for c in eval_cols
        if c not in {"frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "taxon"}
    ]
    method_row = {
        "frac": float(frac),
        "repeat": int(repeat),
        "method": method,
        "method_group": _method_group(method),
        "mask_seed": mask_seed,
        "algo_seed": algo_seed,
        "scope": "TaxonTrajectory",
        "n_taxa_used": int(df_eval[metric_cols].notna().any(axis=1).sum()),
    }
    for col in metric_cols:
        vals = pd.to_numeric(df_eval[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        method_row[col] = float(np.mean(vals)) if vals.size > 0 else np.nan
    df_method = pd.DataFrame([method_row]).reindex(columns=method_cols)
    return df_eval, df_detail, df_method


def evaluate_bin_distribution_preservation(
    da_truth_abund: xr.DataArray,
    da_before_abund: xr.DataArray,
    da_after_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate bin-wise abundance structure preservation across taxa."""
    columns = [
        "frac",
        "repeat",
        "method",
        "method_group",
        "mask_seed",
        "algo_seed",
        "bin",
        "n_taxa_used",
        "bin_ratio_wd_before",
        "bin_ratio_wd_after",
        "bin_ratio_wd_improve",
        "bin_avg_euclid_before",
        "bin_avg_euclid_after",
        "bin_avg_euclid_improve",
    ]
    da_truth_abund = _validate_da(da_truth_abund, name="da_truth_abund")
    da_before_abund = _validate_da(da_before_abund, name="da_before_abund")
    da_after_abund = _validate_da(da_after_abund, name="da_after_abund")

    rows = []
    for bin_label in da_truth_abund.coords["bin"].values:
        truth_stats = _bin_taxon_stats_from_abundance(da_truth_abund, bin_label)
        before_stats = _bin_taxon_stats_from_abundance(da_before_abund, bin_label)
        after_stats = _bin_taxon_stats_from_abundance(da_after_abund, bin_label)
        wide = truth_stats.rename(
            columns={
                "n_obs": "n_obs_truth",
                "abund_mean_bin": "abund_mean_bin_truth",
                "abund_sd_bin": "abund_sd_bin_truth",
                "abund_mean_over_sd_bin": "abund_mean_over_sd_bin_truth",
            }
        )
        wide = wide.merge(
            before_stats.rename(
                columns={
                    "n_obs": "n_obs_before",
                    "abund_mean_bin": "abund_mean_bin_before",
                    "abund_sd_bin": "abund_sd_bin_before",
                    "abund_mean_over_sd_bin": "abund_mean_over_sd_bin_before",
                }
            ),
            on="taxon",
            how="outer",
        )
        wide = wide.merge(
            after_stats.rename(
                columns={
                    "n_obs": "n_obs_after",
                    "abund_mean_bin": "abund_mean_bin_after",
                    "abund_sd_bin": "abund_sd_bin_after",
                    "abund_mean_over_sd_bin": "abund_mean_over_sd_bin_after",
                }
            ),
            on="taxon",
            how="outer",
        )
        bin_ratio_wd_before, n_ratio_before = _distribution_wd_from_series(
            wide["abund_mean_over_sd_bin_truth"], wide["abund_mean_over_sd_bin_before"]
        )
        bin_ratio_wd_after, n_ratio_after = _distribution_wd_from_series(
            wide["abund_mean_over_sd_bin_truth"], wide["abund_mean_over_sd_bin_after"]
        )
        bin_avg_euclid_before = _mean_euclid_2d(
            wide["abund_mean_bin_truth"].to_numpy(dtype=float),
            wide["abund_sd_bin_truth"].to_numpy(dtype=float),
            wide["abund_mean_bin_before"].to_numpy(dtype=float),
            wide["abund_sd_bin_before"].to_numpy(dtype=float),
        )
        bin_avg_euclid_after = _mean_euclid_2d(
            wide["abund_mean_bin_truth"].to_numpy(dtype=float),
            wide["abund_sd_bin_truth"].to_numpy(dtype=float),
            wide["abund_mean_bin_after"].to_numpy(dtype=float),
            wide["abund_sd_bin_after"].to_numpy(dtype=float),
        )
        euclid_before_mask = (
            np.isfinite(wide["abund_mean_bin_truth"].to_numpy(dtype=float))
            & np.isfinite(wide["abund_sd_bin_truth"].to_numpy(dtype=float))
            & np.isfinite(wide["abund_mean_bin_before"].to_numpy(dtype=float))
            & np.isfinite(wide["abund_sd_bin_before"].to_numpy(dtype=float))
        )
        euclid_after_mask = (
            np.isfinite(wide["abund_mean_bin_truth"].to_numpy(dtype=float))
            & np.isfinite(wide["abund_sd_bin_truth"].to_numpy(dtype=float))
            & np.isfinite(wide["abund_mean_bin_after"].to_numpy(dtype=float))
            & np.isfinite(wide["abund_sd_bin_after"].to_numpy(dtype=float))
        )
        rows.append(
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": mask_seed,
                "algo_seed": algo_seed,
                "bin": bin_label,
                "n_taxa_used": int(max(n_ratio_before, n_ratio_after, euclid_before_mask.sum(), euclid_after_mask.sum())),
                "bin_ratio_wd_before": bin_ratio_wd_before,
                "bin_ratio_wd_after": bin_ratio_wd_after,
                "bin_ratio_wd_improve": bin_ratio_wd_before - bin_ratio_wd_after,
                "bin_avg_euclid_before": bin_avg_euclid_before,
                "bin_avg_euclid_after": bin_avg_euclid_after,
                "bin_avg_euclid_improve": bin_avg_euclid_before - bin_avg_euclid_after,
            }
        )
    return pd.DataFrame(rows).reindex(columns=columns)


def build_abundance_eval_repeat_summary(
    df_taxon_marginal_method: pd.DataFrame,
    df_taxon_traj_method: pd.DataFrame,
    df_bin_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Build a repeat-level long summary for abundance-space evaluation."""
    long_tables = []
    for df_method in [df_taxon_marginal_method, df_taxon_traj_method]:
        if df_method is None or df_method.empty:
            continue
        id_cols = ["frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "scope"]
        value_cols = [c for c in df_method.columns if c not in id_cols]
        long_tables.append(
            df_method[id_cols + value_cols].melt(
                id_vars=id_cols,
                value_vars=value_cols,
                var_name="metric",
                value_name="value",
            )
        )

    if df_bin_distribution is not None and not df_bin_distribution.empty:
        grouped = (
            df_bin_distribution.groupby(
                ["frac", "repeat", "method", "method_group", "mask_seed", "algo_seed"],
                as_index=False,
            )[
                [
                    "n_taxa_used",
                    "bin_ratio_wd_before",
                    "bin_ratio_wd_after",
                    "bin_ratio_wd_improve",
                    "bin_avg_euclid_before",
                    "bin_avg_euclid_after",
                    "bin_avg_euclid_improve",
                ]
            ]
            .mean(numeric_only=True)
        )
        grouped["scope"] = "BinDistribution"
        long_tables.append(
            grouped.melt(
                id_vars=["frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "scope"],
                value_vars=[
                    "n_taxa_used",
                    "bin_ratio_wd_before",
                    "bin_ratio_wd_after",
                    "bin_ratio_wd_improve",
                    "bin_avg_euclid_before",
                    "bin_avg_euclid_after",
                    "bin_avg_euclid_improve",
                ],
                var_name="metric",
                value_name="value",
            )
        )

    if not long_tables:
        return _empty_df(
            ["frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "scope", "metric", "value"]
        )
    out = pd.concat(long_tables, ignore_index=True)
    return out.sort_values(["frac", "repeat", "scope", "method", "metric"]).reset_index(drop=True)


def summarize_abundance_eval_across_repeats(df_repeat_abund: pd.DataFrame) -> pd.DataFrame:
    """Summarize abundance-space metrics across repeats."""
    columns = ["frac", "scope", "method", "metric", "n", "mean", "sd", "sem", "ci95_low", "ci95_high"]
    if df_repeat_abund is None or df_repeat_abund.empty:
        return _empty_df(columns)
    rows = []
    for keys, g in df_repeat_abund.groupby(["frac", "scope", "method", "metric"], dropna=False):
        summary = _summary_real(g["value"])
        row = {"frac": keys[0], "scope": keys[1], "method": keys[2], "metric": keys[3]}
        row.update(summary.to_dict())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.reindex(columns=columns).sort_values(["frac", "scope", "method", "metric"]).reset_index(drop=True)


def merge_abundance_eval_outputs(outdir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge repeat-level abundance-space outputs and write combined summaries."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _read_many(pattern: str) -> pd.DataFrame:
        paths = sorted(outdir.glob(pattern))
        tables = []
        for path in paths:
            try:
                tables.append(pd.read_csv(path))
            except Exception:
                continue
        if not tables:
            return pd.DataFrame()
        df = pd.concat(tables, ignore_index=True)
        return df.drop_duplicates().reset_index(drop=True)

    df_taxon_marginal = _read_many("taxon_marginal_eval_frac_*_repeat_*.csv")
    df_taxon_trajectory = _read_many("taxon_trajectory_eval_frac_*_repeat_*.csv")
    df_bin_distribution = _read_many("bin_distribution_eval_frac_*_repeat_*.csv")
    df_abundance_repeat = _read_many("abundance_eval_summary_frac_*_repeat_*.csv")
    df_abundance_errorbar = summarize_abundance_eval_across_repeats(df_abundance_repeat)

    df_taxon_marginal.to_csv(outdir / "taxon_marginal_eval_all.csv", index=False)
    df_taxon_trajectory.to_csv(outdir / "taxon_trajectory_eval_all.csv", index=False)
    df_bin_distribution.to_csv(outdir / "bin_distribution_eval_all.csv", index=False)
    df_abundance_repeat.to_csv(outdir / "abundance_repeat_level_metrics.csv", index=False)
    df_abundance_errorbar.to_csv(outdir / "abundance_errorbar_summary.csv", index=False)
    return df_taxon_marginal, df_taxon_trajectory, df_bin_distribution, df_abundance_repeat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sim1-only repeated imputation experiment with explicit initial-missing mask."
    )
    parser.add_argument("--tax-path", type=str, default="hmp2_mgx_taxonomy.tsv")
    parser.add_argument("--meta-path", type=str, default="metagenomics.xlsx")
    parser.add_argument("--outdir", type=str, default="results_sim1")
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--seed-base", type=int, default=20260322)

    parser.add_argument(
        "--truth-observed-mode",
        type=str,
        default="positive_only",
        choices=["positive_only", "positive_only_except_replace", "all_finite_except_replace", "all_finite"],
        help="Observed set used in the truth fitting stage before building the full p2d2 reconstruction truth.",
    )
    parser.add_argument(
        "--second-observed-mode",
        type=str,
        default="positive_only_except_replace",
        choices=["positive_only", "positive_only_except_replace", "all_finite_except_replace", "all_finite"],
        help="Observed set used in the second-stage imputation after random masking.",
    )
    parser.add_argument(
        "--mask-eligible-mode",
        type=str,
        default="all_finite",
        choices=["positive_only", "all_finite"],
        help="Which cells are eligible for random masking in da_truth: all_finite masks any finite cell in the full truth, positive_only masks only positive-valued cells.",
    )

    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--t0", type=int, default=30)
    parser.add_argument("--t", type=int, default=30)
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--temporal-knn-k", type=int, default=3)
    parser.add_argument("--temporal-knn-bandwidth", type=float, default=4.0)
    parser.add_argument("--spline-min-points", type=int, default=4)
    parser.add_argument("--equal-value", type=float, default=0.0)
    parser.add_argument("--dbn-dense-alpha", type=float, default=1e-3)
    parser.add_argument("--dbn-sparse-alpha", type=float, default=1e-2)
    parser.add_argument("--glv-alpha", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_sim1_pipeline(args)


if __name__ == "__main__":
    main()
