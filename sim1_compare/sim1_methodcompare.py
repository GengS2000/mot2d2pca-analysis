import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Lasso, LinearRegression, Ridge

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from competitor_methods import (  # noqa: E402
    COMPETITOR_GROUPS,
    available_requested_methods,
    impute_competitor,
)


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
    "ctf",
    "mean",
    "nearest_positive",
    "temporal_knn",
]

SIM1_SUPPORTED_METHODS = frozenset(DEFAULT_METHODS)

METHOD_GROUP_MAP = {
    "penalized_2d2_pca": "matrix_factorization",
    "mean": "naive",
    "nearest_positive": "local_time",
    "temporal_knn": "local_time",
    "ctf": COMPETITOR_GROUPS["ctf"],
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
    floor_value: Optional[float] = None,
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

    if floor_value is None:
        finite_values = a[np.isfinite(a)]
        if finite_values.size == 0:
            raise ValueError("Cannot infer floor_value from an entirely non-finite tensor")
        floor_value = float(np.min(finite_values))
    floor_value = float(floor_value)

    # Abar is a log-scale baseline.  An unsupported trajectory therefore uses
    # log10(eps), not 0 (which would mean 1 percent on this data scale).
    obs_count = obs.sum(axis=0)
    obs_sum = np.where(obs, a, 0.0).sum(axis=0)
    abar = np.full((m, n), floor_value, dtype=float)
    np.divide(obs_sum, obs_count, out=abar, where=obs_count > 0)
    # Missing *centered residuals* are correctly zero after the baseline above
    # has been defined; this is distinct from setting Abar itself to zero.
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
    taxon_penalty_weight = 0.0
    participant_penalty_weight = 0.0
    taxon_penalty_ratio = np.nan
    participant_penalty_ratio = np.nan
    for _ in range(t):
        zzt = z @ z.T
        astar = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astar += bi.T @ zzt @ bi
        astar /= m_time

        pen = np.zeros((n, n), dtype=float)
        taxon_penalty_weight = 0.0
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
                taxon_penalty_weight += float(wij)
        if taxon_penalty_weight > 0.0:
            pen /= taxon_penalty_weight
        taxon_penalty_ratio = float(
            lam * np.linalg.norm(pen, ord="fro") / max(np.linalg.norm(astar, ord="fro"), 1e-12)
        )
        astar = astar - lam * pen
        x = _top_eigvecs(astar, d)

        xxt = x @ x.T
        astarstar = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar += bi @ xxt @ bi.T
        astarstar /= m_time

        pen2 = np.zeros((m, m), dtype=float)
        participant_penalty_weight = 0.0
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
                participant_penalty_weight += float(wij)
        if participant_penalty_weight > 0.0:
            pen2 /= participant_penalty_weight
        participant_penalty_ratio = float(
            lam * np.linalg.norm(pen2, ord="fro") / max(np.linalg.norm(astarstar, ord="fro"), 1e-12)
        )
        astarstar = astarstar - lam * pen2
        z = _top_eigvecs(astarstar, q)

    c = np.zeros((m_time, q, d), dtype=float)
    for i in range(m_time):
        ai_eff = np.where(obs[i], a[i], abar)
        c[i] = z.T @ (ai_eff - abar) @ x

    meta = {
        "bin": time,
        "t": tvals,
        "W": w_time,
        "Abar": abar,
        "floor_value": floor_value,
        "unsupported_trajectory_count": int(np.sum(obs_count == 0)),
        "taxon_penalty_weight": float(taxon_penalty_weight),
        "participant_penalty_weight": float(participant_penalty_weight),
        "taxon_penalty_ratio": float(taxon_penalty_ratio),
        "participant_penalty_ratio": float(participant_penalty_ratio),
    }
    if verbose:
        print(
            "[P2D2 diagnostics] "
            f"unsupported_trajectories={meta['unsupported_trajectory_count']}, "
            f"taxon_penalty_ratio={taxon_penalty_ratio:.6g}, "
            f"participant_penalty_ratio={participant_penalty_ratio:.6g}"
        )
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

    floor_value = meta.get("floor_value")
    if floor_value is not None:
        full_vals = np.maximum(full_vals, float(floor_value))

    da_full = template_da.copy(deep=True)
    da_full.values[:] = full_vals
    return da_full


def build_full_p2d2_truth(
    da_base: xr.DataArray,
    zero_value: float,
    pca_params: Dict,
    model_seed: int,
    truth_observed_mode: str = "positive_only",
    hybrid_observed: bool = False,
    verbose: bool = False,
) -> xr.DataArray:
    """
    Fit penalized_2d2_pca on the truth-stage observed set and return the full tensor reconstruction.

    With hybrid_observed=True the observed cells keep their raw base values and only
    originally-unobserved cells are filled with the fitted reconstruction.  The frozen
    truth therefore preserves the genuine temporal dynamics of the data where it was
    observed, while the smooth fit covers the 90%+ of cells with no observation.
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
        floor_value=zero_value,
        verbose=verbose,
    )
    da_truth = reconstruct_full_tensor_from_p2d2_fit(
        x=x,
        z=z,
        c=c,
        meta=meta,
        template_da=da_base,
    )
    if hybrid_observed:
        truth_vals = da_truth.values.copy()
        truth_vals = np.where(observed_mask.values, da_base.values, truth_vals)
        da_truth.values = truth_vals
    da_truth.name = "da_truth_sim1"
    da_truth.attrs.update(
        {
            "generator": "penalized_2d2_pca",
            "d": int(pca_params["d"]),
            "q": int(pca_params["q"]),
            "lambda_normalized": float(pca_params["lam"]),
            "sigma": float(pca_params["sigma"]),
            "floor_value": float(zero_value),
            "unsupported_trajectory_count": int(meta["unsupported_trajectory_count"]),
            "taxon_penalty_ratio": float(meta["taxon_penalty_ratio"]),
            "participant_penalty_ratio": float(meta["participant_penalty_ratio"]),
            "truth_hybrid": int(bool(hybrid_observed)),
        }
    )
    return da_truth


def summarize_truth_dynamics(
    da_truth: xr.DataArray,
    eps: float,
    observed_mask: Optional[xr.DataArray] = None,
) -> Dict[str, float]:
    """Return compact diagnostics used to freeze a Sim1 truth before benchmarking.

    observed_mask marks the cells that were actually observed in the base data.  When
    provided, extra observed-only diagnostics are added: for hybrid truths the full-tensor
    time variance fraction is diluted by the smooth fill of the ~93% unobserved cells, so
    the observed-only fraction and within-trajectory sd report the genuine dynamics.
    """
    values = da_truth.values.astype(float)
    centered = values - np.mean(values, axis=2, keepdims=True)
    total_ss = float(np.sum((values - np.mean(values)) ** 2))
    time_ss = float(np.sum(centered**2))
    matrix = centered.reshape(-1, centered.shape[2])
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_energy = singular_values**2
    energy_total = float(np.sum(singular_energy))
    shares = singular_energy / energy_total if energy_total > 0.0 else np.zeros_like(singular_energy)
    flat = values.reshape(-1, values.shape[2])
    correlation = np.corrcoef(flat, rowvar=False)
    off_diagonal = correlation[np.triu_indices(values.shape[2], k=1)]
    abundance = np.maximum(np.power(10.0, values) - float(eps), 0.0)
    composition_sum = np.sum(abundance, axis=1)
    floor_value = float(np.log10(eps))
    out = {
        "log_time_variance_fraction": time_ss / total_ss if total_ss > 0.0 else np.nan,
        "temporal_rank1_share": float(shares[0]) if shares.size else np.nan,
        "temporal_rank3_cumulative_share": float(np.sum(shares[:3])),
        "effective_temporal_rank": int(np.sum(shares > 1e-8)),
        "cross_bin_correlation_min": float(np.nanmin(off_diagonal)),
        "cross_bin_correlation_median": float(np.nanmedian(off_diagonal)),
        "cross_bin_correlation_max": float(np.nanmax(off_diagonal)),
        "floor_fraction": float(np.mean(values == floor_value)),
        "composition_sum_mean": float(np.mean(composition_sum)),
        "composition_sum_min": float(np.min(composition_sum)),
        "composition_sum_max": float(np.max(composition_sum)),
    }
    if observed_mask is not None:
        obs = observed_mask.values.astype(bool)
        obs_vals = np.where(obs, values, np.nan)
        traj_mean = np.nanmean(obs_vals, axis=2, keepdims=True)
        ctr_obs = np.where(obs, values - traj_mean, 0.0)
        time_ss_obs = float(np.sum(ctr_obs**2))
        obs_mean = float(np.nanmean(obs_vals))
        tot_obs = float(np.sum(np.where(obs, (values - obs_mean) ** 2, 0.0)))
        traj_sd = np.sqrt(np.nanmean(np.where(obs, ctr_obs**2, np.nan), axis=2))
        out["log_time_variance_fraction_observed_only"] = (
            time_ss_obs / tot_obs if tot_obs > 0.0 else np.nan
        )
        out["observed_trajectory_sd_mean"] = float(np.nanmean(traj_sd))
        out["observed_fraction"] = float(obs.mean())
    return out


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


def _fill_nearest_positive(
    values: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    idx: int,
    fallback_value: float,
) -> float:
    """Copy the closest original positive observation; ties go to the earlier bin."""
    donor_idx = np.flatnonzero(observed)
    if donor_idx.size == 0:
        return float(fallback_value)
    distance = np.abs(times[donor_idx] - times[idx])
    min_distance = float(np.min(distance))
    tied = donor_idx[np.isclose(distance, min_distance, rtol=0.0, atol=1e-12)]
    chosen = int(tied[np.argmin(times[tied])])
    return float(values[chosen])


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
    return _fill_nearest_positive(values, observed, times, idx, mean_value)


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
    return _fill_nearest_positive(values, observed, times, idx, mean_value)


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
    fallback_values: Optional[np.ndarray] = None,
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
        fallback_value = (
            float(fallback_values[idx])
            if fallback_values is not None and np.isfinite(fallback_values[idx])
            else float(zero_value)
        )
        if method == "mean":
            out[idx] = mean_value if obs_idx.size > 0 else fallback_value
        elif method == "median":
            out[idx] = median_value
        elif method == "nearest_positive":
            out[idx] = _fill_nearest_positive(values, observed, times, idx, fallback_value)
        elif method == "weighted_time_average":
            out[idx] = _fill_weighted_time_average(values, observed, times, idx, mean_value)
        elif method == "temporal_knn":
            out[idx] = _fill_temporal_knn(
                values,
                observed,
                times,
                idx,
                fallback_value,
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


def _build_taxon_positive_fallbacks(
    values: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    zero_value: float,
) -> np.ndarray:
    """Build observed-only taxon fallbacks at the closest supported time bin."""
    n_taxa = values.shape[1]
    n_time = values.shape[2]
    fallback = np.full((n_taxa, n_time), float(zero_value), dtype=float)
    for taxon_idx in range(n_taxa):
        supported_times = np.flatnonzero(np.any(observed[:, taxon_idx, :], axis=0))
        if supported_times.size == 0:
            continue
        for target_idx in range(n_time):
            distance = np.abs(times[supported_times] - times[target_idx])
            min_distance = float(np.min(distance))
            tied = supported_times[np.isclose(distance, min_distance, rtol=0.0, atol=1e-12)]
            donor_time = int(tied[np.argmin(times[tied])])
            donor_values = values[:, taxon_idx, donor_time][observed[:, taxon_idx, donor_time]]
            fallback[taxon_idx, target_idx] = float(np.median(donor_values))
    return fallback


def _write_back_only_masked_finite(
    da_input: xr.DataArray,
    candidate_da: xr.DataArray,
    replace_mask: xr.DataArray,
) -> xr.DataArray:
    """
    For the random cell-mask task, update only `replace_mask=True` cells with finite candidates.

    Legacy comparator methods may build a full day-vector or full candidate tensor internally, but
    this helper guarantees that only the masked cells are overwritten in the returned DataArray.
    """
    base = _validate_da(da_input, name="da_input").copy(deep=True)
    candidate = _validate_da(candidate_da, name="candidate_da")
    replace = _validate_da(replace_mask, name="replace_mask").values.astype(bool)
    cand_vals = candidate.values.astype(float)
    out_vals = base.values.astype(float).copy()
    valid = replace & np.isfinite(cand_vals)
    out_vals[valid] = cand_vals[valid]
    base.values[:] = out_vals
    return base


def _abundance_to_relative_abundance(da_abund: xr.DataArray, name: str = "relative_abundance") -> xr.DataArray:
    """Normalize abundance-space values along `taxon` so each participant-bin composition sums to 1."""
    da_abund = _validate_da(da_abund, name="da_abund")
    vals = da_abund.values.astype(float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.where(vals < 0.0, 0.0, vals)
    denom = vals.sum(axis=1, keepdims=True)
    rel = np.divide(vals, denom, out=np.full_like(vals, np.nan), where=denom > 0.0)
    return xr.DataArray(rel, coords=da_abund.coords, dims=da_abund.dims, name=name, attrs=dict(da_abund.attrs))


def _relative_abundance_to_log_scale(
    da_rel: xr.DataArray,
    eps: float = 1e-3,
    name: str = "legacy_log_candidate",
) -> xr.DataArray:
    """Map relative abundance values back to the script's `log10(value + eps)` scale."""
    da_rel = _validate_da(da_rel, name="da_rel")
    vals = da_rel.values.astype(float)
    vals = np.where(vals < 0.0, 0.0, vals)
    vals = np.log10(vals + float(eps))
    return xr.DataArray(vals, coords=da_rel.coords, dims=da_rel.dims, name=name, attrs=dict(da_rel.attrs))


def _mean_initialize_relative_abundance(
    da_input: xr.DataArray,
    observed_mask: xr.DataArray,
    eps: float = 1e-3,
) -> xr.DataArray:
    """
    Build a complete relative-abundance tensor for legacy dynamic comparators using mean initialization.

    This initialization is internal only. The outer task remains random cell masking, and only masked
    cells are written back after a legacy method generates candidate vectors.
    """
    values = _validate_da(da_input, name="da_input").values.astype(float)
    observed = _validate_da(observed_mask, name="observed_mask").values.astype(bool)
    abund = np.power(10.0, values) - float(eps)
    abund = np.where(abund < 0.0, 0.0, abund)
    initialized = abund.copy()

    n_participants, n_taxa, n_bins = initialized.shape
    for i in range(n_participants):
        for j in range(n_taxa):
            obs = observed[i, j, :]
            traj = initialized[i, j, :]
            obs_vals = traj[obs & np.isfinite(traj)]
            fill_value = float(np.mean(obs_vals)) if obs_vals.size > 0 else 0.0
            fill_mask = ~obs | ~np.isfinite(traj)
            initialized[i, j, fill_mask] = fill_value
        for b in range(n_bins):
            col = initialized[i, :, b]
            col = np.where(np.isfinite(col), col, 0.0)
            col = np.where(col < 0.0, 0.0, col)
            total = float(np.sum(col))
            if total > 0.0:
                initialized[i, :, b] = col / total
            else:
                initialized[i, :, b] = np.full(n_taxa, 1.0 / max(n_taxa, 1), dtype=float)

    return xr.DataArray(
        initialized,
        coords=da_input.coords,
        dims=da_input.dims,
        name="legacy_relative_abundance_initialized",
    )


def _legacy_weighted_avg_vector(matrix: np.ndarray, times: np.ndarray, idx: int) -> np.ndarray:
    prev_idx = idx - 1
    next_idx = idx + 1
    if prev_idx < 0 or next_idx >= matrix.shape[1]:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    prev_time = times[prev_idx]
    cur_time = times[idx]
    next_time = times[next_idx]
    denom = next_time - prev_time
    if not np.isfinite(denom) or denom == 0.0:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    prev_vec = matrix[:, prev_idx]
    next_vec = matrix[:, next_idx]
    if not (np.all(np.isfinite(prev_vec)) and np.all(np.isfinite(next_vec))):
        return np.full(matrix.shape[0], np.nan, dtype=float)
    weighted = (next_vec * (cur_time - prev_time) + prev_vec * (next_time - cur_time)) / denom
    return weighted.astype(float)


def _legacy_spline_vector(matrix: np.ndarray, times: np.ndarray, idx: int) -> np.ndarray:
    out = np.full(matrix.shape[0], np.nan, dtype=float)
    target_time = times[idx]
    for spec in range(matrix.shape[0]):
        y = matrix[spec, :]
        finite = np.isfinite(y)
        if int(finite.sum()) < 4:
            continue
        uniq_times = np.unique(times[finite])
        if uniq_times.size < 4:
            continue
        try:
            model = interp1d(times[finite], y[finite], kind="cubic", bounds_error=True)
            pred = float(model(target_time))
        except Exception:
            pred = np.nan
        if np.isfinite(pred):
            out[spec] = max(0.0, pred)
    return out


def _legacy_knn_epanechnikov_vector(matrix: np.ndarray, times: np.ndarray, idx: int, k: int) -> np.ndarray:
    if matrix.shape[1] < 2:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    dists = np.abs(times - times[idx])
    nonzero = np.sort(dists[dists > 0.0])
    if nonzero.size == 0:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    bw_index = min(max(int(k), 1), nonzero.size) - 1
    bandwidth = float(nonzero[bw_index])
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        return np.full(matrix.shape[0], np.nan, dtype=float)

    candidate = np.zeros(matrix.shape[0], dtype=float)
    total_weight = 0.0
    for col_idx in range(matrix.shape[1]):
        if col_idx == idx:
            continue
        dist = abs(times[col_idx] - times[idx])
        if dist > bandwidth:
            continue
        vec = matrix[:, col_idx]
        if not np.all(np.isfinite(vec)):
            continue
        ker_norm = dist / bandwidth
        weight = 0.75 * (1.0 - ker_norm**2)
        if weight <= 0.0:
            continue
        candidate += weight * vec
        total_weight += weight

    if total_weight <= 0.0:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    candidate /= total_weight
    total = float(np.sum(candidate))
    if total > 0.0 and np.isfinite(total):
        candidate /= total
    else:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    return candidate


def _apply_legacy_day_vector_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    method: str,
    method_params: Dict,
    eps: float = 1e-3,
) -> xr.DataArray:
    """
    Apply a legacy local-time comparator under the random cell-mask task.

    The legacy method may construct a full participant-day candidate vector, but the final output only
    writes back finite values at `replace_mask=True` positions.
    """
    rel = _abundance_to_relative_abundance(to_abundance_space(da_input, eps=eps, name="legacy_local_abundance"))
    replace = _validate_da(replace_mask, name="replace_mask").values.astype(bool)
    times = np.array([_bin_to_time_value(x) for x in da_input.coords["bin"].values], dtype=float)
    candidate_vals = np.full(rel.shape, np.nan, dtype=float)

    for p in range(rel.shape[0]):
        matrix = rel.values[p, :, :].astype(float)
        for b in range(matrix.shape[1]):
            if not np.any(replace[p, :, b]):
                continue
            if method == "weighted_avg_legacy":
                candidate = _legacy_weighted_avg_vector(matrix, times, b)
            elif method == "spline_legacy":
                candidate = _legacy_spline_vector(matrix, times, b)
            elif method == "knn_epanechnikov_legacy":
                candidate = _legacy_knn_epanechnikov_vector(matrix, times, b, int(method_params.get("k", 3)))
            else:
                raise ValueError(f"Unsupported legacy day-vector method: {method}")
            if np.all(np.isfinite(candidate)):
                candidate_vals[p, :, b] = candidate

    candidate_da = xr.DataArray(candidate_vals, coords=rel.coords, dims=rel.dims, name=f"{method}_relative_candidate")
    candidate_log = _relative_abundance_to_log_scale(candidate_da, eps=eps, name=f"{method}_log_candidate")
    return _write_back_only_masked_finite(da_input, candidate_log, replace_mask)


def _legacy_predict_abundances_day_by_day(
    x: np.ndarray,
    delta_t: float,
    self_growth: np.ndarray,
    interaction_matrix: np.ndarray,
    perturbation_matrix: Optional[np.ndarray] = None,
    perturbation_incidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Predict the next relative-abundance vector in the style of the reference gLV implementation.

    The reference code assumes integer unit steps and contains a perturbation loop bug. Here we keep
    the same gLV update rule but make it runnable for arbitrary positive `delta_t` and for the current
    no-perturbation setting.
    """
    x_t = np.asarray(copy.deepcopy(x), dtype=float).reshape(-1, 1)
    if not np.all(np.isfinite(x_t)) or float(np.sum(x_t)) <= 0.0 or delta_t <= 0.0:
        return np.full_like(x_t, np.nan, dtype=float)

    pert = None
    if perturbation_matrix is not None and perturbation_incidence is not None:
        pert = np.asarray(perturbation_matrix, dtype=float)
        pert_inc = np.asarray(perturbation_incidence, dtype=float)
    else:
        pert_inc = None

    delta = np.zeros_like(x_t, dtype=float)
    for i in range(x_t.shape[0]):
        dx = float(self_growth[i, 0])
        for j in range(x_t.shape[0]):
            dx += float(interaction_matrix[i, j]) * float(x_t[j, 0])
        if pert is not None and pert_inc is not None:
            for j in range(pert_inc.shape[0]):
                dx += float(pert_inc[j, 0]) * float(pert[i, j])
        delta[i, 0] = np.exp(dx * float(delta_t)) * float(x_t[i, 0])

    delta = np.where(delta < 0.0, 0.0, delta)
    total = float(np.sum(delta))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(x_t, np.nan, dtype=float)
    return delta / total


def _legacy_create_y_matrix(X: np.ndarray, pert_incidence: Optional[np.ndarray] = None) -> np.ndarray:
    ones = np.ones((1, X.shape[1] - 1))
    y_mat = np.concatenate((X[:, 1:], ones), axis=0)
    if pert_incidence is not None:
        y_mat = np.concatenate((y_mat, pert_incidence), axis=0)
    return y_mat


def _legacy_create_F_matrix(X: np.ndarray, time_points: np.ndarray) -> np.ndarray:
    cols = []
    for i in range(X.shape[1] - 1):
        dt = float(time_points[i + 1] - time_points[i])
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("time_points must be strictly increasing for legacy gLV methods")
        ratio = np.divide(X[:, i + 1], X[:, i], out=np.ones(X.shape[0], dtype=float), where=X[:, i] > 0.0)
        f_vec = np.log(ratio).reshape(X.shape[0], 1)
        f_vec = np.nan_to_num(f_vec, nan=0.0, posinf=0.0, neginf=0.0) / dt
        cols.append(f_vec)
    return np.concatenate(cols, axis=1) if cols else np.empty((X.shape[0], 0), dtype=float)


def _legacy_mse_glv(F: np.ndarray, Y_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    tmp = np.matmul(Y_mat, Y_mat.T)
    tmp = np.linalg.inv(tmp)
    tmp = np.matmul(tmp, Y_mat)
    tmp = np.matmul(tmp, F.T)
    m = F.shape[0]
    M = tmp[:m, :].T
    mue = tmp[m : m + 1, :].reshape(1, tmp.shape[1]).T
    pert = None
    if Y_mat.shape[0] > F.shape[0] + 1:
        pert = tmp[m + 1 :, :].T
    return M, mue, pert


def _legacy_coef_matrix_by_paper_trick_ridge(
    Y: np.ndarray,
    F: np.ndarray,
    lambda_M: float,
    lambda_mue: float,
    lambda_E: float,
) -> np.ndarray:
    num_of_spec = F.shape[0]
    num_of_pert = Y.shape[0] - num_of_spec - 1
    lambda_for_D = [lambda_M for _ in range(num_of_spec)] + [lambda_mue] + [lambda_E for _ in range(num_of_pert)]
    D = np.diagflat(lambda_for_D)
    Y_mult = np.matmul(Y, Y.T)
    Y_add = np.add(Y_mult, D)
    Y_add_inv = np.linalg.inv(Y_add)
    return np.matmul(F, np.matmul(Y.T, Y_add_inv))


def _legacy_limits(
    F: np.ndarray,
    data: np.ndarray,
    error_threshold: float,
    repeats: int,
    med_or_avg: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    all_M = []
    all_mue = []
    n_cols = F.shape[1]
    if n_cols < 2:
        raise ValueError("LIMITS requires at least two transition columns")

    for _ in range(max(int(repeats), 1)):
        data_nums = np.arange(n_cols)
        rng.shuffle(data_nums)
        split = max(1, int(np.floor(n_cols * 0.5)))
        split = min(split, n_cols - 1)
        train_nums = np.sort(data_nums[:split])
        test_nums = np.sort(data_nums[split:])
        if train_nums.size == 0 or test_nums.size == 0:
            raise ValueError("LIMITS train/test split failed")

        train_F = F[:, train_nums]
        train_data = data[:, train_nums]
        test_F = F[:, test_nums]
        test_data = data[:, test_nums]
        tmp_mue = np.zeros((F.shape[0], 1), dtype=float)
        tmp_M = np.zeros((F.shape[0], F.shape[0]), dtype=float)

        for main_spec in range(F.shape[0]):
            cur_spec_train_F = train_F[main_spec : main_spec + 1, :].T.copy()
            inactive_list = list(range(F.shape[0]))

            cur_spec = np.ones((cur_spec_train_F.shape[0], 1), dtype=float)
            reg = LinearRegression(fit_intercept=False).fit(cur_spec, cur_spec_train_F)
            tmp_mue[main_spec, 0] = float(reg.coef_[0][0])
            cur_spec_train_F = cur_spec_train_F - cur_spec * tmp_mue[main_spec, 0]

            cur_spec_test = np.ones((test_F.shape[1], 1), dtype=float)
            cur_spec_test_F = cur_spec_test * tmp_mue[main_spec, 0]

            inactive_list.remove(main_spec)
            cur_spec = train_data[main_spec : main_spec + 1, :].T
            reg = LinearRegression(fit_intercept=False).fit(cur_spec, cur_spec_train_F)
            tmp_M[main_spec, main_spec] = float(reg.coef_[0][0])
            cur_spec_train_F = cur_spec_train_F - cur_spec * tmp_M[main_spec, main_spec]

            cur_spec_test = test_data[main_spec : main_spec + 1, :].T
            cur_spec_test_F = cur_spec_test_F + cur_spec_test * tmp_M[main_spec, main_spec]
            denom = max(cur_spec_test_F.shape[0], 1)
            error = float(np.mean((cur_spec_test_F - test_F[main_spec : main_spec + 1, :].T) ** 2))
            if not np.isfinite(error):
                error = np.inf

            improve_model = True
            while improve_model:
                best_error = np.inf
                best_spec = None
                best_spec_coeff = None
                for spec_index in inactive_list:
                    cur_spec = train_data[spec_index : spec_index + 1, :].T
                    reg = LinearRegression(fit_intercept=False).fit(cur_spec, cur_spec_train_F)
                    tmp_cur_coeff = float(reg.coef_[0][0])
                    cur_spec_test = test_data[spec_index : spec_index + 1, :].T * tmp_cur_coeff
                    tmp_test_F = cur_spec_test_F + cur_spec_test
                    tmp_error = float(np.mean((tmp_test_F - test_F[main_spec : main_spec + 1, :].T) ** 2))
                    if tmp_error < best_error:
                        best_error = tmp_error
                        best_spec = spec_index
                        best_spec_coeff = tmp_cur_coeff

                if best_spec is None or not np.isfinite(best_error):
                    improve_model = False
                    continue
                improve_ratio = np.inf if error == 0.0 else (error - best_error) / error
                if improve_ratio >= error_threshold:
                    tmp_M[main_spec, best_spec] = best_spec_coeff
                    inactive_list.remove(best_spec)
                    error = best_error
                    cur_spec = train_data[best_spec : best_spec + 1, :].T
                    cur_spec_train_F = cur_spec_train_F - cur_spec * best_spec_coeff
                    cur_spec_test = test_data[best_spec : best_spec + 1, :].T
                    cur_spec_test_F = cur_spec_test_F + cur_spec_test * best_spec_coeff
                else:
                    improve_model = False
                if not inactive_list:
                    improve_model = False

        all_mue.append(tmp_mue)
        all_M.append(tmp_M)

    final_mue = np.zeros((F.shape[0], 1), dtype=float)
    final_M = np.zeros((F.shape[0], F.shape[0]), dtype=float)
    reducer = np.median if int(med_or_avg) == 0 else np.mean
    for k in range(F.shape[0]):
        final_mue[k, 0] = float(reducer([all_mue[r][k, 0] for r in range(len(all_mue))]))
        for l in range(F.shape[0]):
            final_M[k, l] = float(reducer([all_M[r][k, l] for r in range(len(all_M))]))
    return final_mue, final_M


def _apply_legacy_glv_family_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    observed_mask: xr.DataArray,
    method: str,
    method_params: Dict,
    eps: float = 1e-3,
) -> xr.DataArray:
    """
    Apply a legacy gLV-family comparator on the random cell-mask task.

    The model is fit on a mean-initialized complete relative-abundance tensor. It may predict full
    participant-day vectors internally, but only masked cells are written back in the returned tensor.
    Perturbation terms are intentionally not supported here.
    """
    rel_init_da = _mean_initialize_relative_abundance(da_input, observed_mask, eps=eps)
    time_points = np.array([_bin_to_time_value(x) for x in da_input.coords["bin"].values], dtype=float)
    candidate_vals = np.full(rel_init_da.shape, np.nan, dtype=float)

    for p in range(rel_init_da.shape[0]):
        X = rel_init_da.values[p, :, :].astype(float)
        if not np.all(np.isfinite(X)):
            continue
        try:
            F = _legacy_create_F_matrix(X, time_points)
            if F.shape[1] == 0:
                continue
            if method == "glv_mse_legacy":
                Y = _legacy_create_y_matrix(X, pert_incidence=None)
                M, mue, pert = _legacy_mse_glv(F, Y)
            elif method == "glv_ridge_legacy":
                Y = _legacy_create_y_matrix(X, pert_incidence=None)
                coef = _legacy_coef_matrix_by_paper_trick_ridge(
                    Y=Y,
                    F=F,
                    lambda_M=float(method_params.get("lambda_M", 1e-3)),
                    lambda_mue=float(method_params.get("lambda_mue", 1e-3)),
                    lambda_E=float(method_params.get("lambda_E", 1e-3)),
                )
                n_taxa = F.shape[0]
                M = coef[:, :n_taxa]
                mue = coef[:, n_taxa : n_taxa + 1]
                pert = coef[:, n_taxa + 1 :] if coef.shape[1] > n_taxa + 1 else None
            elif method == "limits_legacy":
                M_input = X[:, :-1]
                mue, M = _legacy_limits(
                    F=F,
                    data=M_input,
                    error_threshold=float(method_params.get("error_threshold", 0.05)),
                    repeats=int(method_params.get("repeats", 20)),
                    med_or_avg=int(method_params.get("med_or_avg", 0)),
                )
                pert = None
            else:
                raise ValueError(f"Unsupported legacy dynamic method: {method}")

            pred = X[:, 0:1].copy()
            for b in range(1, X.shape[1]):
                dt = float(time_points[b] - time_points[b - 1])
                if dt <= 0.0 or not np.isfinite(dt):
                    pred = np.full_like(pred, np.nan)
                else:
                    pred = _legacy_predict_abundances_day_by_day(
                        x=pred,
                        delta_t=dt,
                        self_growth=mue,
                        interaction_matrix=M,
                        perturbation_matrix=pert,
                        perturbation_incidence=None,
                    )
                if np.all(np.isfinite(pred)):
                    candidate_vals[p, :, b] = pred[:, 0]
        except Exception:
            continue

    candidate_da = xr.DataArray(
        candidate_vals,
        coords=rel_init_da.coords,
        dims=rel_init_da.dims,
        name=f"{method}_relative_candidate",
    )
    candidate_log = _relative_abundance_to_log_scale(candidate_da, eps=eps, name=f"{method}_log_candidate")
    return _write_back_only_masked_finite(da_input, candidate_log, replace_mask)


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
    taxon_fallbacks = _build_taxon_positive_fallbacks(
        values=values,
        observed=observed,
        times=times,
        zero_value=zero_value,
    )
    trajectory_fallback_count = 0
    no_positive_support_count = 0

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.any(replace[i, j, :]):
                continue
            if not np.any(observed[i, j, :]):
                n_targets = int(np.sum(replace[i, j, :]))
                trajectory_fallback_count += n_targets
                no_positive_support_count += int(
                    np.sum(replace[i, j, :] & (taxon_fallbacks[j, :] == zero_value))
                )
            out_vals[i, j, :] = _impute_local_trajectory(
                values=values[i, j, :],
                replace=replace[i, j, :],
                observed=observed[i, j, :],
                times=times,
                method=method,
                method_params=method_params,
                zero_value=zero_value,
                fallback_values=taxon_fallbacks[j, :],
            )

    out = da_input.copy(deep=True)
    out.values[:] = np.where(replace, out_vals, values)
    out.attrs["positive_trajectory_fallback_count"] = int(trajectory_fallback_count)
    out.attrs["no_positive_support_count"] = int(no_positive_support_count)
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
    eps: float = 1e-3,
    verbose: bool = False,
) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Unified second-stage imputation dispatcher.

    Fallback policy for retained local methods:
    - donors are original, unmasked, finite values strictly above log-floor;
    - `nearest_positive` searches the full trajectory by actual time distance;
    - `temporal_knn` uses all available positive donors when fewer than K exist;
    - if a trajectory has no donor, use the same taxon's median at the nearest
      supported time across other participants; only a wholly unsupported taxon
      remains at log-floor.
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
            floor_value=zero_value,
            verbose=verbose,
        )

        bins_trained = np.array(meta["bin"])
        abar = meta["Abar"]
        da_imp = da_input.copy(deep=True)

        for ti, b in enumerate(bins_trained):
            ai = da_imp.sel(bin=b).values
            ahat = np.maximum(abar + (z @ c[ti] @ x.T), zero_value)
            mk = replace_mask.sel(bin=b).values.astype(bool)
            if np.any(mk):
                ai_new = ai.copy()
                ai_new[mk] = ahat[mk]
                da_imp.loc[dict(bin=b)] = ai_new
    elif method == "ctf":
        da_imp = impute_competitor(
            da_input=da_input,
            replace_mask=replace_mask,
            observed_mask=observed_mask,
            zero_value=zero_value,
            method=method,
            seed=int(model_seed),
            params={**method_params, "eps": float(eps)},
            n_threads=1,
        )
    elif method in {"mean", "nearest_positive", "temporal_knn"}:
        da_imp = _impute_local_method(
            da_input=da_input,
            replace_mask=replace_mask,
            observed_mask=observed_mask,
            method=method,
            method_params=method_params,
            zero_value=zero_value,
        )
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    da_imp = da_imp.transpose(*da_input.dims)
    off_mask = ~replace_mask.transpose(*da_input.dims).values.astype(bool)
    if not np.array_equal(da_imp.values[off_mask], da_input.values[off_mask]):
        raise AssertionError(f"method={method} modified an off-mask cell")
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
    compact_output: bool = False,
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
            eps=eps,
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
        init_method = (
            "mean"
            if method in {"dbn_dense", "dbn_sparse", "glv", "glv_mse_legacy", "glv_ridge_legacy", "limits_legacy"}
            else method
        )

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

        df_taxon_marginal_eval_all.to_csv(
            outdir / f"taxon_marginal_eval_frac_{frac_str}_repeat_{replicate}.csv",
            index=False,
        )
        df_taxon_trajectory_eval_all.to_csv(
            outdir / f"taxon_trajectory_eval_frac_{frac_str}_repeat_{replicate}.csv",
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
        if not compact_output:
            df_taxon_marginal_stats_all.to_csv(
                outdir / f"taxon_marginal_stats_frac_{frac_str}_repeat_{replicate}.csv",
                index=False,
            )
            df_taxon_bin_stats_all.to_csv(
                outdir / f"taxon_bin_stats_frac_{frac_str}_repeat_{replicate}.csv",
                index=False,
            )
            df_taxon_trajectory_detail_all.to_csv(
                outdir / f"taxon_trajectory_detail_frac_{frac_str}_repeat_{replicate}.csv",
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
    hybrid_observed: bool = False,
    verbose: bool = False,
) -> xr.DataArray:
    del init_missing_mask
    da_truth = build_full_p2d2_truth(
        da_base=da_base.copy(deep=True),
        zero_value=zero_value,
        pca_params=pca_params,
        model_seed=seed_base + 500000,
        truth_observed_mode=truth_observed_mode,
        hybrid_observed=hybrid_observed,
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


def merge_sim1_shards(outdir: Path) -> None:
    """Merge race-free single-repeat array outputs after all shards finish."""
    outdir = Path(outdir)
    rep_paths = sorted(outdir.glob("replicate_metrics_frac_*_repeat_*.csv"))
    bin_paths = sorted(outdir.glob("per_bin_metrics_frac_*_repeat_*.csv"))
    if not rep_paths:
        raise RuntimeError(f"No Sim1 replicate shard files found in {outdir}")
    all_reps = pd.concat([pd.read_csv(path) for path in rep_paths], ignore_index=True)
    all_reps = all_reps.drop_duplicates(
        subset=["simulation", "frac", "replicate", "scope", "method"], keep="last"
    ).sort_values(["simulation", "frac", "replicate", "scope", "method"])
    all_reps.to_csv(outdir / "all_replicates_metrics_sim1.csv", index=False)
    summarize_replicates(all_reps).to_csv(outdir / "summary_with_errorbar_sim1.csv", index=False)
    if bin_paths:
        all_bins = pd.concat([pd.read_csv(path) for path in bin_paths], ignore_index=True)
        all_bins = all_bins.drop_duplicates(
            subset=["simulation", "frac", "replicate", "scope", "method", "bin"], keep="last"
        ).sort_values(["simulation", "frac", "replicate", "scope", "method", "bin"])
        all_bins.to_csv(outdir / "per_bin_metrics_sim1.csv", index=False)
    merge_abundance_eval_outputs(outdir=outdir)


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
    unsupported = sorted(set(methods) - SIM1_SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(f"Unsupported methods requested: {unsupported}")
    impute_method_params = {
        "mean": {},
        "nearest_positive": {},
        "temporal_knn": {
            "k": args.temporal_knn_k,
            "bandwidth": args.temporal_knn_bandwidth,
        },
        "ctf": {
            "rank": args.competitor_ctf_rank,
            "n_initializations": args.competitor_ctf_initializations,
            "max_iterations_als": args.competitor_ctf_max_iterations,
            "max_iterations_rptm": args.competitor_ctf_max_iterations,
        },
    }
    methods, skipped = available_requested_methods(methods, impute_method_params)
    if skipped:
        details = "; ".join(f"{method}: {reason}" for method, reason in skipped.items())
        raise RuntimeError(f"Requested retained competitor is unavailable: {details}")
    if not methods:
        raise RuntimeError("No requested methods are available")

    if args.merge_only:
        merge_sim1_shards(outdir)
        return

    da_base, init_missing_mask = load_base_tensor_with_missing_mask(args.tax_path, args.meta_path, eps=eps)

    requested_truth_path = Path(args.truth_path) if args.truth_path else None
    truth_loaded = bool(
        requested_truth_path is not None
        and requested_truth_path.exists()
        and not args.force_rebuild_truth
    )
    if args.require_existing_truth and not truth_loaded:
        raise FileNotFoundError(
            f"Frozen Sim1 truth does not exist: {requested_truth_path}. "
            "Array shards are read-only and are not allowed to rebuild truth."
        )
    if truth_loaded:
        da_truth = xr.open_dataarray(requested_truth_path).load()
        print(f"Loaded frozen truth tensor: {requested_truth_path}")
    else:
        da_truth = prepare_truth_data_for_sim1(
            da_base=da_base,
            init_missing_mask=init_missing_mask,
            zero_value=zero_value,
            pca_params=pca_params,
            seed_base=args.seed_base,
            truth_observed_mode=args.truth_observed_mode,
            hybrid_observed=args.truth_hybrid,
            verbose=args.verbose,
        )
        truth_obs_mask = build_observed_mask(
            da_input=da_base,
            zero_value=zero_value,
            mode=args.truth_observed_mode,
            replace_mask=None,
        )
        truth_diagnostics = {
            **pca_params,
            **summarize_truth_dynamics(da_truth, eps=eps, observed_mask=truth_obs_mask),
            "truth_hybrid": int(bool(args.truth_hybrid)),
            "unsupported_trajectory_count": int(da_truth.attrs.get("unsupported_trajectory_count", 0)),
            "taxon_penalty_ratio": float(da_truth.attrs.get("taxon_penalty_ratio", np.nan)),
            "participant_penalty_ratio": float(da_truth.attrs.get("participant_penalty_ratio", np.nan)),
        }
        diagnostics_path = _safe_output_path(outdir, "truth_diagnostics_sim1.csv")
        pd.DataFrame([truth_diagnostics]).to_csv(diagnostics_path, index=False)
        print(f"Saved truth diagnostics: {diagnostics_path}")
    if args.build_truth_only:
        truth_path = requested_truth_path or outdir / "truth_tensor_sim1.nc"
        if not truth_loaded:
            truth_path.parent.mkdir(parents=True, exist_ok=True)
            da_truth.to_netcdf(truth_path)
            print(f"Saved truth tensor: {truth_path}")
        return

    if (args.frac is None) != (args.repeat is None):
        raise ValueError("--frac and --repeat must be provided together")
    if args.frac is not None:
        frac = float(args.frac)
        replicate = int(args.repeat)
        frac_seed_base = args.seed_base + int(round(frac * 1_000_000))
        df_bins, df_reps = run_one_replication(
            da_truth=da_truth,
            simulation="sim1",
            frac=frac,
            replicate=replicate,
            mask_seed=frac_seed_base + replicate,
            model_seed=frac_seed_base + 10000 + replicate,
            zero_value=zero_value,
            pca_params=pca_params,
            methods=methods,
            impute_method_params=impute_method_params,
            second_observed_mode=args.second_observed_mode,
            mask_eligible_mode=args.mask_eligible_mode,
            outdir=outdir,
            eps=eps,
            verbose=args.verbose,
            compact_output=bool(args.compact_output),
        )
        frac_str = _format_frac(frac)
        df_reps.to_csv(outdir / f"replicate_metrics_frac_{frac_str}_repeat_{replicate}.csv", index=False)
        df_bins.to_csv(outdir / f"per_bin_metrics_frac_{frac_str}_repeat_{replicate}.csv", index=False)
        return

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
    parser.add_argument("--truth-path", type=str, default=None)
    parser.add_argument("--build-truth-only", action="store_true")
    parser.add_argument("--force-rebuild-truth", action="store_true")
    parser.add_argument(
        "--require-existing-truth", action="store_true",
        help="Fail if --truth-path is absent; array shards never rebuild frozen truth.",
    )
    parser.add_argument("--frac", type=float, default=None, help="Run one masking fraction shard.")
    parser.add_argument("--repeat", type=int, default=None, help="Run one repeat shard; requires --frac.")
    parser.add_argument("--merge-only", action="store_true", help="Merge completed array shards without fitting.")
    parser.add_argument(
        "--compact-output", action="store_true",
        help="Omit large intermediate taxon stats/detail CSVs while retaining summary metrics.",
    )
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
        "--truth-hybrid",
        action="store_true",
        help=(
            "Build a hybrid frozen truth: keep raw observed cells and fill only "
            "originally-unobserved cells with the fitted reconstruction.  Preserves "
            "the genuine temporal dynamics of the observed data."
        ),
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
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--lam", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--t0", type=int, default=30)
    parser.add_argument("--t", type=int, default=30)
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--competitor-ctf-rank", type=int, default=3)
    parser.add_argument("--competitor-ctf-initializations", type=int, default=10)
    parser.add_argument("--competitor-ctf-max-iterations", type=int, default=50)
    parser.add_argument("--temporal-knn-k", type=int, default=3)
    parser.add_argument("--temporal-knn-bandwidth", type=float, default=4.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_sim1_pipeline(args)


if __name__ == "__main__":
    main()
