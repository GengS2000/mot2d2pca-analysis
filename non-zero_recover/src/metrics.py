from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .preprocessing import validate_tensor


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).correlation)
    except Exception:
        return safe_pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def safe_wasserstein(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(x, y))
    except Exception:
        qs = np.linspace(0, 1, min(x.size, y.size))
        return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def js_divergence(x, y, bins: int = 30) -> float:
    """Jensen-Shannon divergence between two one-dimensional distributions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    if np.isclose(lo, hi):
        return 0.0
    px, edges = np.histogram(x, bins=int(bins), range=(lo, hi), density=False)
    py, _ = np.histogram(y, bins=edges, density=False)
    px = px.astype(float) + 1e-12
    py = py.astype(float) + 1e-12
    px /= px.sum()
    py /= py.sum()
    m = 0.5 * (px + py)
    return float(0.5 * np.sum(px * np.log(px / m)) + 0.5 * np.sum(py * np.log(py / m)))


def evaluate_recovery(da_true: xr.DataArray, da_imputed: xr.DataArray, eval_mask: xr.DataArray) -> Dict[str, Any]:
    """Evaluate numeric recovery on artificial masked nonzero entries."""
    validate_tensor(da_true, "da_true")
    m = eval_mask.values.astype(bool)
    y = da_true.values[m].astype(float)
    p = da_imputed.values[m].astype(float)
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    err = p - y
    return {
        "n_eval": int(y.size),
        "mae": float(np.mean(np.abs(err))) if y.size else np.nan,
        "rmse": float(np.sqrt(np.mean(err**2))) if y.size else np.nan,
        "pearson": safe_pearson(p, y),
        "spearman": safe_spearman(p, y),
        "wasserstein": safe_wasserstein(p, y),
        "bias": float(np.mean(err)) if y.size else np.nan,
        "median_absolute_error": float(np.median(np.abs(err))) if y.size else np.nan,
    }


def lag1_autocorr(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    y = y[m]
    return safe_pearson(y[:-1], y[1:])


def trajectory_summary_metrics(da_true: xr.DataArray, da_masked: xr.DataArray, da_imputed: xr.DataArray, artificial_mask: xr.DataArray) -> Dict[str, Any]:
    """Compact trajectory metrics used by experiment 1."""
    affected = artificial_mask.values.any(axis=2)
    t = da_true.values
    p = da_imputed.values
    m = da_masked.values
    rmses, ac_err_imp, ac_err_mask = [], [], []
    for i, j in np.argwhere(affected):
        yt, yp, ym = t[i, j, :], p[i, j, :], m[i, j, :]
        finite = np.isfinite(yt) & np.isfinite(yp)
        if finite.sum() >= 1:
            rmses.append(float(np.sqrt(np.mean((yp[finite] - yt[finite]) ** 2))))
        ac_t = lag1_autocorr(yt)
        ac_p = lag1_autocorr(yp)
        ac_m = lag1_autocorr(ym)
        if np.isfinite(ac_t) and np.isfinite(ac_p):
            ac_err_imp.append(abs(ac_p - ac_t))
        if np.isfinite(ac_t) and np.isfinite(ac_m):
            ac_err_mask.append(abs(ac_m - ac_t))
    return {
        "trajectory_rmse_affected": float(np.nanmean(rmses)) if rmses else np.nan,
        "lag1_autocorr_abs_error": float(np.nanmean(ac_err_imp)) if ac_err_imp else np.nan,
        "lag1_autocorr_abs_error_masked": float(np.nanmean(ac_err_mask)) if ac_err_mask else np.nan,
    }


def trajectory_metrics_long(da_true: xr.DataArray, da_masked: xr.DataArray, da_imputed: xr.DataArray, artificial_mask: xr.DataArray, method: str, mechanism: str, frac: float, repeat: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trajectory and per-subject metrics for experiment 3."""
    rows, subject_rows = [], []
    true, masked, imp = da_true.values, da_masked.values, da_imputed.values
    affected = artificial_mask.values.any(axis=2)
    participants = da_true.coords["participant"].values
    taxa = da_true.coords["taxon"].values
    for i in range(true.shape[0]):
        subj_err = []
        for j in range(true.shape[1]):
            yt, ym, yp = true[i, j], masked[i, j], imp[i, j]
            finite = np.isfinite(yt) & np.isfinite(yp)
            if finite.sum() < 3 and not affected[i, j]:
                continue
            rmse = float(np.sqrt(np.mean((yp[finite] - yt[finite]) ** 2))) if finite.any() else np.nan
            subj_err.append(rmse)
            d2_true = np.diff(yt, n=2)
            d2_masked = np.diff(ym, n=2)
            d2_imp = np.diff(yp, n=2)
            rows.append({
                "mechanism": mechanism, "frac": float(frac), "repeat": int(repeat), "method": method,
                "participant": str(participants[i]), "taxon": str(taxa[j]), "affected": bool(affected[i, j]),
                "trajectory_pearson": safe_pearson(yp, yt), "trajectory_spearman": safe_spearman(yp, yt),
                "trajectory_rmse": rmse,
                "lag1_error_imputed": abs(lag1_autocorr(yp) - lag1_autocorr(yt)) if np.isfinite(lag1_autocorr(yp)) and np.isfinite(lag1_autocorr(yt)) else np.nan,
                "lag1_error_masked": abs(lag1_autocorr(ym) - lag1_autocorr(yt)) if np.isfinite(lag1_autocorr(ym)) and np.isfinite(lag1_autocorr(yt)) else np.nan,
                "second_difference_mae_imputed": float(np.nanmean(np.abs(d2_imp - d2_true))) if d2_true.size else np.nan,
                "second_difference_mae_masked": float(np.nanmean(np.abs(d2_masked - d2_true))) if d2_true.size else np.nan,
            })
        subject_rows.append({"mechanism": mechanism, "frac": float(frac), "repeat": int(repeat), "method": method, "participant": str(participants[i]), "subject_trajectory_rmse": float(np.nanmean(subj_err)) if subj_err else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(subject_rows)


def simple_dtw(x: np.ndarray, y: np.ndarray) -> float:
    """Small dependency-free dynamic time warping distance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def summarize_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Mean/std summary for numeric columns."""
    if df.empty:
        return df
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in group_cols]
    return df.groupby(group_cols, dropna=False)[num_cols].agg(["mean", "std", "count"]).reset_index()


def taxon_distribution_metrics(da_true: xr.DataArray, da_masked: xr.DataArray, da_imputed: xr.DataArray, artificial_mask: xr.DataArray, method: str, mechanism: str, frac: float, repeat: int) -> pd.DataFrame:
    """Taxon-wise distribution preservation metrics for experiment 2."""
    rows = []
    taxa = da_true.coords["taxon"].values
    true, masked, imp, mk = da_true.values, da_masked.values, da_imputed.values, artificial_mask.values.astype(bool)
    for j, taxon in enumerate(taxa):
        true_tax = true[:, j, :].ravel()
        imp_tax = imp[:, j, :].ravel()
        mask_tax = mk[:, j, :].ravel()
        true_masked = true[:, j, :].ravel()[mask_tax]
        imp_masked = imp[:, j, :].ravel()[mask_tax]
        rows.append({
            "mechanism": mechanism, "frac": float(frac), "repeat": int(repeat), "method": method, "taxon": str(taxon),
            "n_masked": int(mask_tax.sum()),
            "wasserstein_masked": safe_wasserstein(imp_masked, true_masked),
            "js_full": js_divergence(imp_tax, true_tax),
            "original_mean": float(np.nanmean(true_tax)), "imputed_mean": float(np.nanmean(imp_tax)),
            "original_sd": float(np.nanstd(true_tax)), "imputed_sd": float(np.nanstd(imp_tax)),
        })
    return pd.DataFrame(rows)


def static_correlation_matrix(da: xr.DataArray, top_taxa: Optional[np.ndarray] = None, nonzero_value: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Taxon-taxon correlation after flattening participant-bin observations."""
    arr = da.values if top_taxa is None else da.values[:, top_taxa, :]
    x = arr.transpose(0, 2, 1).reshape(-1, arr.shape[1])
    n = x.shape[1]
    corr = np.full((n, n), np.nan)
    n_pair = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            m = np.isfinite(x[:, a]) & np.isfinite(x[:, b])
            if nonzero_value is not None:
                m &= (x[:, a] > nonzero_value) & (x[:, b] > nonzero_value)
            n_pair[a, b] = int(m.sum())
            corr[a, b] = safe_pearson(x[m, a], x[m, b])
    return corr, n_pair


def lagged_correlation_matrix(da: xr.DataArray, top_taxa: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute cor(x_j,t, x_k,t+1) pooled across participants."""
    arr = da.values if top_taxa is None else da.values[:, top_taxa, :]
    x0 = arr[:, :, :-1].transpose(0, 2, 1).reshape(-1, arr.shape[1])
    x1 = arr[:, :, 1:].transpose(0, 2, 1).reshape(-1, arr.shape[1])
    n = arr.shape[1]
    corr = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            corr[a, b] = safe_pearson(x0[:, a], x1[:, b])
    return corr


def corr_matrix_similarity(reference: np.ndarray, candidate: np.ndarray, upper_only: bool = True) -> Dict[str, float]:
    """Similarity metrics between two correlation matrices."""
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if upper_only and ref.shape[0] == ref.shape[1]:
        idx = np.triu_indices(ref.shape[0], k=1)
        x, y = ref[idx], cand[idx]
    else:
        x, y = ref.ravel(), cand.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    diff = y[m] - x[m]
    return {"pearson": safe_pearson(y[m], x[m]), "spearman": safe_spearman(y[m], x[m]), "frobenius": float(np.sqrt(np.sum(diff**2))) if diff.size else np.nan, "n_entries": int(m.sum())}


def top_edges(corr: np.ndarray, taxa: Iterable[str], k: int = 100) -> pd.DataFrame:
    """Create a ranked directed edge list from a correlation matrix."""
    taxa = list(map(str, taxa))
    rows = []
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i == j or not np.isfinite(corr[i, j]):
                continue
            rows.append((taxa[i], taxa[j], float(corr[i, j]), abs(float(corr[i, j]))))
    rows = sorted(rows, key=lambda x: x[3], reverse=True)[: int(k)]
    return pd.DataFrame([{"source_taxon": a, "target_taxon": b, "lagged_correlation": c, "abs_corr": ac, "rank": r + 1} for r, (a, b, c, ac) in enumerate(rows)])


def edge_overlap_metrics(ref_edges: pd.DataFrame, cand_edges: pd.DataFrame, k_values=(50, 100, 200)) -> pd.DataFrame:
    """Top-k directed edge overlap, Jaccard, and precision."""
    rows = []
    for k in k_values:
        r = set(map(tuple, ref_edges.head(k)[["source_taxon", "target_taxon"]].to_numpy()))
        c = set(map(tuple, cand_edges.head(k)[["source_taxon", "target_taxon"]].to_numpy()))
        inter = len(r & c)
        union = len(r | c)
        rows.append({"k": int(k), "edge_overlap": inter, "jaccard": inter / union if union else np.nan, "topk_precision": inter / len(c) if c else np.nan})
    return pd.DataFrame(rows)

