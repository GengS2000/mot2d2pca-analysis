from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import svd
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitor_methods import (
    COMPETITOR_GROUPS,
    COMPETITOR_METHODS,
    available_requested_methods,
    impute_competitor,
)


DEFAULT_METHODS = [
    "penalized_2d2_pca",
    "ctf",
    "mean",
    "nearest_positive",
    "temporal_knn",
]
SIM3_SUPPORTED_METHODS = frozenset(DEFAULT_METHODS)

METHOD_GROUP_MAP = {
    "penalized_2d2_pca": "matrix_factorization",
    "ctf": COMPETITOR_GROUPS["ctf"],
    "mean": "naive",
    "nearest_positive": "local_time",
    "median": "naive",
    "equal": "naive",
    "last": "local_time",
    "weighted_time_average": "local_time",
    "temporal_knn": "local_time",
    "spline": "local_time",
    "weighted_avg_legacy": "legacy_local_time",
    "spline_legacy": "legacy_local_time",
    "knn_epanechnikov_legacy": "legacy_local_time",
    "dbn_dense": "dynamic",
    "dbn_sparse": "dynamic",
    "glv": "dynamic",
    "glv_mse_legacy": "legacy_dynamic",
    "glv_ridge_legacy": "legacy_dynamic",
    "limits_legacy": "legacy_dynamic",
    **COMPETITOR_GROUPS,
}

LEGACY_LOCAL_TIME_METHODS = {
    "weighted_avg_legacy",
    "spline_legacy",
    "knn_epanechnikov_legacy",
}

LEGACY_DYNAMIC_METHODS = {
    "glv_mse_legacy",
    "glv_ridge_legacy",
    "limits_legacy",
}


# =====================
# Data preprocessing
# =====================

def week_to_bin(week: int) -> Optional[str]:
    if week == 0:
        return "0"
    if 1 <= week <= 36:
        start = ((week - 1) // 4) * 4 + 1
        end = start + 3
        return f"{start}-{end}"
    return None


def build_base_tensor(
    tax_path: str = "hmp2_mgx_taxonomy.tsv",
    meta_path: str = "metagenomics.xlsx",
    eps: float = 1e-3,
    min_pos_cells: int = 10,
    min_participants: int = 5,
    verbose: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray, Dict[str, object]]:
    """
    Build the base log10(percent + eps) tensor.

    Logic aligned with the user's intended pipeline:
    1) True missing values in participant-bin grid are filled with 0.
    2) Then transform by log10(x + eps), so missing/zero -> log10(eps).
    3) CTF truth-building uses only original positive observations as observed entries.
    """
    tax = pd.read_csv(tax_path, sep="\t")
    tax = tax.rename(columns={tax.columns[0]: "taxon"}).set_index("taxon")
    meta = pd.read_excel(meta_path)

    tax.columns = tax.columns.str.replace(r"_profile$", "", regex=True)

    meta_mgx = (
        meta.loc[
            meta["data_type"].eq("metagenomics"),
            ["External ID", "Participant ID", "week_num"],
        ]
        .dropna(subset=["External ID", "Participant ID", "week_num"])
        .rename(
            columns={
                "External ID": "external_id",
                "Participant ID": "participant",
                "week_num": "week",
            }
        )
    )
    meta_mgx["week"] = meta_mgx["week"].astype(int)
    meta_mgx["participant"] = meta_mgx["participant"].astype(str)

    idx = tax.index.astype(str)
    has_species = idx.str.contains("s__", na=False)
    not_euk = ~idx.str.contains("k__Eukaryota", na=False)
    has_unclassified = idx.str.contains("unclassified", na=False)
    has_cag = idx.str.contains(r"CAG[_-]?\d+", regex=True, na=False)
    bad_unclassified = has_unclassified & (~has_cag)
    keep_tax_mask = has_species & not_euk & (~bad_unclassified)

    tax_before = tax.shape[0]
    tax = tax.loc[keep_tax_mask].copy()
    tax_after = tax.shape[0]
    if verbose:
        print(f"[taxon filter] taxa: {tax_before} -> {tax_after}")
    assert tax.index.to_series().str.contains("s__", na=False).all()

    common = sorted(set(tax.columns).intersection(set(meta_mgx["external_id"])))
    tax = tax[common]
    meta_mgx = meta_mgx.drop_duplicates("external_id").set_index("external_id").loc[common]

    x = tax.T
    x.index.name = "external_id"
    x = x.join(meta_mgx, how="left")
    assert x["participant"].isna().sum() == 0, "some External ID cannot be mapped to participant/week"

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

    pos_mask_full = x_full.gt(0).fillna(False)
    pos_cells_per_taxon = pos_mask_full.sum(axis=0)
    pos_by_participant = pos_mask_full.groupby(level="participant").any()
    pos_participants_per_taxon = pos_by_participant.sum(axis=0)
    keep_prev_mask = (
        (pos_cells_per_taxon >= min_pos_cells)
        & (pos_participants_per_taxon >= min_participants)
    )

    tax_before_prev = x_full.shape[1]
    x_full = x_full.loc[:, keep_prev_mask].copy()
    tax_after_prev = x_full.shape[1]
    taxa = x_full.columns.astype(str).tolist()

    if verbose:
        print(
            f"[prevalence filter] taxa: {tax_before_prev} -> {tax_after_prev} "
            f"(min_pos_cells={min_pos_cells}, min_participants={min_participants})"
        )
        print("[diag] kept pos_cells_per_taxon summary:")
        print(pos_cells_per_taxon[keep_prev_mask].describe())
        print("[diag] kept pos_participants_per_taxon summary:")
        print(pos_participants_per_taxon[keep_prev_mask].describe())

    # Important: fill missing with 0 first, then log10(x + eps)
    x_full_0 = x_full.fillna(0.0)
    x_full_log = np.log10(x_full_0 + eps)

    # Observed entries for CTF truth-building: only original positive cells.
    observed_positive_df = x_full.gt(0.0).fillna(False)

    arr = x_full_log.to_numpy().reshape(len(participants), len(bins), len(taxa))
    tensor = np.transpose(arr, (0, 2, 1))

    obs_arr = observed_positive_df.to_numpy().reshape(len(participants), len(bins), len(taxa))
    obs_tensor = np.transpose(obs_arr, (0, 2, 1))

    da = xr.DataArray(
        tensor,
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="abundance_binned_log",
    )
    da_observed_mask = xr.DataArray(
        obs_tensor,
        dims=("participant", "taxon", "bin"),
        coords={"participant": participants, "taxon": taxa, "bin": bins},
        name="observed_positive_mask",
    )

    meta_info = {
        "participants": participants,
        "bins": bins,
        "taxa": taxa,
        "eps": eps,
        "min_pos_cells": min_pos_cells,
        "min_participants": min_participants,
    }
    return da, da_observed_mask, meta_info


# =====================
# CTF-style observed-only reconstruction
# =====================

def _validate_da(da: xr.DataArray) -> None:
    assert da.dims == ("participant", "taxon", "bin"), (
        f"Expected dims ('participant','taxon','bin'), got {da.dims}"
    )
    for d, n in zip(da.dims, da.shape):
        if d not in da.coords:
            raise ValueError(f"Missing coord for dim '{d}'")
        if da.coords[d].size != n:
            raise ValueError(
                f"Coord length mismatch on dim '{d}': coord={da.coords[d].size}, shape={n}"
            )


def _unfold(x: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(x, mode, 0).reshape(x.shape[mode], -1)


def _init_factors_svd(
    x_filled: np.ndarray, rank: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    i, j, k = x_filled.shape
    a = np.zeros((i, rank), dtype=float)
    b = np.zeros((j, rank), dtype=float)
    c = np.zeros((k, rank), dtype=float)

    for mode, size, out in [(0, i, a), (1, j, b), (2, k, c)]:
        m = _unfold(x_filled, mode)
        try:
            u, _, _ = svd(m, full_matrices=False)
            r_take = min(rank, u.shape[1])
            out[:, :r_take] = u[:, :r_take]
            if r_take < rank:
                out[:, r_take:] = rng.normal(0, 1e-2, size=(size, rank - r_take))
        except Exception:
            out[:] = rng.normal(0, 1e-2, size=(size, rank))
    return a, b, c


def _init_factors_random(shape: Tuple[int, int, int], rank: int, seed: int):
    rng = np.random.default_rng(seed)
    i, j, k = shape
    a = rng.normal(0.0, 0.1, size=(i, rank))
    b = rng.normal(0.0, 0.1, size=(j, rank))
    c = rng.normal(0.0, 0.1, size=(k, rank))
    return a, b, c


def _normalize_factors(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, sigma: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rank = a.shape[1]
    for r in range(rank):
        na = np.linalg.norm(a[:, r]) + 1e-12
        nb = np.linalg.norm(b[:, r]) + 1e-12
        nc = np.linalg.norm(c[:, r]) + 1e-12
        a[:, r] /= na
        b[:, r] /= nb
        c[:, r] /= nc
        sigma[r] *= na * nb * nc
    return a, b, c, sigma


def _cp_reconstruct(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    return np.einsum("ir,jr,kr,r->ijk", a, b, c, sigma)


def _observed_loss(t: np.ndarray, m: np.ndarray, t_hat: np.ndarray) -> float:
    diff = (t_hat - t)[m]
    if diff.size == 0:
        return np.nan
    return float(np.mean(diff * diff))


def rclr_transform_tensor(da_ra_pct: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    if da_ra_pct.dims != ("participant", "taxon", "bin"):
        raise ValueError("da_ra_pct dims must be ('participant','taxon','bin')")

    x = da_ra_pct.values.astype(float)
    i, j, k = x.shape
    r = np.full((i, j, k), np.nan, dtype=float)
    m = np.zeros((i, j, k), dtype=bool)

    for ii in range(i):
        for kk in range(k):
            v = x[ii, :, kk]
            obs = np.isfinite(v) & (v > 0)
            if not np.any(obs):
                continue
            lv = np.log10(v[obs])
            gm_log = np.mean(lv)
            out = np.full(j, np.nan, dtype=float)
            out[obs] = lv - gm_log
            r[ii, :, kk] = out
            m[ii, :, kk] = obs

    rclr_da = xr.DataArray(r, dims=da_ra_pct.dims, coords=da_ra_pct.coords, name="rclr_da")
    observed_mask = xr.DataArray(m, dims=da_ra_pct.dims, coords=da_ra_pct.coords, name="observed_mask")
    return rclr_da, observed_mask


def _update_a(t, m, a, b, c, sigma, ridge=1e-6):
    i, _, _ = t.shape
    rank = a.shape[1]
    for ii in range(i):
        idx = np.argwhere(m[ii, :, :])
        if idx.size == 0:
            a[ii, :] = 0.0
            continue
        j_idx = idx[:, 0]
        k_idx = idx[:, 1]
        z = (b[j_idx, :] * c[k_idx, :]) * sigma[None, :]
        y = t[ii, j_idx, k_idx]
        g = z.T @ z + ridge * np.eye(rank)
        rhs = z.T @ y
        a[ii, :] = np.linalg.solve(g, rhs)
    return a


def _update_b(t, m, a, b, c, sigma, ridge=1e-6):
    _, j, _ = t.shape
    rank = b.shape[1]
    for jj in range(j):
        idx = np.argwhere(m[:, jj, :])
        if idx.size == 0:
            b[jj, :] = 0.0
            continue
        i_idx = idx[:, 0]
        k_idx = idx[:, 1]
        z = (a[i_idx, :] * c[k_idx, :]) * sigma[None, :]
        y = t[i_idx, jj, k_idx]
        g = z.T @ z + ridge * np.eye(rank)
        rhs = z.T @ y
        b[jj, :] = np.linalg.solve(g, rhs)
    return b


def _update_c(t, m, a, b, c, sigma, ridge=1e-6):
    _, _, k = t.shape
    rank = c.shape[1]
    for kk in range(k):
        idx = np.argwhere(m[:, :, kk])
        if idx.size == 0:
            c[kk, :] = 0.0
            continue
        i_idx = idx[:, 0]
        j_idx = idx[:, 1]
        z = (a[i_idx, :] * b[j_idx, :]) * sigma[None, :]
        y = t[i_idx, j_idx, kk]
        g = z.T @ z + ridge * np.eye(rank)
        rhs = z.T @ y
        c[kk, :] = np.linalg.solve(g, rhs)
    return c


def cp_als_observed_only(
    rclr_da: xr.DataArray,
    observed_mask: xr.DataArray,
    rank: int = 3,
    max_iter: int = 200,
    tol: float = 1e-5,
    init: str = "svd",
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, object]:
    t = rclr_da.values.astype(float)
    m = observed_mask.values.astype(bool)
    i, j, k = t.shape

    if rank <= 0:
        raise ValueError("rank must be > 0")
    if rank > min(i, j, k):
        warnings.warn(
            f"rank={rank} > min(shape)={min(i, j, k)} may be unstable.", RuntimeWarning
        )

    x_filled = np.where(m, t, 0.0)
    if init == "svd":
        a, b, c = _init_factors_svd(x_filled, rank, seed)
    else:
        a, b, c = _init_factors_random(t.shape, rank, seed)

    sigma = np.ones(rank, dtype=float)
    a, b, c, sigma = _normalize_factors(a, b, c, sigma)

    prev_loss = np.inf
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        a = _update_a(t, m, a, b, c, sigma)
        b = _update_b(t, m, a, b, c, sigma)
        c = _update_c(t, m, a, b, c, sigma)
        a, b, c, sigma = _normalize_factors(a, b, c, sigma)

        t_hat = _cp_reconstruct(a, b, c, sigma)
        loss = _observed_loss(t, m, t_hat)
        rel = abs(prev_loss - loss) / (abs(prev_loss) + 1e-12)

        if verbose:
            print(f"[fallback-ALS] iter={it:03d} loss={loss:.8e} rel_change={rel:.3e}")

        n_iter = it
        if np.isfinite(prev_loss) and rel < tol:
            converged = True
            break
        prev_loss = loss

    t_hat = _cp_reconstruct(a, b, c, sigma)
    final_loss = _observed_loss(t, m, t_hat)
    return {
        "T_hat": t_hat,
        "A": a,
        "B": b,
        "C": c,
        "sigma": sigma,
        "n_iter": n_iter,
        "final_loss": final_loss,
        "converged": converged,
    }


def _try_gemelli_path(*args, **kwargs) -> Optional[Dict[str, object]]:
    try:
        import gemelli  # noqa: F401
    except Exception:
        return None
    return None


def ctf_complete_tensor_from_log_da(
    da: xr.DataArray,
    observed_mask: Optional[xr.DataArray] = None,
    eps: float = 1e-3,
    rank: int = 3,
    max_iter: int = 200,
    tol: float = 1e-5,
    init: str = "svd",
    seed: int = 0,
    method: str = "auto",
    verbose: bool = True,
):
    _validate_da(da)
    if method not in {"auto", "gemelli", "fallback"}:
        raise ValueError("method must be one of {'auto','gemelli','fallback'}")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    raw = np.power(10.0, da.values.astype(float)) - eps
    raw[np.abs(raw) < 1e-12] = 0.0
    neg_count = int(np.sum(raw < 0))
    neg_ratio = neg_count / raw.size
    if verbose:
        print(f"[sanity] negative ratio in 10**da-eps before clip: {neg_ratio:.6f}")
    if neg_ratio > 0.05:
        warnings.warn(
            f"More than 5% negatives after 10**da-eps ({neg_ratio:.3%}). Check eps/log scale.",
            RuntimeWarning,
        )
    raw = np.where(raw < 0, 0.0, raw)

    da_ra_pct = xr.DataArray(raw, dims=da.dims, coords=da.coords, name="da_ra_pct")

    if observed_mask is None:
        rclr_da, observed_mask = rclr_transform_tensor(da_ra_pct)
    else:
        if observed_mask.dims != da.dims or observed_mask.shape != da.shape:
            raise ValueError("observed_mask dims/shape must match da")

        x = da_ra_pct.values.astype(float)
        m = observed_mask.values.astype(bool)
        r = np.full_like(x, np.nan, dtype=float)
        i, _, k = x.shape
        for ii in range(i):
            for kk in range(k):
                obs = m[ii, :, kk]
                if not np.any(obs):
                    continue
                v = x[ii, obs, kk]
                lv = np.log10(v)
                gm_log = np.mean(lv)
                r[ii, obs, kk] = lv - gm_log

        rclr_da = xr.DataArray(r, dims=da.dims, coords=da.coords, name="rclr_da")

    obs_ratio = float(observed_mask.values.mean())
    if verbose:
        print(f"[sanity] observed ratio in rclr tensor: {obs_ratio:.6f}")

    fit = None
    method_used = "fallback"
    if method in {"auto", "gemelli"}:
        fit = _try_gemelli_path(
            rclr_da=rclr_da,
            observed_mask=observed_mask,
            rank=rank,
            max_iter=max_iter,
            tol=tol,
            init=init,
            seed=seed,
            verbose=verbose,
        )
        if fit is not None:
            method_used = "gemelli"
        elif method == "gemelli" and verbose:
            print("[info] gemelli path unavailable/unstable, falling back to observed-only CP-ALS.")

    if fit is None:
        fit = cp_als_observed_only(
            rclr_da=rclr_da,
            observed_mask=observed_mask,
            rank=rank,
            max_iter=max_iter,
            tol=tol,
            init=init,
            seed=seed,
            verbose=verbose,
        )
        method_used = "fallback"

    t_hat = fit["T_hat"]
    ctf_rclr_hat = xr.DataArray(t_hat, dims=da.dims, coords=da.coords, name="ctf_rclr_hat")

    tmax = np.max(t_hat, axis=1, keepdims=True)
    w = np.exp(t_hat - tmax)
    wsum = np.sum(w, axis=1, keepdims=True)
    wsum = np.where(wsum <= 0, 1.0, wsum)
    p = w / wsum

    m = observed_mask.values.astype(bool)
    sample_time_observed = np.any(m, axis=1)
    no_obs = ~sample_time_observed
    if np.any(no_obs):
        j = da.sizes["taxon"]
        p[:, :, :] = np.where(no_obs[:, None, :], 1.0 / j, p)

    ctf_comp_pct_np = 100.0 * p
    ctf_comp_pct_np = np.where(ctf_comp_pct_np < 0, 0.0, ctf_comp_pct_np)
    ctf_comp_pct = xr.DataArray(ctf_comp_pct_np, dims=da.dims, coords=da.coords, name="ctf_comp_pct")

    ctf_imp_np = np.log10(ctf_comp_pct_np + eps)
    ctf_imp_np = np.where(np.isfinite(ctf_imp_np), ctf_imp_np, np.log10(eps))
    ctf_imp = xr.DataArray(ctf_imp_np, dims=da.dims, coords=da.coords, name="ctf_imp")

    sum_taxon = ctf_comp_pct.sum("taxon")
    if verbose:
        vals = sum_taxon.values.ravel()
        print(
            "[sanity] ctf_comp_pct.sum('taxon') summary: "
            f"min={np.nanmin(vals):.6f}, mean={np.nanmean(vals):.6f}, "
            f"max={np.nanmax(vals):.6f}, std={np.nanstd(vals):.6e}"
        )

    if not np.all(np.isfinite(ctf_imp.values)):
        raise RuntimeError("ctf_imp contains non-finite values.")

    return {
        "da_ra_pct": da_ra_pct,
        "rclr_da": rclr_da,
        "observed_mask": observed_mask,
        "ctf_rclr_hat": ctf_rclr_hat,
        "ctf_comp_pct": ctf_comp_pct,
        "ctf_imp": ctf_imp,
        "factors": {
            "A": fit["A"],
            "B": fit["B"],
            "C": fit["C"],
            "sigma": fit["sigma"],
        },
        "meta": {
            "method_used": method_used,
            "rank": rank,
            "n_iter": int(fit["n_iter"]),
            "final_loss": float(fit["final_loss"]),
            "converged": bool(fit["converged"]),
        },
    }


# =====================
# Mask-aware penalized 2D2-PCA
# =====================

def _bin_to_time_value(bin_label):
    s = str(bin_label)
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return float(s)


def _bin_key(x):
    return _bin_to_time_value(x)


def _format_frac(frac: float) -> str:
    return f"{float(frac):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _orthonormalize(m: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(m, mode="reduced")
    return q


def _top_eigvecs(s: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, s.shape[0])
    vals, vecs = np.linalg.eigh(s)
    idx = np.argsort(vals)[::-1][:k_eff]
    u = vecs[:, idx]
    return _orthonormalize(u)


def _build_time_weights(t: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    dt2 = (t[:, None] - t[None, :]) ** 2
    w = np.exp(-dt2 / (2.0 * sigma**2))
    np.fill_diagonal(w, 0.0)
    return w


def _method_group(method: str) -> str:
    return METHOD_GROUP_MAP.get(str(method), "other")


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
    elif mode == "all_finite":
        obs = finite
    elif mode == "all_finite_except_replace":
        if replace_mask is None:
            raise ValueError("replace_mask is required when mode='all_finite_except_replace'")
        obs = finite & (~replace_mask.values.astype(bool))
    elif mode == "positive_only_except_replace":
        if replace_mask is None:
            raise ValueError("replace_mask is required when mode='positive_only_except_replace'")
        obs = finite & (vals > zero_value) & (~replace_mask.values.astype(bool))
    else:
        raise ValueError(f"Unsupported observed mode: {mode}")

    return xr.DataArray(obs, coords=da_input.coords, dims=da_input.dims, name="observed_mask")


def penalized_2d2_pca(
    da: xr.DataArray,
    time=None,
    d: int = 1,
    q: int = 1,
    lam: float = 0.1,
    sigma: float = 4,
    T0: int = 30,
    T: int = 30,
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

    d_eff = min(d, n)
    q_eff = min(q, m)
    if d_eff <= 0 or q_eff <= 0:
        raise ValueError("effective d and q must be positive")

    tvals = np.array([_bin_to_time_value(x) for x in time], dtype=float)
    w_time = _build_time_weights(tvals, sigma=sigma)

    if floor_value is None:
        finite_values = a[np.isfinite(a)]
        if finite_values.size == 0:
            raise ValueError("Cannot infer floor_value from an entirely non-finite tensor")
        floor_value = float(np.min(finite_values))
    floor_value = float(floor_value)
    obs_count = obs.sum(axis=0)
    obs_sum = np.where(obs, a, 0.0).sum(axis=0)
    abar = np.full((m, n), floor_value, dtype=float)
    np.divide(obs_sum, obs_count, out=abar, where=obs_count > 0)
    ac = np.where(obs, a - abar[None, :, :], 0.0)

    z = _orthonormalize(rng.standard_normal((m, q_eff)))

    if verbose:
        print(f"[Phase 1] lambda=0, iter={T0}, M={m_time}, m={m}, n={n}, q={q_eff}, d={d_eff}")
    for _ in range(T0):
        zzt = z @ z.T
        astar0 = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astar0 += bi.T @ zzt @ bi
        astar0 /= max(m_time, 1)
        x = _top_eigvecs(astar0, d_eff)

        xxt = x @ x.T
        astarstar0 = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar0 += bi @ xxt @ bi.T
        astarstar0 /= max(m_time, 1)
        z = _top_eigvecs(astarstar0, q_eff)

    if verbose:
        print(f"[Phase 2] lambda={lam}, sigma={sigma}, iter={T}")
    for _ in range(T):
        zzt = z @ z.T
        astar = np.zeros((n, n), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astar += bi.T @ zzt @ bi
        astar /= max(m_time, 1)

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
        x = _top_eigvecs(astar, d_eff)

        xxt = x @ x.T
        astarstar = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar += bi @ xxt @ bi.T
        astarstar /= max(m_time, 1)

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
        z = _top_eigvecs(astarstar, q_eff)

    c = np.zeros((m_time, q_eff, d_eff), dtype=float)
    for i in range(m_time):
        ai_eff = np.where(obs[i], a[i], abar)
        c[i] = z.T @ ai_eff @ x

    meta = {
        "bin": time,
        "t": tvals,
        "W": w_time,
        "Abar": abar,
        "unsupported_trajectory_count": int(np.sum(obs_count == 0)),
        "floor_value": floor_value,
    }
    return x, z, c, meta


# =====================
# Evaluation utilities
# =====================

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


def _build_weighted_summary(df_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scope"]
    for key in ["frac", "repeat", "method", "method_group", "init_method"]:
        if key in df_metrics.columns:
            keys = [key] + keys

    for group_vals, g in df_metrics.groupby(keys):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        kv = dict(zip(keys, group_vals))
        w = g["n_eval"].to_numpy(dtype=float)
        rows.append(
            {
                **kv,
                "n_eval": int(np.nansum(w)),
                "mse_before": _weighted_avg(g["mse_before"], w),
                "mse_after": _weighted_avg(g["mse_after"], w),
                "mse_improve": _weighted_avg(g["mse_improve"], w),
                "pearson_before": _weighted_avg(g["pearson_before"], w),
                "pearson_after": _weighted_avg(g["pearson_after"], w),
                "pearson_improve": _weighted_avg(g["pearson_improve"], w),
                "wd_before": _weighted_avg(g["wd_before"], w),
                "wd_after": _weighted_avg(g["wd_after"], w),
                "wd_improve": _weighted_avg(g["wd_improve"], w),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["frac", "repeat", "method", "scope"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _summary_real(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(np.mean(v))
    if n < 2:
        return {"n": n, "mean": mean, "sd": np.nan, "sem": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    sd = float(np.std(v, ddof=1))
    sem = float(sd / np.sqrt(n))
    tcrit = float(student_t.ppf(0.975, df=n - 1))
    half = tcrit * sem
    return {"n": n, "mean": mean, "sd": sd, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def _summary_pearson_fisher(values):
    r = np.asarray(values, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    r_clip = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r_clip)
    mean_z = float(np.mean(z))
    mean_r = float(np.tanh(mean_z))

    if n < 2:
        return {"n": n, "mean": mean_r, "sd": np.nan, "sem": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    sd_r = float(np.std(r, ddof=1))
    sem_r = float(sd_r / np.sqrt(n))
    sd_z = float(np.std(z, ddof=1))
    sem_z = float(sd_z / np.sqrt(n))
    tcrit = float(student_t.ppf(0.975, df=n - 1))
    low_r = float(np.tanh(mean_z - tcrit * sem_z))
    high_r = float(np.tanh(mean_z + tcrit * sem_z))
    return {"n": n, "mean": mean_r, "sd": sd_r, "sem": sem_r, "ci95_low": low_r, "ci95_high": high_r}


def summarize_across_repeats(df_repeat: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mse_before", "mse_after", "mse_improve",
        "pearson_before", "pearson_after", "pearson_improve",
        "wd_before", "wd_after", "wd_improve",
    ]
    if df_repeat.empty:
        return pd.DataFrame(columns=["frac", "scope", "method", "metric", "n", "mean", "sd", "sem", "ci95_low", "ci95_high"])
    if "method" not in df_repeat.columns:
        df_repeat = df_repeat.copy()
        df_repeat["method"] = "penalized_2d2_pca"

    rows = []
    for (frac, scope, method), g in df_repeat.groupby(["frac", "scope", "method"]):
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            if metric in {"pearson_before", "pearson_after"}:
                stat = _summary_pearson_fisher(vals)
            else:
                stat = _summary_real(vals)
            rows.append({"frac": float(frac), "scope": scope, "method": method, "metric": metric, **stat})

    return pd.DataFrame(rows).sort_values(["frac", "scope", "method", "metric"]).reset_index(drop=True)


# =====================
# IO utilities
# =====================

def _atomic_to_csv(df: pd.DataFrame, out_path, index: bool = False) -> Path:
    """Write a CSV atomically so merge never observes a partially written file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    try:
        df.to_csv(tmp_path, index=index)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path


def _atomic_write_text(text: str, out_path, encoding: str = "utf-8") -> Path:
    """Write a small text/JSON marker atomically."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding=encoding)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path


def _repeat_key_from_path(path) -> Optional[Tuple[str, int]]:
    """Extract the formatted fraction and repeat number from a repeat artifact."""
    stem = Path(path).stem
    marker = "_frac_"
    repeat_marker = "_repeat_"
    if marker not in stem or repeat_marker not in stem:
        return None
    suffix = stem.split(marker, 1)[1]
    frac_str, repeat_str = suffix.rsplit(repeat_marker, 1)
    try:
        return frac_str, int(repeat_str)
    except ValueError:
        return None


def _completed_repeat_keys(outdir: Path) -> Optional[set[Tuple[str, int]]]:
    keys = {
        key
        for path in outdir.glob("repeat_complete_frac_*_repeat_*.json")
        if (key := _repeat_key_from_path(path)) is not None
    }
    if not keys and not (outdir / "run_config.json").exists():
        return None
    return keys


def _filter_completed_repeat_files(paths, completed_keys):
    """Use completion markers when present; preserve legacy merge behavior otherwise."""
    paths = list(paths)
    if completed_keys is None:
        return paths
    return [path for path in paths if _repeat_key_from_path(path) in completed_keys]


def _collect_repeat_files(outdir: Path, pattern: str, completed_keys):
    paths = sorted(_filter_completed_repeat_files(outdir.glob(pattern), completed_keys))
    if completed_keys is not None:
        present_keys = {_repeat_key_from_path(path) for path in paths}
        missing_keys = sorted(completed_keys - present_keys)
        if missing_keys:
            preview = ", ".join(f"{frac}/{repeat:03d}" for frac, repeat in missing_keys[:5])
            raise RuntimeError(
                f"Completed repeats are missing files for pattern '{pattern}': {preview}"
            )
    return paths


def _save_dataarray(da_to_save, var_name, out_path, save_format="netcdf", compress_level=4):
    out_path = Path(out_path)
    fmt = str(save_format).lower()
    ds = xr.Dataset({var_name: da_to_save})

    if fmt == "zarr" or out_path.suffix.lower() == ".zarr":
        zarr_path = out_path if out_path.suffix.lower() == ".zarr" else out_path.with_suffix(".zarr")
        ds.to_zarr(str(zarr_path), mode="w")
        return zarr_path

    nc_path = out_path if out_path.suffix.lower() == ".nc" else out_path.with_suffix(".nc")
    try:
        ds.to_netcdf(nc_path, encoding={var_name: {"zlib": True, "complevel": compress_level}})
        return nc_path
    except Exception:
        ds.to_netcdf(nc_path)
        return nc_path


def _load_dataarray(data_path, var_name="y_true"):
    data_path = Path(data_path)
    if data_path.suffix.lower() == ".zarr":
        ds = xr.open_zarr(str(data_path))
    else:
        ds = xr.open_dataset(data_path)
    if var_name in ds.data_vars:
        out = ds[var_name].load()
    else:
        out = ds[list(ds.data_vars)[0]].load()
    ds.close()
    return out


def inject_pseudo_group_effect_on_ctf_imp(
    ctf_imp: xr.DataArray,
    eps: float = 1e-3,
    seed: int = 2026,
    n_cd: int = 65,
    n_noncd: int = 65,
    diff_frac: float = 0.10,
    group_effect_mean: float = 0.8,
    group_effect_sd: float = 0.2,
    interaction_mean: float = 0.15,
    interaction_sd: float = 0.05,
    random_slope_sd: float = 0.05,
    noise_sd: float = 0.10,
    verbose: bool = True,
):
    """Inject pseudo-group effects on ctf_imp in log10(percent + eps) space."""
    _validate_da(ctf_imp)
    if not (0.0 <= float(diff_frac) <= 1.0):
        raise ValueError(f"diff_frac must be in [0, 1], got {diff_frac}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    rng = np.random.default_rng(seed)
    participants = np.asarray(ctf_imp.coords["participant"].values)
    taxa = np.asarray(ctf_imp.coords["taxon"].values)
    bins = np.asarray(ctf_imp.coords["bin"].values)
    i, j, k = ctf_imp.shape

    if i != (int(n_cd) + int(n_noncd)):
        raise ValueError(
            f"Participant count mismatch: I={i}, but n_cd + n_noncd = {int(n_cd) + int(n_noncd)}."
        )

    y0 = ctf_imp.values.astype(float)
    b = np.maximum(np.power(10.0, y0) - eps, 0.0)

    shuffled_idx = rng.permutation(i)
    idx_cd = shuffled_idx[: int(n_cd)]
    idx_noncd = shuffled_idx[int(n_cd): int(n_cd) + int(n_noncd)]
    x = np.zeros(i, dtype=int)
    x[idx_cd] = 1
    pseudo_group = np.full(i, "nonCD", dtype=object)
    pseudo_group[idx_cd] = "CD"

    n_diff = int(np.round(float(diff_frac) * j))
    n_diff = int(np.clip(n_diff, 0, j))
    diff_idx = (
        rng.choice(j, size=n_diff, replace=False)
        if n_diff > 0
        else np.array([], dtype=int)
    )
    is_diff = np.zeros(j, dtype=bool)
    is_diff[diff_idx] = True

    t_numeric = np.array([_bin_to_time_value(bi) for bi in bins], dtype=float)

    direction = np.zeros(j, dtype=int)
    a_k = np.zeros(j, dtype=float)
    h_k = np.zeros(j, dtype=float)
    sigma_k = np.zeros(j, dtype=float)
    c_ki = np.zeros((i, j), dtype=float)
    eta = np.zeros((i, j, k), dtype=float)

    if n_diff > 0:
        direction_diff = rng.choice(np.array([-1, 1], dtype=int), size=n_diff, replace=True)
        a_abs = np.clip(rng.normal(group_effect_mean, group_effect_sd, size=n_diff), 0.0, None)
        h_abs = np.clip(rng.normal(interaction_mean, interaction_sd, size=n_diff), 0.0, None)

        direction[diff_idx] = direction_diff
        a_k[diff_idx] = direction_diff * a_abs
        h_k[diff_idx] = direction_diff * h_abs
        sigma_k[diff_idx] = noise_sd

        c_ki[:, diff_idx] = rng.normal(0.0, random_slope_sd, size=(i, n_diff))
        eta[:, diff_idx, :] = rng.normal(0.0, noise_sd, size=(i, n_diff, k))

    x_3d = x[:, None, None].astype(float)
    t_3d = t_numeric[None, None, :]
    log_mult = (
        a_k[None, :, None] * x_3d
        + c_ki[:, :, None] * t_3d
        + h_k[None, :, None] * x_3d * t_3d
        + eta
    )

    b_tilde = b * np.power(10.0, log_mult)
    b_tilde = np.where(np.isfinite(b_tilde), b_tilde, 0.0)
    b_tilde = np.maximum(b_tilde, 0.0)

    y_star = np.log10(b_tilde + eps)
    y_star = np.where(np.isfinite(y_star), y_star, np.log10(eps))

    ctf_imp_effect = xr.DataArray(
        y_star,
        dims=ctf_imp.dims,
        coords=ctf_imp.coords,
        name="ctf_imp_effect",
    )

    df_pseudo_group = pd.DataFrame(
        {
            "participant": participants.astype(str),
            "pseudo_group": pseudo_group,
            "X": x.astype(int),
        }
    )
    df_taxon_effect = pd.DataFrame(
        {
            "taxon": taxa.astype(str),
            "taxon_index": np.arange(j, dtype=int),
            "is_differential": is_diff.astype(bool),
            "direction": direction.astype(int),
            "a_k": a_k.astype(float),
            "h_k": h_k.astype(float),
            "mean_abs_c_ki": np.mean(np.abs(c_ki), axis=0),
            "sigma_k": sigma_k.astype(float),
        }
    )
    effect_meta = {
        "seed": int(seed),
        "eps": float(eps),
        "diff_frac": float(diff_frac),
        "participants_cd": participants[idx_cd].astype(str).tolist(),
        "participants_noncd": participants[idx_noncd].astype(str).tolist(),
        "differential_taxa": taxa[diff_idx].astype(str).tolist(),
        "t_numeric": {str(bi): float(ti) for bi, ti in zip(bins, t_numeric)},
        "parameter_settings": {
            "n_cd": int(n_cd),
            "n_noncd": int(n_noncd),
            "group_effect_mean": float(group_effect_mean),
            "group_effect_sd": float(group_effect_sd),
            "interaction_mean": float(interaction_mean),
            "interaction_sd": float(interaction_sd),
            "random_slope_sd": float(random_slope_sd),
            "noise_sd": float(noise_sd),
        },
    }

    if verbose:
        print(f"[effect] participants={i}, taxa={j}, bins={k}")
        print("[effect] group counts:", df_pseudo_group["pseudo_group"].value_counts().to_dict())
        print(f"[effect] differential taxa: {int(is_diff.sum())}")
        pct_back = np.maximum(np.power(10.0, ctf_imp_effect.values) - eps, 0.0)
        sums = np.sum(pct_back, axis=1).ravel()
        print(
            "[effect sanity] sum over taxon (percent space) summary: "
            f"min={np.nanmin(sums):.6f}, mean={np.nanmean(sums):.6f}, "
            f"max={np.nanmax(sums):.6f}, std={np.nanstd(sums):.6e}"
        )

    return {
        "ctf_imp_effect": ctf_imp_effect,
        "df_pseudo_group": df_pseudo_group,
        "df_taxon_effect": df_taxon_effect,
        "effect_meta": effect_meta,
    }


def _effect_truth_path(outdir: Path, save_format: str) -> Path:
    return outdir / ("truth_effect.zarr" if str(save_format).lower() == "zarr" else "truth_effect.nc")


def _save_effect_artifacts(
    outdir,
    da_gt_effect: xr.DataArray,
    df_pseudo_group: pd.DataFrame,
    df_taxon_effect: pd.DataFrame,
    effect_meta: Dict[str, object],
    save_format: str = "netcdf",
    compress_level: int = 4,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_pseudo_group.to_csv(outdir / "effect_pseudo_group.csv", index=False)
    df_taxon_effect.to_csv(outdir / "effect_taxon_effect.csv", index=False)
    (outdir / "effect_meta.json").write_text(
        json.dumps(effect_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_dataarray(
        da_gt_effect,
        "y_true",
        _effect_truth_path(outdir, save_format),
        save_format=save_format,
        compress_level=compress_level,
    )


def _default_truth_path(
    outdir: Path,
    save_format: str,
    enable_effect: bool,
    effect_seed: int,
    n_cd: int,
    n_noncd: int,
    diff_frac: float,
    group_effect_mean: float,
    group_effect_sd: float,
    interaction_mean: float,
    interaction_sd: float,
    random_slope_sd: float,
    noise_sd: float,
) -> Path:
    ext = "zarr" if str(save_format).lower() == "zarr" else "nc"
    if not enable_effect:
        return outdir / f"truth_ctf.{ext}"
    effect_tag = (
        f"seed_{int(effect_seed)}"
        f"_cd_{int(n_cd)}"
        f"_noncd_{int(n_noncd)}"
        f"_diff_{float(diff_frac):.4f}"
        f"_gm_{float(group_effect_mean):.4f}"
        f"_gsd_{float(group_effect_sd):.4f}"
        f"_im_{float(interaction_mean):.4f}"
        f"_isd_{float(interaction_sd):.4f}"
        f"_rsd_{float(random_slope_sd):.4f}"
        f"_nsd_{float(noise_sd):.4f}"
    ).replace(".", "p")
    return outdir / f"truth_ctf_effect_{effect_tag}.{ext}"


# =====================
# Truth building and mask-aware imputation
# =====================

def build_truth_once(
    base_da: xr.DataArray,
    base_observed_mask: xr.DataArray,
    truth_path,
    force_rebuild=False,
    save_format="netcdf",
    compress_level=4,
    rank=3,
    method="fallback",
    ctf_seed=0,
    ctf_verbose=True,
    eps: float = 1e-3,
    enable_effect: bool = False,
    effect_seed: int = 2026,
    n_cd: int = 65,
    n_noncd: int = 65,
    diff_frac: float = 0.10,
    group_effect_mean: float = 0.8,
    group_effect_sd: float = 0.2,
    interaction_mean: float = 0.15,
    interaction_sd: float = 0.05,
    random_slope_sd: float = 0.05,
    noise_sd: float = 0.10,
    effect_artifact_dir=None,
    base_truth_path=None,
    require_existing: bool = False,
):
    truth_path = Path(truth_path)
    effect_artifact_dir = truth_path.parent if effect_artifact_dir is None else Path(effect_artifact_dir)
    if require_existing and not truth_path.is_file():
        raise FileNotFoundError(
            f"Frozen Sim3 truth does not exist: {truth_path}. "
            "Array shards are read-only and are not allowed to rebuild truth."
        )
    if truth_path.exists() and (not force_rebuild):
        da_gt = _load_dataarray(truth_path, var_name="y_true")
        da_gt.name = "y_true"
        print(f"[truth] loaded existing truth from {truth_path}")
        return da_gt
    if require_existing:
        raise RuntimeError("Internal error: required frozen truth was not loaded")

    if base_truth_path is not None:
        base_truth_path = Path(base_truth_path)
        if base_truth_path.resolve() == truth_path.resolve():
            raise ValueError("--base-truth-path and --truth-path must be different files")
        if not base_truth_path.is_file():
            raise FileNotFoundError(f"Frozen Sim2 base truth does not exist: {base_truth_path}")
        da_gt_raw = _load_dataarray(base_truth_path, var_name="y_true")
        _validate_da(da_gt_raw)
        if da_gt_raw.shape != base_da.shape:
            raise ValueError(
                f"Base truth shape {da_gt_raw.shape} does not match Sim3 input shape {base_da.shape}"
            )
        for dim in ("participant", "taxon", "bin"):
            if not np.array_equal(da_gt_raw.coords[dim].values, base_da.coords[dim].values):
                raise ValueError(f"Base truth coordinate mismatch on dimension {dim!r}")
        print(f"[truth] loaded shared Sim2 base truth from {base_truth_path}")
    else:
        out = ctf_complete_tensor_from_log_da(
            da=base_da,
            observed_mask=base_observed_mask,
            eps=eps,
            rank=rank,
            method=method,
            seed=ctf_seed,
            verbose=ctf_verbose,
        )
        da_gt_raw = out["ctf_imp"].copy(deep=True)
    da_gt_raw.name = "y_true"

    if enable_effect:
        effect_out = inject_pseudo_group_effect_on_ctf_imp(
            ctf_imp=da_gt_raw,
            eps=eps,
            seed=effect_seed,
            n_cd=n_cd,
            n_noncd=n_noncd,
            diff_frac=diff_frac,
            group_effect_mean=group_effect_mean,
            group_effect_sd=group_effect_sd,
            interaction_mean=interaction_mean,
            interaction_sd=interaction_sd,
            random_slope_sd=random_slope_sd,
            noise_sd=noise_sd,
            verbose=ctf_verbose,
        )
        da_gt = effect_out["ctf_imp_effect"].copy(deep=True)
        da_gt.name = "y_true"
        _save_effect_artifacts(
            outdir=effect_artifact_dir,
            da_gt_effect=da_gt,
            df_pseudo_group=effect_out["df_pseudo_group"],
            df_taxon_effect=effect_out["df_taxon_effect"],
            effect_meta=effect_out["effect_meta"],
            save_format=save_format,
            compress_level=compress_level,
        )
    else:
        da_gt = da_gt_raw
    da_gt.name = "y_true"

    truth_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = _save_dataarray(da_gt, "y_true", truth_path, save_format=save_format, compress_level=compress_level)
    print(f"[truth] built and saved truth to {saved_path}")
    return da_gt


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
    if idx.size == 0 or frac <= 0:
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


#
# Global fallback rules used uniformly by second-stage imputers:
# 1) spline uses cubic spline only when >= 4 observed points; otherwise it falls back to linear interpolation.
# 2) linear interpolation is only used for interior gaps; boundary gaps fall back to last/next logic.
# 3) temporal_knn uses all available neighbors up to k; if no usable neighbors exist it falls back to the trajectory mean.
# 4) dbn_dense, dbn_sparse, and glv always start from a mean-initialized complete tensor before dynamic refinement.
# 5) equal is a fixed-constant null model and therefore has no fallback.
#
def _trajectory_mean(obs_vals: np.ndarray, default_value: float) -> float:
    finite = obs_vals[np.isfinite(obs_vals)]
    if finite.size == 0:
        return float(default_value)
    return float(np.mean(finite))


def _trajectory_fill_constant(obs_vals: np.ndarray, statistic: str, default_value: float) -> float:
    finite = obs_vals[np.isfinite(obs_vals)]
    if finite.size == 0:
        return float(default_value)
    if statistic == "mean":
        return float(np.mean(finite))
    if statistic == "median":
        return float(np.median(finite))
    raise ValueError(f"Unsupported statistic: {statistic}")


def _fill_nearest_positive(
    times: np.ndarray,
    observed_idx: np.ndarray,
    observed_vals: np.ndarray,
    target_time: float,
    fallback_value: float,
) -> float:
    """Copy the closest original positive observation; ties use the earlier time."""
    if observed_idx.size == 0:
        return float(fallback_value)
    obs_times = times[observed_idx]
    distance = np.abs(obs_times - float(target_time))
    min_distance = float(np.min(distance))
    tied = np.flatnonzero(np.isclose(distance, min_distance, rtol=0.0, atol=1e-12))
    chosen = int(tied[np.argmin(obs_times[tied])])
    return float(observed_vals[chosen])


def _fill_last_value(times: np.ndarray, observed_idx: np.ndarray, observed_vals: np.ndarray, target_time: float, fallback_value: float) -> float:
    if observed_idx.size == 0:
        return float(fallback_value)
    obs_times = times[observed_idx]
    left = np.flatnonzero(obs_times <= target_time)
    if left.size > 0:
        return float(observed_vals[left[-1]])
    right = np.flatnonzero(obs_times > target_time)
    if right.size > 0:
        return float(observed_vals[right[0]])
    return float(fallback_value)


def _fill_weighted_time_average(
    times: np.ndarray,
    observed_idx: np.ndarray,
    observed_vals: np.ndarray,
    target_time: float,
    fallback_value: float,
) -> float:
    if observed_idx.size == 0:
        return float(fallback_value)
    obs_times = times[observed_idx]
    left = np.flatnonzero(obs_times <= target_time)
    right = np.flatnonzero(obs_times >= target_time)
    if left.size > 0 and right.size > 0:
        li = left[-1]
        ri = right[0]
        lt = float(obs_times[li])
        rt = float(obs_times[ri])
        lv = float(observed_vals[li])
        rv = float(observed_vals[ri])
        if np.isclose(lt, rt):
            return lv
        w_left = 1.0 / max(abs(target_time - lt), 1e-8)
        w_right = 1.0 / max(abs(rt - target_time), 1e-8)
        return float((w_left * lv + w_right * rv) / (w_left + w_right))
    if left.size > 0:
        return float(observed_vals[left[-1]])
    if right.size > 0:
        return float(observed_vals[right[0]])
    return float(fallback_value)


def _fill_temporal_knn(
    times: np.ndarray,
    observed_idx: np.ndarray,
    observed_vals: np.ndarray,
    target_time: float,
    fallback_value: float,
    k: int,
    kernel_sigma: float,
) -> float:
    if observed_idx.size == 0:
        return float(fallback_value)
    obs_times = times[observed_idx]
    dist = np.abs(obs_times - float(target_time))
    order = np.argsort(dist)[: max(int(k), 1)]
    nn_dist = dist[order]
    nn_vals = observed_vals[order]
    sigma = max(float(kernel_sigma), 1e-8)
    w = np.exp(-(nn_dist**2) / (2.0 * sigma**2))
    if np.all(w <= 0):
        return float(np.mean(nn_vals))
    return float(np.sum(w * nn_vals) / np.sum(w))


def _fill_spline_or_linear(
    times: np.ndarray,
    observed_idx: np.ndarray,
    observed_vals: np.ndarray,
    target_time: float,
    fallback_value: float,
    min_points: int,
) -> float:
    if observed_idx.size == 0:
        return float(fallback_value)
    obs_times = times[observed_idx]
    if observed_idx.size >= max(int(min_points), 4):
        try:
            cs = CubicSpline(obs_times, observed_vals, extrapolate=False)
            pred = float(cs(target_time))
            if np.isfinite(pred):
                return pred
        except Exception:
            pass
    if observed_idx.size >= 2:
        left = np.flatnonzero(obs_times <= target_time)
        right = np.flatnonzero(obs_times >= target_time)
        if left.size > 0 and right.size > 0:
            li = left[-1]
            ri = right[0]
            lt = float(obs_times[li])
            rt = float(obs_times[ri])
            lv = float(observed_vals[li])
            rv = float(observed_vals[ri])
            if np.isclose(lt, rt):
                return lv
            return float(lv + (rv - lv) * ((target_time - lt) / (rt - lt)))
    return _fill_last_value(times, observed_idx, observed_vals, target_time, fallback_value)


def _impute_local_trajectory(
    y: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    method: str,
    params: Dict,
    zero_value: float,
    fallback_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = y.copy()
    # Donors must be original, unmasked positive observations.  Values filled
    # earlier in this loop never become donors, so chained floor propagation is
    # impossible.
    obs_idx = np.flatnonzero((~mask) & np.isfinite(y) & (y > zero_value))
    obs_vals = y[obs_idx]
    mean_fallback = _trajectory_mean(obs_vals, default_value=zero_value)
    fill_value = params.get("fill_value", zero_value)
    for idx in np.flatnonzero(mask):
        t = float(times[idx])
        target_fallback = (
            float(fallback_values[idx])
            if fallback_values is not None and np.isfinite(fallback_values[idx])
            else float(zero_value)
        )
        if method == "mean":
            out[idx] = _trajectory_fill_constant(obs_vals, "mean", default_value=target_fallback)
        elif method == "nearest_positive":
            out[idx] = _fill_nearest_positive(
                times, obs_idx, obs_vals, t, fallback_value=target_fallback
            )
        elif method == "median":
            out[idx] = _trajectory_fill_constant(obs_vals, "median", default_value=zero_value)
        elif method == "last":
            out[idx] = _fill_last_value(times, obs_idx, obs_vals, t, fallback_value=mean_fallback)
        elif method == "weighted_time_average":
            out[idx] = _fill_weighted_time_average(times, obs_idx, obs_vals, t, fallback_value=mean_fallback)
        elif method == "temporal_knn":
            out[idx] = _fill_temporal_knn(
                times,
                obs_idx,
                obs_vals,
                t,
                fallback_value=target_fallback,
                k=int(params.get("k", 3)),
                kernel_sigma=float(params.get("kernel_sigma", 4.0)),
            )
        elif method == "spline":
            out[idx] = _fill_spline_or_linear(
                times,
                obs_idx,
                obs_vals,
                t,
                fallback_value=mean_fallback,
                min_points=int(params.get("min_points", 4)),
            )
        elif method == "equal":
            out[idx] = float(fill_value)
        else:
            raise ValueError(f"Unsupported local trajectory method: {method}")
    return out


def _build_taxon_positive_fallbacks(
    values: np.ndarray,
    replace_mask: np.ndarray,
    times: np.ndarray,
    zero_value: float,
) -> np.ndarray:
    """Configuration B fallback for trajectories with no positive donor.

    At every target time, use the same taxon's positive values from other
    participants at the closest supported time.  Ties use the earlier time and
    donors are summarized by their median.  A wholly unsupported taxon remains
    at log-floor.
    """
    observed = (~replace_mask) & np.isfinite(values) & (values > zero_value)
    fallback = np.full((values.shape[1], values.shape[2]), float(zero_value), dtype=float)
    for taxon_idx in range(values.shape[1]):
        supported_times = np.flatnonzero(np.any(observed[:, taxon_idx, :], axis=0))
        if supported_times.size == 0:
            continue
        for target_idx in range(values.shape[2]):
            distance = np.abs(times[supported_times] - times[target_idx])
            min_distance = float(np.min(distance))
            tied = supported_times[np.isclose(distance, min_distance, rtol=0.0, atol=1e-12)]
            donor_time = int(tied[np.argmin(times[tied])])
            donor = values[:, taxon_idx, donor_time][observed[:, taxon_idx, donor_time]]
            fallback[taxon_idx, target_idx] = float(np.median(donor))
    return fallback


def _mean_initialize_tensor(data: np.ndarray, replace_mask: np.ndarray, zero_value: float) -> np.ndarray:
    init = data.copy()
    n_p, n_t, _ = init.shape
    for p in range(n_p):
        for j in range(n_t):
            y = init[p, j, :]
            mk = replace_mask[p, j, :]
            if np.any(mk):
                fill = _trajectory_fill_constant(y[~mk], "mean", default_value=zero_value)
                y_new = y.copy()
                y_new[mk] = fill
                init[p, j, :] = y_new
    return init


def _apply_candidate_only_on_mask(
    da_input: xr.DataArray,
    candidate_da: xr.DataArray,
    replace_mask: xr.DataArray,
) -> xr.DataArray:
    """
    Write candidate values back only on masked cells with finite predictions.

    The simulation task remains random cell-mask imputation. Legacy methods may
    internally construct a full day-vector candidate, but only replace_mask=True
    cells are updated in the returned tensor.
    """
    _validate_da(da_input)
    _validate_da(candidate_da)
    _validate_da(replace_mask)
    out = da_input.values.astype(float).copy()
    cand = candidate_da.values.astype(float)
    mask = replace_mask.values.astype(bool)
    use = mask & np.isfinite(cand)
    out[use] = cand[use]
    return xr.DataArray(out, coords=da_input.coords, dims=da_input.dims, name="log_abundance_imputed")


def _normalize_relative_vector(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=float).copy()
    out = np.where(np.isfinite(out), np.maximum(out, 0.0), np.nan)
    if not np.all(np.isfinite(out)):
        return np.full(out.shape, np.nan, dtype=float)
    total = float(np.sum(out))
    if total <= 0:
        return np.full(out.shape, np.nan, dtype=float)
    return out / total


def _abundance_to_relative_da(
    da_abund: xr.DataArray,
    name: str = "relative_abundance",
) -> xr.DataArray:
    """
    Convert abundance / percent space to participant-bin relative abundance.

    This is used by legacy KNN and legacy gLV-family comparators, whose
    reference implementations operate on vectors summing to 1 across taxa.
    """
    _validate_da(da_abund)
    vals = da_abund.values.astype(float)
    vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), np.nan)
    denom = np.sum(vals, axis=1, keepdims=True)
    rel = np.divide(vals, denom, out=np.full_like(vals, np.nan, dtype=float), where=denom > 0)
    return xr.DataArray(rel, coords=da_abund.coords, dims=da_abund.dims, name=name)


def _abundance_vector_to_log_scale(vec: np.ndarray, eps: float) -> np.ndarray:
    out = np.asarray(vec, dtype=float).copy()
    finite = np.isfinite(out)
    out = np.where(finite, np.maximum(out, 0.0), np.nan)
    out = np.where(finite, np.log10(out + float(eps)), np.nan)
    return out


def _relative_vector_to_log_scale(vec: np.ndarray, eps: float) -> np.ndarray:
    rel = _normalize_relative_vector(vec)
    if not np.all(np.isfinite(rel)):
        return np.full(rel.shape, np.nan, dtype=float)
    return _abundance_vector_to_log_scale(100.0 * rel, eps=eps)


def _mean_initialize_relative_tensor(
    rel_data: np.ndarray,
    replace_mask: np.ndarray,
) -> np.ndarray:
    """
    Mean-initialize masked cells for legacy dynamic models in relative space.

    The external task still masks random individual cells only. This helper is
    strictly an internal modeling step to construct complete taxa x time
    trajectories before fitting legacy gLV-family models.
    """
    init = np.asarray(rel_data, dtype=float).copy()
    n_p, n_j, n_t = init.shape
    for p in range(n_p):
        for j in range(n_j):
            y = init[p, j, :]
            mk = replace_mask[p, j, :]
            if np.any(mk):
                finite_obs = y[(~mk) & np.isfinite(y)]
                fill = float(np.mean(finite_obs)) if finite_obs.size > 0 else 0.0
                y_new = y.copy()
                y_new[mk] = fill
                init[p, j, :] = y_new
        for t in range(n_t):
            init[p, :, t] = _normalize_relative_vector(init[p, :, t])
    return init


def _legacy_weighted_avg_candidate(
    abund_pt: np.ndarray,
    mask_pt: np.ndarray,
    time_vals: np.ndarray,
    target_idx: int,
) -> np.ndarray:
    n_taxa, n_time = abund_pt.shape
    if target_idx <= 0 or target_idx >= n_time - 1:
        return np.full(n_taxa, np.nan, dtype=float)
    prev_idx = target_idx - 1
    next_idx = target_idx + 1
    prev_t = float(time_vals[prev_idx])
    cur_t = float(time_vals[target_idx])
    next_t = float(time_vals[next_idx])
    if not (next_t > prev_t):
        return np.full(n_taxa, np.nan, dtype=float)

    prev_vec = abund_pt[:, prev_idx]
    next_vec = abund_pt[:, next_idx]
    valid = (~mask_pt[:, prev_idx]) & (~mask_pt[:, next_idx]) & np.isfinite(prev_vec) & np.isfinite(next_vec)
    out = np.full(n_taxa, np.nan, dtype=float)
    out[valid] = (
        next_vec[valid] * (cur_t - prev_t) + prev_vec[valid] * (next_t - cur_t)
    ) / (next_t - prev_t)
    return out


def _legacy_spline_candidate(
    abund_pt: np.ndarray,
    mask_pt: np.ndarray,
    time_vals: np.ndarray,
    target_idx: int,
    min_points: int,
) -> np.ndarray:
    n_taxa = abund_pt.shape[0]
    target_time = float(time_vals[target_idx])
    out = np.full(n_taxa, np.nan, dtype=float)
    for j in range(n_taxa):
        obs = (~mask_pt[j, :]) & np.isfinite(abund_pt[j, :])
        obs[target_idx] = False
        if int(np.sum(obs)) < max(int(min_points), 4):
            continue
        try:
            spline = interp1d(
                time_vals[obs],
                abund_pt[j, obs],
                kind="cubic",
            )
            pred = float(spline(target_time))
        except Exception:
            continue
        if np.isfinite(pred):
            out[j] = max(0.0, pred)
    return out


def _legacy_knn_epanechnikov_candidate(
    rel_pt: np.ndarray,
    mask_pt: np.ndarray,
    time_vals: np.ndarray,
    target_idx: int,
    k: int,
) -> np.ndarray:
    target_time = float(time_vals[target_idx])
    full_cols = (~np.any(mask_pt, axis=0)) & np.all(np.isfinite(rel_pt), axis=0)
    full_cols[target_idx] = False
    train_idx = np.flatnonzero(full_cols)
    if train_idx.size <= int(k):
        return np.full(rel_pt.shape[0], np.nan, dtype=float)

    dist = np.abs(time_vals[train_idx] - target_time)
    dist_sorted = np.sort(dist)
    bandwidth = float(dist_sorted[int(k)])
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return np.full(rel_pt.shape[0], np.nan, dtype=float)

    acc = np.zeros(rel_pt.shape[0], dtype=float)
    used = False
    for idx in train_idx:
        cur_dist = abs(float(time_vals[idx]) - target_time)
        if cur_dist <= bandwidth:
            ker_norm = cur_dist / bandwidth
            kernel = 0.75 * (1.0 - ker_norm**2)
            if kernel > 0:
                acc += kernel * rel_pt[:, idx]
                used = True
    if not used:
        return np.full(rel_pt.shape[0], np.nan, dtype=float)
    return _normalize_relative_vector(acc)


def _apply_legacy_day_vector_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    method: str,
    params: Dict,
    eps: float,
) -> xr.DataArray:
    """
    Apply a legacy day-vector comparator under the current random cell-mask task.

    Legacy local-time comparators may generate a full participant-bin candidate
    vector, but only masked cells are written back. Unmasked entries remain
    unchanged.
    """
    da_abund = to_abundance_space(da_input, eps=eps, name="legacy_abundance_input")
    da_rel = _abundance_to_relative_da(da_abund, name="legacy_relative_input")
    abund_arr = da_abund.transpose("participant", "taxon", "bin").values.astype(float)
    rel_arr = da_rel.transpose("participant", "taxon", "bin").values.astype(float)
    mask = replace_mask.transpose("participant", "taxon", "bin").values.astype(bool)
    time_vals = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
    cand_log = np.full_like(abund_arr, np.nan, dtype=float)

    for p in range(abund_arr.shape[0]):
        abund_pt = abund_arr[p]
        rel_pt = rel_arr[p]
        mask_pt = mask[p]
        target_bins = np.flatnonzero(np.any(mask_pt, axis=0))
        for target_idx in target_bins:
            if method == "weighted_avg_legacy":
                cand = _legacy_weighted_avg_candidate(
                    abund_pt=abund_pt,
                    mask_pt=mask_pt,
                    time_vals=time_vals,
                    target_idx=target_idx,
                )
                cand_log[p, :, target_idx] = _abundance_vector_to_log_scale(cand, eps=eps)
            elif method == "spline_legacy":
                cand = _legacy_spline_candidate(
                    abund_pt=abund_pt,
                    mask_pt=mask_pt,
                    time_vals=time_vals,
                    target_idx=target_idx,
                    min_points=int(params.get("min_points", 4)),
                )
                cand_log[p, :, target_idx] = _abundance_vector_to_log_scale(cand, eps=eps)
            elif method == "knn_epanechnikov_legacy":
                cand = _legacy_knn_epanechnikov_candidate(
                    rel_pt=rel_pt,
                    mask_pt=mask_pt,
                    time_vals=time_vals,
                    target_idx=target_idx,
                    k=max(int(params.get("k", 3)), 1),
                )
                cand_log[p, :, target_idx] = _relative_vector_to_log_scale(cand, eps=eps)
            else:
                raise ValueError(f"Unsupported legacy local-time method: {method}")

    candidate_da = xr.DataArray(
        np.transpose(cand_log, (0, 1, 2)),
        coords=da_input.coords,
        dims=da_input.dims,
        name="legacy_candidate_log",
    )
    return _apply_candidate_only_on_mask(da_input, candidate_da, replace_mask)


def _legacy_create_y_matrix(X: np.ndarray, pert_incidence: Optional[np.ndarray] = None) -> np.ndarray:
    ones = np.ones((1, X.shape[1] - 1), dtype=float)
    y_mat = np.concatenate((X[:, 1:], ones), axis=0)
    if pert_incidence is not None:
        y_mat = np.concatenate((y_mat, pert_incidence), axis=0)
    return y_mat


def _legacy_create_F_matrix(X: np.ndarray, time_points: np.ndarray) -> np.ndarray:
    cols = []
    for idx in range(X.shape[1] - 1):
        dt = float(time_points[idx + 1] - time_points[idx])
        if dt <= 0:
            raise ValueError("time_points must be strictly increasing for legacy gLV methods")
        ratio = np.divide(X[:, idx + 1], X[:, idx], out=np.ones(X.shape[0], dtype=float), where=X[:, idx] > 0)
        cur = np.log(ratio).reshape(X.shape[0], 1)
        cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0) / dt
        cols.append(cur)
    if not cols:
        return np.zeros((X.shape[0], 0), dtype=float)
    return np.concatenate(cols, axis=1)


def _legacy_mse_glv(F: np.ndarray, Y_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    tmp = np.linalg.solve(Y_mat @ Y_mat.T, Y_mat @ F.T)
    m = F.shape[0]
    M = tmp[:m, :].T
    mue = tmp[m: m + 1, :].reshape(1, tmp.shape[1]).T
    pert_coef = tmp[m + 1:, :].T if Y_mat.shape[0] > m + 1 else None
    return M, mue, pert_coef


def _legacy_coef_matrix_by_paper_trick_ridge(
    Y: np.ndarray,
    F: np.ndarray,
    lambda_M: float,
    lambda_mue: float,
    lambda_E: float,
) -> np.ndarray:
    num_of_spec = F.shape[0]
    num_of_pert = Y.shape[0] - num_of_spec - 1
    diag_vals = [float(lambda_M)] * num_of_spec + [float(lambda_mue)] + [float(lambda_E)] * max(num_of_pert, 0)
    D = np.diagflat(diag_vals)
    return F @ (Y.T @ np.linalg.inv(Y @ Y.T + D))


def _legacy_parse_ridge_coef_matrix(coef_mat: np.ndarray, num_taxa: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    M = coef_mat[:, :num_taxa]
    mue = coef_mat[:, num_taxa:num_taxa + 1]
    pert = coef_mat[:, num_taxa + 1:] if coef_mat.shape[1] > num_taxa + 1 else None
    return M, mue, pert


def _safe_linear_coef_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _legacy_limits(
    F: np.ndarray,
    data: np.ndarray,
    error_threshold: float,
    repeats: int,
    med_or_avg: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(0)
    n_taxa, n_samples = F.shape
    if n_samples < 2:
        raise ValueError("LIMITS requires at least two transition samples")

    all_M = []
    all_mue = []
    for _ in range(max(int(repeats), 1)):
        perm = rng.permutation(n_samples)
        split = max(1, n_samples // 2)
        split = min(split, n_samples - 1)
        train_nums = np.sort(perm[:split])
        test_nums = np.sort(perm[split:])
        if train_nums.size == 0 or test_nums.size == 0:
            raise ValueError("LIMITS train/test split failed")

        train_F = F[:, train_nums]
        train_data = data[:, train_nums]
        test_F = F[:, test_nums]
        test_data = data[:, test_nums]
        tmp_mue = np.zeros((n_taxa, 1), dtype=float)
        tmp_M = np.zeros((n_taxa, n_taxa), dtype=float)

        for main_spec in range(n_taxa):
            cur_train_F = train_F[main_spec, :].reshape(-1, 1).copy()
            inactive = list(range(n_taxa))

            ones_train = np.ones(cur_train_F.shape[0], dtype=float)
            tmp_mue[main_spec, 0] = _safe_linear_coef_no_intercept(ones_train, cur_train_F[:, 0])
            cur_train_F[:, 0] -= tmp_mue[main_spec, 0]

            cur_test_pred = np.ones((test_F.shape[1], 1), dtype=float) * tmp_mue[main_spec, 0]

            inactive.remove(main_spec)
            main_train = train_data[main_spec, :].astype(float)
            tmp_M[main_spec, main_spec] = _safe_linear_coef_no_intercept(main_train, cur_train_F[:, 0])
            cur_train_F[:, 0] -= main_train * tmp_M[main_spec, main_spec]

            main_test = test_data[main_spec, :].reshape(-1, 1)
            cur_test_pred = cur_test_pred + main_test * tmp_M[main_spec, main_spec]
            error = float(np.mean((cur_test_pred[:, 0] - test_F[main_spec, :]) ** 2))

            improve_model = True
            while improve_model:
                best_error = np.inf
                best_spec = None
                best_coeff = 0.0
                for spec_index in inactive:
                    cur_train = train_data[spec_index, :].astype(float)
                    coeff = _safe_linear_coef_no_intercept(cur_train, cur_train_F[:, 0])
                    tmp_test_pred = cur_test_pred[:, 0] + test_data[spec_index, :] * coeff
                    tmp_error = float(np.mean((tmp_test_pred - test_F[main_spec, :]) ** 2))
                    if tmp_error < best_error:
                        best_error = tmp_error
                        best_spec = spec_index
                        best_coeff = coeff

                if best_spec is None:
                    break
                improve_ratio = np.inf if error <= 0 else (error - best_error) / error
                if improve_ratio >= float(error_threshold):
                    tmp_M[main_spec, best_spec] = best_coeff
                    inactive.remove(best_spec)
                    cur_train_F[:, 0] -= train_data[best_spec, :] * best_coeff
                    cur_test_pred[:, 0] += test_data[best_spec, :] * best_coeff
                    error = best_error
                else:
                    improve_model = False
                if not inactive:
                    improve_model = False

        all_mue.append(tmp_mue)
        all_M.append(tmp_M)

    stack_M = np.stack(all_M, axis=0)
    stack_mue = np.stack(all_mue, axis=0)
    if int(med_or_avg) == 0:
        final_M = np.median(stack_M, axis=0)
        final_mue = np.median(stack_mue, axis=0)
    else:
        final_M = np.mean(stack_M, axis=0)
        final_mue = np.mean(stack_mue, axis=0)
    return final_mue, final_M


def _legacy_predict_abundances_day_by_day(
    x: np.ndarray,
    delta_t: float,
    self_growth: np.ndarray,
    interaction_matrix: np.ndarray,
    perturbation_matrix: Optional[np.ndarray] = None,
    perturbation_incidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Predict relative abundances using the reference gLV recursion.

    Perturbations are intentionally unsupported here; the legacy comparator
    branch currently implements the no-perturbation case only.
    """
    if perturbation_matrix is not None or perturbation_incidence is not None:
        raise NotImplementedError("Legacy comparator methods currently support only pert=None")

    x_t = _normalize_relative_vector(np.asarray(x, dtype=float).reshape(-1))
    if not np.all(np.isfinite(x_t)):
        raise ValueError("Initial relative abundance vector is not finite")
    # The reference routine advances one discrete day at a time. Our bin labels
    # are converted by _bin_to_time_value and may be non-integer, so we use the
    # nearest positive integer number of forward steps here.
    steps = max(int(np.round(float(delta_t))), 1)
    for _ in range(steps):
        delta_x = self_growth[:, 0] + interaction_matrix @ x_t
        next_x = np.exp(delta_x) * x_t
        next_x = np.maximum(next_x, 0.0)
        x_t = _normalize_relative_vector(next_x)
        if not np.all(np.isfinite(x_t)):
            raise ValueError("Legacy gLV prediction produced a non-finite abundance vector")
    return x_t.reshape(-1, 1)


def _apply_legacy_glv_family_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    method: str,
    params: Dict,
    eps: float,
    model_seed: int,
) -> xr.DataArray:
    """
    Apply legacy gLV-family comparators under random cell-mask evaluation.

    These methods internally build a complete relative-abundance tensor using
    mean initialization, fit one model per participant, and may generate full
    day-vector predictions. Only masked cells are written back.
    """
    da_abund = to_abundance_space(da_input, eps=eps, name="legacy_glv_abundance_input")
    da_rel = _abundance_to_relative_da(da_abund, name="legacy_glv_relative_input")
    rel_arr = da_rel.transpose("participant", "taxon", "bin").values.astype(float)
    mask = replace_mask.transpose("participant", "taxon", "bin").values.astype(bool)
    init_rel = _mean_initialize_relative_tensor(rel_arr, replace_mask=mask)
    time_vals = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
    cand_log = np.full_like(rel_arr, np.nan, dtype=float)
    rng = np.random.default_rng(int(model_seed))

    for p in range(rel_arr.shape[0]):
        X_init = init_rel[p]
        participant_mask = mask[p]
        if X_init.shape[1] < 2:
            continue
        try:
            F = _legacy_create_F_matrix(X_init, time_vals)
            if method == "glv_mse_legacy":
                Y = _legacy_create_y_matrix(X_init)
                M, mue, pert = _legacy_mse_glv(F, Y)
            elif method == "glv_ridge_legacy":
                Y = _legacy_create_y_matrix(X_init)
                coef = _legacy_coef_matrix_by_paper_trick_ridge(
                    Y=Y,
                    F=F,
                    lambda_M=float(params.get("lambda_M", 1e-3)),
                    lambda_mue=float(params.get("lambda_mue", 1e-3)),
                    lambda_E=float(params.get("lambda_E", 1e-3)),
                )
                M, mue, pert = _legacy_parse_ridge_coef_matrix(coef_mat=coef, num_taxa=X_init.shape[0])
            elif method == "limits_legacy":
                data = X_init[:, :-1]
                mue, M = _legacy_limits(
                    F=F,
                    data=data,
                    error_threshold=float(params.get("error_threshold", 0.05)),
                    repeats=int(params.get("repeats", 20)),
                    med_or_avg=int(params.get("med_or_avg", 0)),
                    rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
                )
                pert = None
            else:
                raise ValueError(f"Unsupported legacy gLV-family method: {method}")

            for target_idx in np.flatnonzero(np.any(participant_mask, axis=0)):
                if target_idx <= 0:
                    continue
                delta_t = float(time_vals[target_idx] - time_vals[target_idx - 1])
                if delta_t <= 0:
                    continue
                pred = _legacy_predict_abundances_day_by_day(
                    x=X_init[:, target_idx - 1],
                    delta_t=delta_t,
                    self_growth=mue,
                    interaction_matrix=M,
                    perturbation_matrix=pert,
                    perturbation_incidence=None,
                )
                cand_log[p, :, target_idx] = _relative_vector_to_log_scale(pred[:, 0], eps=eps)
        except Exception:
            continue

    candidate_da = xr.DataArray(cand_log, coords=da_input.coords, dims=da_input.dims, name="legacy_glv_candidate_log")
    return _apply_candidate_only_on_mask(da_input, candidate_da, replace_mask)


def _solve_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    XtX = X.T @ X
    XtY = X.T @ Y
    coef = np.linalg.solve(XtX + float(alpha) * np.eye(X.shape[1]), XtY)
    pred = X @ coef
    resid = Y - pred
    intercept = np.mean(resid, axis=0)
    return intercept, coef


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _fit_lasso_cd(X: np.ndarray, y: np.ndarray, alpha: float, n_iter: int) -> np.ndarray:
    coef = np.zeros(X.shape[1], dtype=float)
    col_norm = np.sum(X * X, axis=0) + 1e-8
    for _ in range(max(int(n_iter), 1)):
        for j in range(X.shape[1]):
            r = y - X @ coef + X[:, j] * coef[j]
            rho = float(np.dot(X[:, j], r))
            coef[j] = _soft_threshold(rho, float(alpha)) / col_norm[j]
    return coef


def _fit_sparse_multioutput(X: np.ndarray, Y: np.ndarray, alpha: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    coef = np.zeros((X.shape[1], Y.shape[1]), dtype=float)
    intercept = np.zeros(Y.shape[1], dtype=float)
    for k in range(Y.shape[1]):
        y = Y[:, k]
        y_center = y - np.mean(y)
        beta = _fit_lasso_cd(X, y_center, alpha=alpha, n_iter=n_iter)
        coef[:, k] = beta
        intercept[k] = float(np.mean(y - X @ beta))
    return intercept, coef


def _fit_dynamic_next_step(init_btj: np.ndarray, time_vals: np.ndarray, sparse: bool, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    n_p, n_t, n_j = init_btj.shape
    if n_t < 2:
        return np.zeros(n_j, dtype=float), np.zeros((n_j + 1, n_j), dtype=float)
    X_rows, Y_rows = [], []
    for p in range(n_p):
        for t in range(n_t - 1):
            dt = float(time_vals[t + 1] - time_vals[t])
            feat = np.concatenate(([dt], init_btj[p, t, :]))
            X_rows.append(feat)
            Y_rows.append(init_btj[p, t + 1, :])
    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)
    if sparse:
        return _fit_sparse_multioutput(
            X=X,
            Y=Y,
            alpha=float(params.get("alpha", 0.05)),
            n_iter=int(params.get("n_iter", 150)),
        )
    return _solve_ridge(X=X, Y=Y, alpha=float(params.get("alpha", 1.0)))


def _fit_glv_style(init_btj: np.ndarray, time_vals: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    n_p, n_t, n_j = init_btj.shape
    if n_t < 2:
        return np.zeros(n_j, dtype=float), np.zeros((n_j, n_j), dtype=float)
    X_rows, Y_rows = [], []
    for p in range(n_p):
        for t in range(n_t - 1):
            dt = float(time_vals[t + 1] - time_vals[t])
            if dt <= 0:
                continue
            X_rows.append(init_btj[p, t, :])
            Y_rows.append((init_btj[p, t + 1, :] - init_btj[p, t, :]) / dt)
    if not X_rows:
        return np.zeros(n_j, dtype=float), np.zeros((n_j, n_j), dtype=float)
    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)
    reg = float(params.get("alpha", 0.5))
    coef = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ Y)
    intercept = np.mean(Y - X @ coef, axis=0)
    return intercept, coef


def _impute_dynamic_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    method: str,
    params: Dict,
    zero_value: float,
) -> xr.DataArray:
    arr = da_input.transpose("participant", "taxon", "bin").values.astype(float)
    mask = replace_mask.transpose("participant", "taxon", "bin").values.astype(bool)
    time_vals = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
    init = _mean_initialize_tensor(arr, mask, zero_value=zero_value)
    init_btj = np.transpose(init, (0, 2, 1))
    mask_btj = np.transpose(mask, (0, 2, 1))
    pred_full = init.copy()

    if method in {"dbn_dense", "dbn_sparse"}:
        intercept, coef = _fit_dynamic_next_step(
            init_btj=init_btj,
            time_vals=time_vals,
            sparse=(method == "dbn_sparse"),
            params=params,
        )
        for p in range(init_btj.shape[0]):
            for t in range(1, init_btj.shape[1]):
                dt = float(time_vals[t] - time_vals[t - 1])
                feat = np.concatenate(([dt], init_btj[p, t - 1, :]))
                pred = intercept + feat @ coef
                pred_full[p, :, t] = pred
    elif method == "glv":
        # This is a gLV-style refinement on the current log-scale tensor, not a strict ecological gLV fit on raw abundance.
        intercept, coef = _fit_glv_style(init_btj=init_btj, time_vals=time_vals, params=params)
        for p in range(init_btj.shape[0]):
            for t in range(1, init_btj.shape[1]):
                dt = float(time_vals[t] - time_vals[t - 1])
                prev = init_btj[p, t - 1, :]
                delta = intercept + prev @ coef
                pred_full[p, :, t] = prev + dt * delta
    else:
        raise ValueError(f"Unsupported dynamic method: {method}")

    out = arr.copy()
    out[mask] = pred_full[mask]
    da_imp = xr.DataArray(out, coords=da_input.coords, dims=da_input.dims, name="log_abundance_imputed")
    return da_imp


def impute_by_mask(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    model_seed: int,
    method: str,
    zero_value: float,
    method_params: Optional[Dict] = None,
    pca_params: Optional[Dict] = None,
    observed_mode: str = "positive_only_except_replace",
    verbose: bool = False,
) -> Tuple[xr.DataArray, np.ndarray]:
    method = str(method)
    method_params = {} if method_params is None else dict(method_params)
    bins_use = np.array(da_input.coords["bin"].values)
    if method == "ctf":
        observed_mask = build_observed_mask(
            da_input=da_input,
            zero_value=zero_value,
            mode=observed_mode,
            replace_mask=replace_mask,
        )
        da_imp = impute_competitor(
            method=method,
            da_input=da_input,
            replace_mask=replace_mask,
            observed_mask=observed_mask,
            zero_value=zero_value,
            seed=int(model_seed),
            params={**method_params, "eps": float(10 ** zero_value)},
            n_threads=1,
        )
        bins_trained = bins_use
    elif method == "penalized_2d2_pca":
        if pca_params is None:
            raise ValueError("pca_params is required for method='penalized_2d2_pca'")
        observed_mask = build_observed_mask(
            da_input=da_input,
            zero_value=zero_value,
            mode=observed_mode,
            replace_mask=replace_mask,
        )
        x, z, c, meta = penalized_2d2_pca(
            da=da_input,
            time=bins_use,
            d=pca_params["d"],
            q=pca_params["q"],
            lam=pca_params["lam"],
            sigma=pca_params["sigma"],
            T0=pca_params["T0"],
            T=pca_params["T"],
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
    elif method in {"mean", "nearest_positive", "temporal_knn"}:
        arr = da_input.transpose("participant", "taxon", "bin").values.astype(float)
        mask = replace_mask.transpose("participant", "taxon", "bin").values.astype(bool)
        time_vals = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
        out = arr.copy()
        taxon_fallbacks = _build_taxon_positive_fallbacks(
            values=arr,
            replace_mask=mask,
            times=time_vals,
            zero_value=zero_value,
        )
        for p in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                y = arr[p, j, :]
                mk = mask[p, j, :]
                if np.any(mk):
                    out[p, j, :] = _impute_local_trajectory(
                        y=y,
                        mask=mk,
                        times=time_vals,
                        method=method,
                        params=method_params,
                        zero_value=zero_value,
                        fallback_values=taxon_fallbacks[j, :],
                    )
        da_imp = xr.DataArray(out, coords=da_input.coords, dims=da_input.dims, name="log_abundance_imputed")
        bins_trained = bins_use
    elif method in LEGACY_LOCAL_TIME_METHODS:
        da_imp = _apply_legacy_day_vector_method(
            da_input=da_input,
            replace_mask=replace_mask,
            method=method,
            params=method_params,
            eps=float(10 ** zero_value),
        )
        bins_trained = bins_use
    elif method in {"dbn_dense", "dbn_sparse", "glv"}:
        da_imp = _impute_dynamic_method(
            da_input=da_input,
            replace_mask=replace_mask,
            method=method,
            params=method_params,
            zero_value=zero_value,
        )
        bins_trained = bins_use
    elif method in LEGACY_DYNAMIC_METHODS:
        da_imp = _apply_legacy_glv_family_method(
            da_input=da_input,
            replace_mask=replace_mask,
            method=method,
            params=method_params,
            eps=float(10 ** zero_value),
            model_seed=int(model_seed),
        )
        bins_trained = bins_use
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    da_imp = da_imp.transpose(*da_input.dims)
    da_imp.name = "log_abundance_imputed"
    off_mask = ~replace_mask.transpose(*da_input.dims).values.astype(bool)
    if not np.array_equal(
        da_imp.values[off_mask],
        da_input.transpose(*da_input.dims).values[off_mask],
        equal_nan=True,
    ):
        raise AssertionError(f"Method {method!r} modified cells outside replace_mask")
    return da_imp, bins_trained


# =====================
# Repeated simulation
# =====================

def run_one_repeat(
    da_gt,
    frac,
    repeat,
    mask_seed,
    algo_seed,
    outdir,
    d=10,
    q=10,
    lam=0.7,
    sigma=8.0,
    T0=30,
    T=30,
    save_imputed_matrix=False,
    save_format="netcdf",
    compress_level=4,
    verbose=True,
    eps: float = 1e-3,
    methods: Optional[Iterable[str]] = None,
    impute_method_params: Optional[Dict[str, Dict]] = None,
    compact_output: bool = False,
):
    bins_use = np.array(da_gt.coords["bin"].values)
    zero_value = np.log10(eps)
    methods = list(methods) if methods is not None else ["penalized_2d2_pca"]
    pca_params = {"d": d, "q": q, "lam": lam, "sigma": sigma, "T0": T0, "T": T}
    impute_method_params = {} if impute_method_params is None else impute_method_params
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frac_str = _format_frac(frac)
    repeat_str = f"{int(repeat):03d}"
    done_path = outdir / f"repeat_complete_frac_{frac_str}_repeat_{repeat_str}.json"
    # A rerun must not retain a stale success marker if this attempt fails.
    done_path.unlink(missing_ok=True)

    # User-specified logic: after truth is built, randomly zero out values in the truth tensor.
    da_rand_mask = make_random_mask(
        da_ref=da_gt,
        bins=bins_use,
        frac=float(frac),
        seed=int(mask_seed),
        zero_value=zero_value,
        eligible_mode="all_finite",
    )
    sampled_cells = int(da_rand_mask.values.sum())

    da_zi = da_gt.copy(deep=True)
    vals = da_zi.sel(bin=bins_use).values.copy()
    vals[da_rand_mask.values.astype(bool)] = zero_value
    da_zi.loc[dict(bin=bins_use)] = vals
    da_zi.name = "log_abundance_zero_inflated"

    bin_parts, repeat_parts = [], []
    imputed_outputs: Dict[str, xr.DataArray] = {}
    for method in methods:
        da_impf, bins_trained = impute_by_mask(
            da_input=da_zi,
            replace_mask=da_rand_mask,
            model_seed=int(algo_seed),
            method=method,
            method_params=impute_method_params.get(method, {}),
            pca_params=pca_params,
            zero_value=zero_value,
            observed_mode="positive_only_except_replace",
            verbose=verbose and method == "penalized_2d2_pca",
        )
        imputed_outputs[method] = da_impf
        replaced = int(da_rand_mask.sel(bin=bins_trained).values.sum())

        df_injected = evaluate_per_bin(
            da_true=da_gt, da_before=da_zi, da_after=da_impf,
            bins=bins_trained, eval_mask=da_rand_mask, bin_dim="bin"
        )
        df_injected["scope"] = "InjectedCells"

        df_all = evaluate_per_bin(
            da_true=da_gt, da_before=da_zi, da_after=da_impf,
            bins=bins_trained, eval_mask=None, bin_dim="bin"
        )
        df_all["scope"] = "AllCells"

        init_method = "mean" if method in {"dbn_dense", "dbn_sparse", "glv", *LEGACY_DYNAMIC_METHODS} else ""
        df_bin_method = pd.concat([df_injected, df_all], ignore_index=True)
        df_bin_method["frac"] = float(frac)
        df_bin_method["repeat"] = int(repeat)
        df_bin_method["method"] = method
        df_bin_method["method_group"] = _method_group(method)
        df_bin_method["init_method"] = init_method
        df_bin_method["mask_seed"] = int(mask_seed)
        df_bin_method["algo_seed"] = int(algo_seed)
        df_bin_method["sampled_cells"] = int(sampled_cells)
        df_bin_method["replaced_cells"] = int(replaced)
        df_bin_method["bin_order"] = df_bin_method["bin"].map(_bin_key)
        df_bin_method = df_bin_method.sort_values(["method", "scope", "bin_order"]).drop(columns=["bin_order"])
        bin_parts.append(df_bin_method)

        df_repeat_method = _build_weighted_summary(df_bin_method)
        df_repeat_method["frac"] = float(frac)
        df_repeat_method["repeat"] = int(repeat)
        df_repeat_method["mask_seed"] = int(mask_seed)
        df_repeat_method["algo_seed"] = int(algo_seed)
        df_repeat_method["sampled_cells"] = int(sampled_cells)
        df_repeat_method["replaced_cells"] = int(replaced)
        df_repeat_method = df_repeat_method[
            [
                "frac", "repeat", "method", "method_group", "init_method", "scope", "n_eval",
                "mse_before", "mse_after", "mse_improve",
                "pearson_before", "pearson_after", "pearson_improve",
                "wd_before", "wd_after", "wd_improve",
                "mask_seed", "algo_seed", "sampled_cells", "replaced_cells",
            ]
        ]
        repeat_parts.append(df_repeat_method)

    df_bin_one = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    df_repeat_one = pd.concat(repeat_parts, ignore_index=True) if repeat_parts else pd.DataFrame()

    if save_imputed_matrix:
        for method, da_impf in imputed_outputs.items():
            _save_dataarray(
                da_impf, "y_imp",
                outdir / f"imputed_{method}_frac_{frac_str}_repeat_{repeat_str}",
                save_format=save_format,
                compress_level=compress_level,
            )

    da_truth_abund = to_abundance_space(da_gt, eps=eps, name="truth_abundance_space")
    da_before_abund = to_abundance_space(da_zi, eps=eps, name="before_abundance_space")
    taxon_marginal_stats_parts = []
    taxon_marginal_eval_parts = []
    taxon_bin_stats_parts = []
    taxon_trajectory_eval_parts = []
    taxon_trajectory_detail_parts = []
    taxon_marginal_method_parts = []
    taxon_trajectory_method_parts = []
    bin_distribution_parts = []

    for method in methods:
        da_after = imputed_outputs.get(method)
        if da_after is None:
            continue
        da_after_abund = to_abundance_space(da_after, eps=eps, name=f"{method}_abundance_space")

        df_taxon_stats = pd.concat(
            [
                compute_taxon_marginal_stats(
                    da_abund=da_truth_abund,
                    frac=float(frac),
                    repeat=int(repeat),
                    method=method,
                    dataset_role="truth",
                    mask_seed=int(mask_seed),
                    algo_seed=int(algo_seed),
                ),
                compute_taxon_marginal_stats(
                    da_abund=da_before_abund,
                    frac=float(frac),
                    repeat=int(repeat),
                    method=method,
                    dataset_role="before",
                    mask_seed=int(mask_seed),
                    algo_seed=int(algo_seed),
                ),
                compute_taxon_marginal_stats(
                    da_abund=da_after_abund,
                    frac=float(frac),
                    repeat=int(repeat),
                    method=method,
                    dataset_role="after",
                    mask_seed=int(mask_seed),
                    algo_seed=int(algo_seed),
                ),
            ],
            ignore_index=True,
        )
        taxon_marginal_stats_parts.append(df_taxon_stats)

        df_taxon_eval, df_taxon_method = evaluate_taxon_marginal_preservation(
            df_taxon_stats=df_taxon_stats,
            frac=float(frac),
            repeat=int(repeat),
            method=method,
            mask_seed=int(mask_seed),
            algo_seed=int(algo_seed),
        )
        taxon_marginal_eval_parts.append(df_taxon_eval)
        taxon_marginal_method_parts.append(df_taxon_method)

        df_taxon_bin_stats = pd.concat(
            [
                compute_taxon_bin_stats(
                    da_abund=da_truth_abund,
                    frac=float(frac),
                    repeat=int(repeat),
                    method=method,
                    dataset_role="truth",
                    mask_seed=int(mask_seed),
                    algo_seed=int(algo_seed),
                ),
                compute_taxon_bin_stats(
                    da_abund=da_before_abund,
                    frac=float(frac),
                    repeat=int(repeat),
                    method=method,
                    dataset_role="before",
                    mask_seed=int(mask_seed),
                    algo_seed=int(algo_seed),
                ),
                compute_taxon_bin_stats(
                    da_abund=da_after_abund,
                    frac=float(frac),
                    repeat=int(repeat),
                    method=method,
                    dataset_role="after",
                    mask_seed=int(mask_seed),
                    algo_seed=int(algo_seed),
                ),
            ],
            ignore_index=True,
        )
        taxon_bin_stats_parts.append(df_taxon_bin_stats)

        df_traj_eval, df_traj_detail, df_traj_method = evaluate_taxon_trajectory_preservation(
            df_taxon_bin_stats=df_taxon_bin_stats,
            da_truth_abund=da_truth_abund,
            da_before_abund=da_before_abund,
            da_after_abund=da_after_abund,
            frac=float(frac),
            repeat=int(repeat),
            method=method,
            mask_seed=int(mask_seed),
            algo_seed=int(algo_seed),
        )
        taxon_trajectory_eval_parts.append(df_traj_eval)
        taxon_trajectory_detail_parts.append(df_traj_detail)
        taxon_trajectory_method_parts.append(df_traj_method)

        df_bin_distribution = evaluate_bin_distribution_preservation(
            da_truth_abund=da_truth_abund,
            da_before_abund=da_before_abund,
            da_after_abund=da_after_abund,
            frac=float(frac),
            repeat=int(repeat),
            method=method,
            mask_seed=int(mask_seed),
            algo_seed=int(algo_seed),
        )
        bin_distribution_parts.append(df_bin_distribution)

    df_taxon_marginal_stats = (
        pd.concat(taxon_marginal_stats_parts, ignore_index=True)
        if taxon_marginal_stats_parts
        else pd.DataFrame(columns=_taxon_marginal_stats_columns())
    )
    df_taxon_marginal_eval = (
        pd.concat(taxon_marginal_eval_parts, ignore_index=True)
        if taxon_marginal_eval_parts
        else pd.DataFrame(columns=_taxon_marginal_eval_columns())
    )
    df_taxon_bin_stats = (
        pd.concat(taxon_bin_stats_parts, ignore_index=True)
        if taxon_bin_stats_parts
        else pd.DataFrame(columns=_taxon_bin_stats_columns())
    )
    df_taxon_trajectory_eval = (
        pd.concat(taxon_trajectory_eval_parts, ignore_index=True)
        if taxon_trajectory_eval_parts
        else pd.DataFrame(columns=_taxon_trajectory_eval_columns())
    )
    df_taxon_trajectory_detail = (
        pd.concat(taxon_trajectory_detail_parts, ignore_index=True)
        if taxon_trajectory_detail_parts
        else pd.DataFrame(columns=_taxon_trajectory_detail_columns())
    )
    df_taxon_marginal_method = (
        pd.concat(taxon_marginal_method_parts, ignore_index=True)
        if taxon_marginal_method_parts
        else pd.DataFrame(columns=_taxon_marginal_method_columns())
    )
    df_taxon_trajectory_method = (
        pd.concat(taxon_trajectory_method_parts, ignore_index=True)
        if taxon_trajectory_method_parts
        else pd.DataFrame(columns=_taxon_trajectory_method_columns())
    )
    df_bin_distribution = (
        pd.concat(bin_distribution_parts, ignore_index=True)
        if bin_distribution_parts
        else pd.DataFrame(columns=_bin_distribution_eval_columns())
    )
    df_abundance_eval_summary = build_abundance_eval_repeat_summary(
        df_taxon_marginal_method=df_taxon_marginal_method,
        df_taxon_traj_method=df_taxon_trajectory_method,
        df_bin_distribution=df_bin_distribution,
    )

    output_frames = [
        (df_bin_one, f"repeat_metrics_frac_{frac_str}_repeat_{repeat_str}.csv"),
        (df_repeat_one, f"repeat_summary_frac_{frac_str}_repeat_{repeat_str}.csv"),
        (df_taxon_marginal_eval, f"taxon_marginal_eval_frac_{frac_str}_repeat_{repeat_str}.csv"),
        (df_taxon_trajectory_eval, f"taxon_trajectory_eval_frac_{frac_str}_repeat_{repeat_str}.csv"),
        (df_bin_distribution, f"bin_distribution_eval_frac_{frac_str}_repeat_{repeat_str}.csv"),
        (df_abundance_eval_summary, f"abundance_eval_summary_frac_{frac_str}_repeat_{repeat_str}.csv"),
    ]
    if not compact_output:
        output_frames.extend(
            [
                (df_taxon_marginal_stats, f"taxon_marginal_stats_frac_{frac_str}_repeat_{repeat_str}.csv"),
                (df_taxon_bin_stats, f"taxon_bin_stats_frac_{frac_str}_repeat_{repeat_str}.csv"),
                (df_taxon_trajectory_detail, f"taxon_trajectory_detail_frac_{frac_str}_repeat_{repeat_str}.csv"),
            ]
        )
    for frame, filename in output_frames:
        _atomic_to_csv(frame, outdir / filename, index=False)

    completion = {
        "frac": float(frac),
        "repeat": int(repeat),
        "mask_seed": int(mask_seed),
        "algo_seed": int(algo_seed),
        "methods": methods,
        "outputs": [filename for _, filename in output_frames],
    }
    _atomic_write_text(
        json.dumps(completion, ensure_ascii=False, indent=2),
        done_path,
    )

    injected_eval = int(df_repeat_one.loc[df_repeat_one["scope"].eq("InjectedCells"), "n_eval"].sum())
    print(
        f"[repeat done] frac={frac:.4f} repeat={repeat} mask_seed={mask_seed} "
        f"algo_seed={algo_seed} sampled={sampled_cells} eval_cells={injected_eval} methods={','.join(methods)}"
    )
    return df_bin_one, df_repeat_one


_REPEAT_WORKER_TRUTH: Optional[xr.DataArray] = None


def _init_repeat_worker(da_gt: xr.DataArray) -> None:
    """Load the immutable truth tensor once in each worker process."""
    global _REPEAT_WORKER_TRUTH
    _REPEAT_WORKER_TRUTH = da_gt


def _run_one_repeat_worker(task_kwargs):
    if _REPEAT_WORKER_TRUTH is None:
        raise RuntimeError("repeat worker was not initialized with the truth tensor")
    return run_one_repeat(da_gt=_REPEAT_WORKER_TRUTH, **task_kwargs)


def run_repeats_for_frac(
    da_gt,
    frac,
    n_repeats=100,
    repeat_start=0,
    mask_seed_base=123,
    algo_seed=0,
    outdir="artifacts/simulation3",
    d=10,
    q=10,
    lam=0.7,
    sigma=8.0,
    T0=30,
    T=30,
    save_imputed_matrix=False,
    save_format="netcdf",
    compress_level=4,
    verbose=True,
    eps: float = 1e-3,
    methods: Optional[Iterable[str]] = None,
    impute_method_params: Optional[Dict[str, Dict]] = None,
    n_workers: int = 1,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if int(n_workers) < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    bin_parts, repeat_parts, fail_rows = [], [], []
    repeat_values = list(range(repeat_start, repeat_start + n_repeats))
    for r in repeat_values:
        marker = outdir / f"repeat_complete_frac_{_format_frac(frac)}_repeat_{int(r):03d}.json"
        marker.unlink(missing_ok=True)
    methods = list(methods) if methods is not None else ["penalized_2d2_pca"]
    impute_method_params = {} if impute_method_params is None else impute_method_params

    task_kwargs = [
        {
            "frac": frac,
            "repeat": r,
            "mask_seed": mask_seed_base + r,
            "algo_seed": algo_seed,
            "outdir": outdir,
            "d": d,
            "q": q,
            "lam": lam,
            "sigma": sigma,
            "T0": T0,
            "T": T,
            "save_imputed_matrix": save_imputed_matrix,
            "save_format": save_format,
            "compress_level": compress_level,
            "verbose": verbose,
            "eps": eps,
            "methods": methods,
            "impute_method_params": impute_method_params,
        }
        for r in repeat_values
    ]

    effective_workers = min(int(n_workers), max(1, len(task_kwargs)))
    if effective_workers == 1:
        repeat_results = []
        for kwargs in task_kwargs:
            try:
                repeat_results.append((kwargs["repeat"], run_one_repeat(da_gt=da_gt, **kwargs), None))
            except Exception as exc:
                repeat_results.append((kwargs["repeat"], None, (exc, traceback.format_exc())))
    else:
        print(
            f"[parallel] frac={float(frac):.4f} repeats={len(task_kwargs)} "
            f"workers={effective_workers}"
        )
        repeat_results = []
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_repeat_worker,
            initargs=(da_gt,),
        ) as executor:
            futures = [executor.submit(_run_one_repeat_worker, kwargs) for kwargs in task_kwargs]
            # Consume in repeat order so returned aggregate row ordering stays deterministic.
            for kwargs, future in zip(task_kwargs, futures):
                try:
                    repeat_results.append((kwargs["repeat"], future.result(), None))
                except Exception as exc:
                    repeat_results.append((kwargs["repeat"], None, (exc, traceback.format_exc())))

    for r, result, error_info in repeat_results:
        frac_str = _format_frac(frac)
        repeat_str = f"{int(r):03d}"
        fail_path = outdir / f"repeat_failed_frac_{frac_str}_repeat_{repeat_str}.log"
        if error_info is None:
            df_bin_one, df_repeat_one = result
            bin_parts.append(df_bin_one)
            repeat_parts.append(df_repeat_one)
            fail_path.unlink(missing_ok=True)
            continue

        exc, err_text = error_info
        _atomic_write_text(err_text, fail_path)
        fail_rows.append(
            {
                "frac": float(frac),
                "repeat": int(r),
                "mask_seed": int(mask_seed_base + r),
                "algo_seed": int(algo_seed),
                "error": str(exc),
                "log_path": str(fail_path),
            }
        )
        print(f"[repeat failed] frac={frac:.4f} repeat={r}, log={fail_path}")

    df_bin_all = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    df_repeat_all = pd.concat(repeat_parts, ignore_index=True) if repeat_parts else pd.DataFrame()
    df_fail = pd.DataFrame(fail_rows)

    frac_str = _format_frac(frac)
    frac_bin_path = outdir / f"frac_{frac_str}_bin_metrics_all_repeats.csv"
    frac_repeat_path = outdir / f"frac_{frac_str}_repeat_metrics_all_repeats.csv"
    if not df_bin_all.empty:
        _atomic_to_csv(df_bin_all, frac_bin_path, index=False)
    else:
        frac_bin_path.unlink(missing_ok=True)
    if not df_repeat_all.empty:
        _atomic_to_csv(df_repeat_all, frac_repeat_path, index=False)
    else:
        frac_repeat_path.unlink(missing_ok=True)
    frac_fail_path = outdir / f"frac_{frac_str}_repeat_failures.csv"
    if not df_fail.empty:
        _atomic_to_csv(df_fail, frac_fail_path, index=False)
    else:
        frac_fail_path.unlink(missing_ok=True)

    return df_bin_all, df_repeat_all, df_fail


def merge_repeat_outputs(outdir):
    outdir = Path(outdir)
    completed_keys = _completed_repeat_keys(outdir)
    repeat_files = _collect_repeat_files(
        outdir, "repeat_summary_frac_*_repeat_*.csv", completed_keys
    )
    bin_files = _collect_repeat_files(
        outdir, "repeat_metrics_frac_*_repeat_*.csv", completed_keys
    )

    df_repeat = pd.concat([pd.read_csv(p) for p in repeat_files], ignore_index=True) if repeat_files else pd.DataFrame()
    df_bin = pd.concat([pd.read_csv(p) for p in bin_files], ignore_index=True) if bin_files else pd.DataFrame()
    if not df_repeat.empty and "method" not in df_repeat.columns:
        df_repeat["method"] = "penalized_2d2_pca"
    if not df_bin.empty and "method" not in df_bin.columns:
        df_bin["method"] = "penalized_2d2_pca"
    if not df_repeat.empty:
        df_repeat = df_repeat.drop_duplicates(subset=["frac", "repeat", "method", "scope"], keep="last")
        df_repeat = df_repeat.sort_values(["frac", "repeat", "method", "scope"]).reset_index(drop=True)
    if not df_bin.empty:
        df_bin = df_bin.drop_duplicates(subset=["frac", "repeat", "method", "scope", "bin"], keep="last")
        df_bin["_bin_order"] = df_bin["bin"].map(_bin_key)
        df_bin = (
            df_bin.sort_values(["frac", "repeat", "method", "scope", "_bin_order"])
            .drop(columns=["_bin_order"])
            .reset_index(drop=True)
        )

    df_errorbar = summarize_across_repeats(df_repeat)

    repeat_path = outdir / "repeat_level_metrics.csv"
    errorbar_path = outdir / "errorbar_summary.csv"
    bin_path = outdir / "repeat_bin_metrics_all.csv"
    if not df_repeat.empty:
        _atomic_to_csv(df_repeat, repeat_path, index=False)
    else:
        repeat_path.unlink(missing_ok=True)
    if not df_errorbar.empty:
        _atomic_to_csv(df_errorbar, errorbar_path, index=False)
    else:
        errorbar_path.unlink(missing_ok=True)
    if not df_bin.empty:
        _atomic_to_csv(df_bin, bin_path, index=False)
    else:
        bin_path.unlink(missing_ok=True)

    completed_label = len(completed_keys) if completed_keys is not None else "legacy/unmarked"
    print(f"[merge] completed repeats: {completed_label}")
    print(f"[merge] repeat-level: {repeat_path}")
    print(f"[merge] errorbar: {errorbar_path}")
    return df_bin, df_repeat, df_errorbar


def _run_simulation_over_fracs(
    da_truth,
    fracs,
    n_repeats=100,
    repeat_start=0,
    mask_seed_base=123,
    algo_seed=0,
    save_dir="artifacts/simulation3",
    save_format="netcdf",
    compress_level=4,
    save_imputed_matrix=False,
    d=10,
    q=10,
    lam=0.7,
    sigma=8.0,
    T0=30,
    T=30,
    verbose=True,
    eps: float = 1e-3,
    methods: Optional[Iterable[str]] = None,
    impute_method_params: Optional[Dict[str, Dict]] = None,
    n_workers: int = 1,
):
    fracs = [float(frac) for frac in fracs]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for frac in fracs:
        for r in range(repeat_start, repeat_start + n_repeats):
            marker = save_dir / f"repeat_complete_frac_{_format_frac(frac)}_repeat_{int(r):03d}.json"
            marker.unlink(missing_ok=True)

    all_bin, all_repeat, all_fail = [], [], []
    for frac in fracs:
        df_bin_one, df_repeat_one, df_fail_one = run_repeats_for_frac(
            da_gt=da_truth,
            frac=float(frac),
            n_repeats=n_repeats,
            repeat_start=repeat_start,
            mask_seed_base=mask_seed_base,
            algo_seed=algo_seed,
            outdir=save_dir,
            d=d, q=q, lam=lam, sigma=sigma, T0=T0, T=T,
            save_imputed_matrix=save_imputed_matrix,
            save_format=save_format,
            compress_level=compress_level,
            verbose=verbose,
            eps=eps,
            methods=methods,
            impute_method_params=impute_method_params,
            n_workers=n_workers,
        )
        if not df_bin_one.empty:
            all_bin.append(df_bin_one)
        if not df_repeat_one.empty:
            all_repeat.append(df_repeat_one)
        if not df_fail_one.empty:
            all_fail.append(df_fail_one)

    df_bin_all = pd.concat(all_bin, ignore_index=True) if all_bin else pd.DataFrame()
    df_repeat_all = pd.concat(all_repeat, ignore_index=True) if all_repeat else pd.DataFrame()
    df_fail_all = pd.concat(all_fail, ignore_index=True) if all_fail else pd.DataFrame()
    save_dir = Path(save_dir)
    failure_path = save_dir / "repeat_failures_all.csv"
    if not df_fail_all.empty:
        _atomic_to_csv(df_fail_all, failure_path, index=False)
    else:
        failure_path.unlink(missing_ok=True)

    # Only the parent reaches these global merge calls, after every worker finished.
    df_bin_all, df_repeat_all, df_errorbar = merge_repeat_outputs(save_dir)
    merge_abundance_eval_outputs(save_dir)

    if not df_fail_all.empty:
        raise RuntimeError(
            f"{len(df_fail_all)} repeat task(s) failed; see {failure_path}. "
            "Only repeats with completion markers were merged."
        )

    return df_bin_all, df_repeat_all, df_errorbar, df_fail_all


# =====================
# CLI
# =====================

def _parse_fracs(frac_arg, fracs_arg):
    if frac_arg is not None:
        return [float(frac_arg)]
    if fracs_arg is None or str(fracs_arg).strip() == "":
        return [0.1, 0.3, 0.5, 0.7, 0.9]
    vals = [float(x.strip()) for x in str(fracs_arg).split(",") if x.strip() != ""]
    return vals if vals else [0.1, 0.3, 0.5, 0.7, 0.9]


def _parse_methods(methods_arg) -> List[str]:
    allowed = set(SIM3_SUPPORTED_METHODS)
    if methods_arg is None or methods_arg == [] or str(methods_arg).strip() == "":
        return list(DEFAULT_METHODS)
    methods = []
    raw_items = methods_arg if isinstance(methods_arg, (list, tuple)) else [methods_arg]
    parsed_items = []
    for item in raw_items:
        parsed_items.extend([x.strip() for x in str(item).split(",") if x.strip()])
    for method in parsed_items:
        if method not in allowed:
            raise ValueError(f"Unsupported method in --methods: {method}")
        if method not in methods:
            methods.append(method)
    return methods if methods else list(DEFAULT_METHODS)


def _build_impute_method_params(args, zero_value: float) -> Dict[str, Dict]:
    return {
        "mean": {},
        "nearest_positive": {},
        "median": {},
        "last": {},
        "weighted_time_average": {},
        "temporal_knn": {"k": int(args.temporal_knn_k), "kernel_sigma": float(args.temporal_knn_sigma)},
        "spline": {"min_points": int(args.spline_min_points)},
        "equal": {"fill_value": float(args.equal_fill_value)},
        "weighted_avg_legacy": {},
        "spline_legacy": {"min_points": int(args.spline_min_points)},
        "knn_epanechnikov_legacy": {"k": int(args.legacy_knn_k)},
        "dbn_dense": {"alpha": float(args.dbn_dense_alpha)},
        "dbn_sparse": {"alpha": float(args.dbn_sparse_alpha), "n_iter": int(args.dbn_sparse_n_iter)},
        "glv": {"alpha": float(args.glv_alpha), "init_method": "mean"},
        "glv_mse_legacy": {"init_method": "mean"},
        "glv_ridge_legacy": {
            "init_method": "mean",
            "lambda_M": float(args.glv_ridge_lambda_M),
            "lambda_mue": float(args.glv_ridge_lambda_mue),
            "lambda_E": float(args.glv_ridge_lambda_E),
        },
        "limits_legacy": {
            "init_method": "mean",
            "error_threshold": float(args.limits_error_threshold),
            "repeats": int(args.limits_repeats),
            "med_or_avg": int(args.limits_med_or_avg),
        },
        "ctf": {
            "rank": int(args.competitor_ctf_rank),
            "n_initializations": int(args.competitor_ctf_initializations),
            "max_iterations_als": int(args.competitor_ctf_max_iterations),
            "max_iterations_rptm": int(args.competitor_ctf_max_iterations),
        },
        "penalized_2d2_pca": {},
    }


def main():
    parser = argparse.ArgumentParser(
        description="CTF truth building + mask-aware penalized 2D2-PCA imputation simulation."
    )
    parser.add_argument("--tax_path", "--tax-path", dest="tax_path", type=str, default="hmp2_mgx_taxonomy.tsv")
    parser.add_argument("--meta_path", "--meta-path", dest="meta_path", type=str, default="metagenomics.xlsx")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--min_pos_cells", type=int, default=10)
    parser.add_argument("--min_participants", type=int, default=5)

    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--n_repeats", "--n-repeats", dest="n_repeats", type=int, default=100)
    parser.add_argument(
        "--n_workers", "--n-workers", dest="n_workers", type=int, default=1,
        help="Worker processes used to parallelize repeats; methods within a repeat remain serial.",
    )
    parser.add_argument("--fracs", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--truth_path", "--truth-path", dest="truth_path", type=str, default=None)
    parser.add_argument(
        "--base_truth_path", "--base-truth-path", dest="base_truth_path", type=str, default=None,
        help="Optional frozen Sim2 CP-ALS truth used as the pre-effect Sim3 base truth.",
    )
    parser.add_argument("--outdir", type=str, default="artifacts/simulation3")
    parser.add_argument("--mask_seed", "--mask-seed", dest="mask_seed", type=int, default=123)
    parser.add_argument("--algo_seed", "--algo-seed", dest="algo_seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--competitor-ctf-rank", type=int, default=3)
    parser.add_argument("--competitor-ctf-initializations", type=int, default=10)
    parser.add_argument("--competitor-ctf-max-iterations", type=int, default=50)
    parser.add_argument("--save_imputed_matrix", action="store_true")
    parser.add_argument("--save_format", type=str, default="netcdf", choices=["netcdf", "zarr"])
    parser.add_argument("--compress_level", type=int, default=4)

    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--q", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--T0", type=int, default=30)
    parser.add_argument("--T", type=int, default=30)

    parser.add_argument("--ctf_rank", type=int, default=3)
    parser.add_argument("--ctf_method", type=str, default="fallback", choices=["auto", "gemelli", "fallback"])
    parser.add_argument("--ctf_seed", type=int, default=0)
    parser.add_argument("--enable_effect", "--enable-effect", dest="enable_effect", action="store_true")
    parser.add_argument("--effect_seed", type=int, default=2026)
    parser.add_argument("--n_cd", type=int, default=65)
    parser.add_argument("--n_noncd", type=int, default=65)
    parser.add_argument("--diff_frac", type=float, default=0.10)
    parser.add_argument("--group_effect_mean", type=float, default=0.8)
    parser.add_argument("--group_effect_sd", type=float, default=0.2)
    parser.add_argument("--interaction_mean", type=float, default=0.15)
    parser.add_argument("--interaction_sd", type=float, default=0.05)
    parser.add_argument("--random_slope_sd", type=float, default=0.05)
    parser.add_argument("--noise_sd", type=float, default=0.10)
    parser.add_argument("--force_rebuild_truth", action="store_true")
    parser.add_argument("--build_truth_only", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument(
        "--require_existing_truth", "--require-existing-truth",
        dest="require_existing_truth", action="store_true",
        help="Fail if --truth-path is absent; array shards never rebuild frozen truth.",
    )
    parser.add_argument(
        "--defer_merge", "--defer-merge", dest="defer_merge", action="store_true",
        help="Do not write shared merged outputs from a single array shard.",
    )
    parser.add_argument(
        "--compact_output", "--compact-output", dest="compact_output", action="store_true",
        help="Omit large intermediate taxon stats/detail CSVs while retaining all summary metrics.",
    )
    parser.add_argument("--verbose", action="store_true", help="Compatibility flag; detailed output is on unless --quiet is set.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--temporal_knn_k", "--temporal-knn-k", dest="temporal_knn_k", type=int, default=3)
    parser.add_argument("--temporal_knn_sigma", "--temporal-knn-bandwidth", "--temporal-knn-sigma", dest="temporal_knn_sigma", type=float, default=4.0)
    parser.add_argument("--spline_min_points", "--spline-min-points", dest="spline_min_points", type=int, default=4)
    parser.add_argument("--equal_fill_value", "--equal-value", dest="equal_fill_value", type=float, default=0.0)
    parser.add_argument("--dbn_dense_alpha", "--dbn-dense-alpha", dest="dbn_dense_alpha", type=float, default=1.0)
    parser.add_argument("--dbn_sparse_alpha", "--dbn-sparse-alpha", dest="dbn_sparse_alpha", type=float, default=0.05)
    parser.add_argument("--dbn_sparse_n_iter", "--dbn-sparse-n-iter", dest="dbn_sparse_n_iter", type=int, default=150)
    parser.add_argument("--glv_alpha", "--glv-alpha", dest="glv_alpha", type=float, default=0.5)
    parser.add_argument(
        "--legacy_knn_k",
        "--legacy-knn-k",
        dest="legacy_knn_k",
        type=int,
        default=3,
        help="Neighbor count for knn_epanechnikov_legacy.",
    )
    parser.add_argument(
        "--glv_ridge_lambda_M",
        "--glv-ridge-lambda-M",
        dest="glv_ridge_lambda_M",
        type=float,
        default=1e-3,
        help="Ridge penalty on the interaction matrix for glv_ridge_legacy.",
    )
    parser.add_argument(
        "--glv_ridge_lambda_mue",
        "--glv-ridge-lambda-mue",
        dest="glv_ridge_lambda_mue",
        type=float,
        default=1e-3,
        help="Ridge penalty on the self-growth term for glv_ridge_legacy.",
    )
    parser.add_argument(
        "--glv_ridge_lambda_E",
        "--glv-ridge-lambda-E",
        dest="glv_ridge_lambda_E",
        type=float,
        default=1e-3,
        help="Ridge penalty on perturbation terms for glv_ridge_legacy; perturbations are currently disabled.",
    )
    parser.add_argument(
        "--limits_error_threshold",
        "--limits-error-threshold",
        dest="limits_error_threshold",
        type=float,
        default=0.05,
        help="Forward-selection improvement threshold for limits_legacy.",
    )
    parser.add_argument(
        "--limits_repeats",
        "--limits-repeats",
        dest="limits_repeats",
        type=int,
        default=20,
        help="Number of repeated train/test splits for limits_legacy.",
    )
    parser.add_argument(
        "--limits_med_or_avg",
        "--limits-med-or-avg",
        dest="limits_med_or_avg",
        type=int,
        default=0,
        help="Aggregation mode for limits_legacy: 0=median, 1=mean.",
    )

    args = parser.parse_args()
    verbose = not args.quiet
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.methods)
    if args.eps <= 0:
        raise ValueError(f"--eps must be > 0, got {args.eps}")
    if args.n_workers < 1:
        raise ValueError(f"--n-workers must be >= 1, got {args.n_workers}")
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    if allocated_cpus and args.n_workers > allocated_cpus:
        raise ValueError(
            f"--n-workers={args.n_workers} exceeds SLURM_CPUS_PER_TASK={allocated_cpus}"
        )
    if args.n_workers > 1:
        blas_thread_vars = [
            name
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            if int(os.environ.get(name, "1") or 1) != 1
        ]
        if blas_thread_vars:
            warnings.warn(
                "Parallel repeat workers should use one BLAS thread each; non-unit values found for "
                + ", ".join(blas_thread_vars),
                RuntimeWarning,
            )
    if not (0.0 <= float(args.diff_frac) <= 1.0):
        raise ValueError(f"--diff_frac must be in [0, 1], got {args.diff_frac}")
    zero_value = np.log10(args.eps)
    impute_method_params = _build_impute_method_params(args, zero_value=zero_value)
    methods, skipped_competitors = available_requested_methods(methods, impute_method_params)
    if skipped_competitors:
        details = "; ".join(
            f"{method}: {reason}" for method, reason in skipped_competitors.items()
        )
        raise RuntimeError(f"Requested retained competitor is unavailable: {details}")
    if not methods:
        raise RuntimeError("No requested imputation methods are available")

    if args.merge_only:
        merge_repeat_outputs(outdir=outdir)
        merge_abundance_eval_outputs(outdir=outdir)
        return

    base_da, base_observed_mask, _ = build_base_tensor(
        tax_path=args.tax_path,
        meta_path=args.meta_path,
        eps=args.eps,
        min_pos_cells=args.min_pos_cells,
        min_participants=args.min_participants,
        verbose=verbose,
    )

    if args.truth_path is None:
        truth_path = _default_truth_path(
            outdir=outdir,
            save_format=args.save_format,
            enable_effect=bool(args.enable_effect),
            effect_seed=int(args.effect_seed),
            n_cd=int(args.n_cd),
            n_noncd=int(args.n_noncd),
            diff_frac=float(args.diff_frac),
            group_effect_mean=float(args.group_effect_mean),
            group_effect_sd=float(args.group_effect_sd),
            interaction_mean=float(args.interaction_mean),
            interaction_sd=float(args.interaction_sd),
            random_slope_sd=float(args.random_slope_sd),
            noise_sd=float(args.noise_sd),
        )
    else:
        truth_path = Path(args.truth_path)

    run_config = {
        "enable_effect": bool(args.enable_effect),
        "truth_path": str(truth_path),
        "n_workers": int(args.n_workers),
        "n_repeats": int(args.n_repeats),
        "fracs": _parse_fracs(args.frac, args.fracs),
        "methods": methods,
        "mask_seed": int(args.mask_seed),
        "algo_seed": int(args.algo_seed),
        "effect": {
            "seed": int(args.effect_seed),
            "n_cd": int(args.n_cd),
            "n_noncd": int(args.n_noncd),
            "diff_frac": float(args.diff_frac),
            "group_effect_mean": float(args.group_effect_mean),
            "group_effect_sd": float(args.group_effect_sd),
            "interaction_mean": float(args.interaction_mean),
            "interaction_sd": float(args.interaction_sd),
            "random_slope_sd": float(args.random_slope_sd),
            "noise_sd": float(args.noise_sd),
        },
    }
    if args.defer_merge and args.frac is not None and args.repeat is not None:
        config_path = (
            outdir
            / "run_config_shards"
            / f"run_config_frac_{_format_frac(args.frac)}_repeat_{int(args.repeat):03d}.json"
        )
    else:
        config_path = outdir / "run_config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if bool(existing_config.get("enable_effect", False)) != bool(args.enable_effect):
            raise RuntimeError(
                "Refusing to mix effect-enabled and effect-disabled results in the same outdir: "
                f"{outdir}"
            )
    _atomic_write_text(json.dumps(run_config, ensure_ascii=False, indent=2), config_path)
    print(
        f"[config] enable_effect={bool(args.enable_effect)} n_workers={args.n_workers} "
        f"truth_path={truth_path} outdir={outdir}"
    )

    da_gt = build_truth_once(
        base_da=base_da,
        base_observed_mask=base_observed_mask,
        truth_path=truth_path,
        force_rebuild=args.force_rebuild_truth,
        save_format=args.save_format,
        compress_level=args.compress_level,
        rank=args.ctf_rank,
        method=args.ctf_method,
        ctf_seed=args.ctf_seed,
        ctf_verbose=verbose,
        eps=args.eps,
        enable_effect=bool(args.enable_effect),
        effect_seed=int(args.effect_seed),
        n_cd=int(args.n_cd),
        n_noncd=int(args.n_noncd),
        diff_frac=float(args.diff_frac),
        group_effect_mean=float(args.group_effect_mean),
        group_effect_sd=float(args.group_effect_sd),
        interaction_mean=float(args.interaction_mean),
        interaction_sd=float(args.interaction_sd),
        random_slope_sd=float(args.random_slope_sd),
        noise_sd=float(args.noise_sd),
        effect_artifact_dir=truth_path.parent,
        base_truth_path=args.base_truth_path,
        require_existing=bool(args.require_existing_truth),
    )

    if args.build_truth_only:
        return

    if args.frac is not None and args.repeat is not None:
        run_one_repeat(
            da_gt=da_gt,
            frac=float(args.frac),
            repeat=int(args.repeat),
            mask_seed=int(args.mask_seed),
            algo_seed=int(args.algo_seed),
            outdir=outdir,
            d=args.d, q=args.q, lam=args.lam, sigma=args.sigma, T0=args.T0, T=args.T,
            save_imputed_matrix=args.save_imputed_matrix,
            save_format=args.save_format,
            compress_level=args.compress_level,
            verbose=verbose,
            eps=args.eps,
            methods=methods,
            impute_method_params=impute_method_params,
            compact_output=bool(args.compact_output),
        )
        if not args.defer_merge:
            merge_repeat_outputs(outdir=outdir)
            merge_abundance_eval_outputs(outdir=outdir)
        return

    if args.frac is not None:
        _, _, df_fail = run_repeats_for_frac(
            da_gt=da_gt,
            frac=float(args.frac),
            n_repeats=int(args.n_repeats),
            repeat_start=0,
            mask_seed_base=int(args.mask_seed),
            algo_seed=int(args.algo_seed),
            outdir=outdir,
            d=args.d, q=args.q, lam=args.lam, sigma=args.sigma, T0=args.T0, T=args.T,
            save_imputed_matrix=args.save_imputed_matrix,
            save_format=args.save_format,
            compress_level=args.compress_level,
            verbose=verbose,
            eps=args.eps,
            methods=methods,
            impute_method_params=impute_method_params,
            n_workers=int(args.n_workers),
        )
        merge_repeat_outputs(outdir=outdir)
        merge_abundance_eval_outputs(outdir=outdir)
        if not df_fail.empty:
            raise RuntimeError(
                f"{len(df_fail)} repeat task(s) failed; see frac-level failure CSV in {outdir}."
            )
        return

    fracs = _parse_fracs(args.frac, args.fracs)
    _run_simulation_over_fracs(
        da_truth=da_gt,
        fracs=fracs,
        n_repeats=int(args.n_repeats),
        repeat_start=0,
        mask_seed_base=int(args.mask_seed),
        algo_seed=int(args.algo_seed),
        save_dir=outdir,
        save_format=args.save_format,
        compress_level=args.compress_level,
        save_imputed_matrix=args.save_imputed_matrix,
        d=args.d, q=args.q, lam=args.lam, sigma=args.sigma, T0=args.T0, T=args.T,
        verbose=verbose,
        eps=args.eps,
        methods=methods,
        impute_method_params=impute_method_params,
        n_workers=int(args.n_workers),
    )


# =====================
# Abundance-space distribution / trajectory evaluation
# =====================

def _taxon_marginal_stats_columns() -> List[str]:
    """Return column layout for taxon marginal stats."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
        "dataset_role", "taxon", "n_obs", "abund_mean", "abund_sd", "abund_mean_over_sd",
    ]


def _taxon_marginal_eval_columns() -> List[str]:
    """Return column layout for taxon marginal evaluation detail."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "taxon",
        "delta_mean_before", "delta_mean_after",
        "delta_sd_before", "delta_sd_after",
        "delta_mean_over_sd_before", "delta_mean_over_sd_after",
    ]


def _taxon_marginal_method_columns() -> List[str]:
    """Return column layout for taxon marginal method summary."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "scope",
        "n_taxa_used",
        "marginal_ratio_wd_before", "marginal_ratio_wd_after", "marginal_ratio_wd_improve",
        "marginal_avg_euclid_before", "marginal_avg_euclid_after", "marginal_avg_euclid_improve",
    ]


def _taxon_bin_stats_columns() -> List[str]:
    """Return column layout for taxon-bin stats."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
        "dataset_role", "taxon", "bin", "n_obs",
        "abund_mean_bin", "abund_sd_bin", "abund_mean_over_sd_bin",
    ]


def _taxon_trajectory_eval_columns() -> List[str]:
    """Return column layout for taxon trajectory evaluation."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "taxon",
        "traj_mean_l2_before", "traj_mean_l2_after",
        "traj_sd_l2_before", "traj_sd_l2_after",
        "traj_mean_sd_joint_euclid_before", "traj_mean_sd_joint_euclid_after",
        "taxon_avg_bin_wd_before", "taxon_avg_bin_wd_after",
        "traj_mean_l2_improve", "traj_sd_l2_improve",
        "traj_mean_sd_joint_euclid_improve", "taxon_avg_bin_wd_improve",
    ]


def _taxon_trajectory_detail_columns() -> List[str]:
    """Return column layout for taxon-bin trajectory detail."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
        "taxon", "bin",
        "wd_before_bin", "wd_after_bin",
        "euclid_mean_sd_before_bin", "euclid_mean_sd_after_bin",
    ]


def _taxon_trajectory_method_columns() -> List[str]:
    """Return column layout for taxon trajectory method summary."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "scope", "n_taxa_used",
        "traj_mean_l2_before", "traj_mean_l2_after", "traj_mean_l2_improve",
        "traj_sd_l2_before", "traj_sd_l2_after", "traj_sd_l2_improve",
        "traj_mean_sd_joint_euclid_before", "traj_mean_sd_joint_euclid_after", "traj_mean_sd_joint_euclid_improve",
        "taxon_avg_bin_wd_before", "taxon_avg_bin_wd_after", "taxon_avg_bin_wd_improve",
    ]


def _bin_distribution_eval_columns() -> List[str]:
    """Return column layout for bin distribution evaluation."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "bin",
        "n_taxa_used",
        "bin_ratio_wd_before", "bin_ratio_wd_after", "bin_ratio_wd_improve",
        "bin_avg_euclid_before", "bin_avg_euclid_after", "bin_avg_euclid_improve",
    ]


def _abundance_eval_summary_columns() -> List[str]:
    """Return column layout for repeat-level abundance summary."""
    return [
        "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
        "scope", "metric", "value",
    ]


def to_abundance_space(
    da_log: xr.DataArray,
    eps: float = 1e-3,
    name: str = "abundance_space",
) -> xr.DataArray:
    """Invert log10(percent + eps) back to abundance / percent space."""
    _validate_da(da_log)
    vals = np.power(10.0, da_log.values.astype(float)) - float(eps)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    vals = np.maximum(vals, 0.0)
    da_out = xr.DataArray(vals, coords=da_log.coords, dims=da_log.dims, name=name)
    da_out.attrs.update(getattr(da_log, "attrs", {}))
    da_out.attrs["eps"] = float(eps)
    da_out.attrs["source_scale"] = "log10(percent+eps)"
    return da_out


def _safe_ratio(mean_value: float, sd_value: float) -> float:
    """Compute a guarded mean-over-sd ratio."""
    if not np.isfinite(mean_value) or not np.isfinite(sd_value) or sd_value <= 0:
        return np.nan
    return float(mean_value / sd_value)


def _mean_euclid_2d(
    x_true_mean: np.ndarray,
    x_true_sd: np.ndarray,
    x_pred_mean: np.ndarray,
    x_pred_sd: np.ndarray,
) -> float:
    """Average Euclidean distance between paired 2D mean-sd points."""
    mask = (
        np.isfinite(x_true_mean) & np.isfinite(x_true_sd)
        & np.isfinite(x_pred_mean) & np.isfinite(x_pred_sd)
    )
    if not np.any(mask):
        return np.nan
    dist = np.sqrt((x_pred_mean[mask] - x_true_mean[mask]) ** 2 + (x_pred_sd[mask] - x_true_sd[mask]) ** 2)
    return float(np.mean(dist))


def _l2_distance(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Compute L2 distance on paired finite entries only."""
    mask = np.isfinite(x_true) & np.isfinite(x_pred)
    if not np.any(mask):
        return np.nan
    return float(np.linalg.norm(x_pred[mask] - x_true[mask]))


def _safe_int(value) -> Optional[int]:
    """Convert optional numeric metadata to int when available."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return int(value)


def compute_taxon_marginal_stats(
    da_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    dataset_role: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute taxon-level abundance marginal summaries across participant-bin cells."""
    _validate_da(da_abund)
    rows = []
    for taxon in da_abund.coords["taxon"].values:
        values = da_abund.sel(taxon=taxon).values.astype(float).ravel()
        values = values[np.isfinite(values)]
        n_obs = int(values.size)
        abund_mean = float(np.mean(values)) if n_obs > 0 else np.nan
        abund_sd = float(np.std(values, ddof=1)) if n_obs >= 2 else np.nan
        rows.append(
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": _safe_int(mask_seed),
                "algo_seed": _safe_int(algo_seed),
                "dataset_role": dataset_role,
                "taxon": taxon,
                "n_obs": n_obs,
                "abund_mean": abund_mean,
                "abund_sd": abund_sd,
                "abund_mean_over_sd": _safe_ratio(abund_mean, abund_sd),
            }
        )
    return pd.DataFrame(rows, columns=_taxon_marginal_stats_columns())


def evaluate_taxon_marginal_preservation(
    df_taxon_stats: pd.DataFrame,
    frac: float,
    repeat: int,
    method: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate taxon-level marginal preservation for before/after against truth."""
    detail_cols = _taxon_marginal_eval_columns()
    method_cols = _taxon_marginal_method_columns()
    if df_taxon_stats.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=method_cols)

    required_roles = {"truth", "before", "after"}
    roles_present = set(df_taxon_stats.get("dataset_role", pd.Series(dtype=object)).dropna().astype(str))
    if not required_roles.issubset(roles_present):
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=method_cols)

    pivot = (
        df_taxon_stats.pivot_table(
            index="taxon",
            columns="dataset_role",
            values=["abund_mean", "abund_sd", "abund_mean_over_sd"],
            aggfunc="first",
        )
        .sort_index()
    )
    if pivot.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=method_cols)

    def _col(metric: str, role: str) -> np.ndarray:
        if (metric, role) not in pivot.columns:
            return np.full(pivot.shape[0], np.nan, dtype=float)
        return pivot[(metric, role)].to_numpy(dtype=float)

    truth_mean = _col("abund_mean", "truth")
    before_mean = _col("abund_mean", "before")
    after_mean = _col("abund_mean", "after")
    truth_sd = _col("abund_sd", "truth")
    before_sd = _col("abund_sd", "before")
    after_sd = _col("abund_sd", "after")
    truth_ratio = _col("abund_mean_over_sd", "truth")
    before_ratio = _col("abund_mean_over_sd", "before")
    after_ratio = _col("abund_mean_over_sd", "after")

    df_detail = pd.DataFrame(
        {
            "frac": float(frac),
            "repeat": int(repeat),
            "method": method,
            "method_group": _method_group(method),
            "mask_seed": _safe_int(mask_seed),
            "algo_seed": _safe_int(algo_seed),
            "taxon": pivot.index.to_numpy(),
            "delta_mean_before": before_mean - truth_mean,
            "delta_mean_after": after_mean - truth_mean,
            "delta_sd_before": before_sd - truth_sd,
            "delta_sd_after": after_sd - truth_sd,
            "delta_mean_over_sd_before": before_ratio - truth_ratio,
            "delta_mean_over_sd_after": after_ratio - truth_ratio,
        },
        columns=detail_cols,
    )

    ratio_before_mask = np.isfinite(truth_ratio) & np.isfinite(before_ratio)
    ratio_after_mask = np.isfinite(truth_ratio) & np.isfinite(after_ratio)
    marginal_ratio_wd_before = _safe_wasserstein(before_ratio[ratio_before_mask], truth_ratio[ratio_before_mask])
    marginal_ratio_wd_after = _safe_wasserstein(after_ratio[ratio_after_mask], truth_ratio[ratio_after_mask])
    marginal_avg_euclid_before = _mean_euclid_2d(truth_mean, truth_sd, before_mean, before_sd)
    marginal_avg_euclid_after = _mean_euclid_2d(truth_mean, truth_sd, after_mean, after_sd)

    used_mask = (
        (np.isfinite(truth_ratio) & (np.isfinite(before_ratio) | np.isfinite(after_ratio)))
        | (
            np.isfinite(truth_mean) & np.isfinite(truth_sd)
            & (
                (np.isfinite(before_mean) & np.isfinite(before_sd))
                | (np.isfinite(after_mean) & np.isfinite(after_sd))
            )
        )
    )
    df_method = pd.DataFrame(
        [
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": _safe_int(mask_seed),
                "algo_seed": _safe_int(algo_seed),
                "scope": "TaxonMarginal",
                "n_taxa_used": int(np.sum(used_mask)),
                "marginal_ratio_wd_before": marginal_ratio_wd_before,
                "marginal_ratio_wd_after": marginal_ratio_wd_after,
                "marginal_ratio_wd_improve": (
                    marginal_ratio_wd_before - marginal_ratio_wd_after
                    if np.isfinite(marginal_ratio_wd_before) and np.isfinite(marginal_ratio_wd_after)
                    else np.nan
                ),
                "marginal_avg_euclid_before": marginal_avg_euclid_before,
                "marginal_avg_euclid_after": marginal_avg_euclid_after,
                "marginal_avg_euclid_improve": (
                    marginal_avg_euclid_before - marginal_avg_euclid_after
                    if np.isfinite(marginal_avg_euclid_before) and np.isfinite(marginal_avg_euclid_after)
                    else np.nan
                ),
            }
        ],
        columns=method_cols,
    )
    return df_detail, df_method


def compute_taxon_bin_stats(
    da_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    dataset_role: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute taxon-bin abundance summaries across participants."""
    _validate_da(da_abund)
    rows = []
    for taxon in da_abund.coords["taxon"].values:
        for bin_label in da_abund.coords["bin"].values:
            values = da_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            values = values[np.isfinite(values)]
            n_obs = int(values.size)
            abund_mean = float(np.mean(values)) if n_obs > 0 else np.nan
            abund_sd = float(np.std(values, ddof=1)) if n_obs >= 2 else np.nan
            rows.append(
                {
                    "frac": float(frac),
                    "repeat": int(repeat),
                    "method": method,
                    "method_group": _method_group(method),
                    "mask_seed": _safe_int(mask_seed),
                    "algo_seed": _safe_int(algo_seed),
                    "dataset_role": dataset_role,
                    "taxon": taxon,
                    "bin": bin_label,
                    "n_obs": n_obs,
                    "abund_mean_bin": abund_mean,
                    "abund_sd_bin": abund_sd,
                    "abund_mean_over_sd_bin": _safe_ratio(abund_mean, abund_sd),
                }
            )
    return pd.DataFrame(rows, columns=_taxon_bin_stats_columns())


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
    """Evaluate taxon-wise trajectory preservation across bins in abundance space."""
    eval_cols = _taxon_trajectory_eval_columns()
    detail_cols = _taxon_trajectory_detail_columns()
    method_cols = _taxon_trajectory_method_columns()
    if df_taxon_bin_stats.empty:
        return (
            pd.DataFrame(columns=eval_cols),
            pd.DataFrame(columns=detail_cols),
            pd.DataFrame(columns=method_cols),
        )

    required_roles = {"truth", "before", "after"}
    roles_present = set(df_taxon_bin_stats.get("dataset_role", pd.Series(dtype=object)).dropna().astype(str))
    if not required_roles.issubset(roles_present):
        return (
            pd.DataFrame(columns=eval_cols),
            pd.DataFrame(columns=detail_cols),
            pd.DataFrame(columns=method_cols),
        )

    bin_order = list(da_truth_abund.coords["bin"].values)
    pivot = (
        df_taxon_bin_stats.pivot_table(
            index=["taxon", "bin"],
            columns="dataset_role",
            values=["abund_mean_bin", "abund_sd_bin"],
            aggfunc="first",
        )
        .sort_index()
    )

    detail_rows = []
    eval_rows = []
    for taxon in da_truth_abund.coords["taxon"].values:
        mean_true, mean_before, mean_after = [], [], []
        sd_true, sd_before, sd_after = [], [], []
        joint_before, joint_after = [], []
        wd_before_vals, wd_after_vals = [], []

        for bin_label in bin_order:
            idx = (taxon, bin_label)
            if idx not in pivot.index:
                mt = mb = ma = st = sb = sa = np.nan
            else:
                mt = pivot.at[idx, ("abund_mean_bin", "truth")] if ("abund_mean_bin", "truth") in pivot.columns else np.nan
                mb = pivot.at[idx, ("abund_mean_bin", "before")] if ("abund_mean_bin", "before") in pivot.columns else np.nan
                ma = pivot.at[idx, ("abund_mean_bin", "after")] if ("abund_mean_bin", "after") in pivot.columns else np.nan
                st = pivot.at[idx, ("abund_sd_bin", "truth")] if ("abund_sd_bin", "truth") in pivot.columns else np.nan
                sb = pivot.at[idx, ("abund_sd_bin", "before")] if ("abund_sd_bin", "before") in pivot.columns else np.nan
                sa = pivot.at[idx, ("abund_sd_bin", "after")] if ("abund_sd_bin", "after") in pivot.columns else np.nan

            mean_true.append(mt)
            mean_before.append(mb)
            mean_after.append(ma)
            sd_true.append(st)
            sd_before.append(sb)
            sd_after.append(sa)

            euclid_before_bin = _mean_euclid_2d(
                np.asarray([mt], dtype=float),
                np.asarray([st], dtype=float),
                np.asarray([mb], dtype=float),
                np.asarray([sb], dtype=float),
            )
            euclid_after_bin = _mean_euclid_2d(
                np.asarray([mt], dtype=float),
                np.asarray([st], dtype=float),
                np.asarray([ma], dtype=float),
                np.asarray([sa], dtype=float),
            )
            if np.isfinite(euclid_before_bin):
                joint_before.append(euclid_before_bin)
            if np.isfinite(euclid_after_bin):
                joint_after.append(euclid_after_bin)

            truth_vals = da_truth_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            before_vals = da_before_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            after_vals = da_after_abund.sel(taxon=taxon, bin=bin_label).values.astype(float).ravel()
            truth_vals = truth_vals[np.isfinite(truth_vals)]
            before_vals = before_vals[np.isfinite(before_vals)]
            after_vals = after_vals[np.isfinite(after_vals)]

            wd_before_bin = _safe_wasserstein(before_vals, truth_vals)
            wd_after_bin = _safe_wasserstein(after_vals, truth_vals)
            if np.isfinite(wd_before_bin):
                wd_before_vals.append(wd_before_bin)
            if np.isfinite(wd_after_bin):
                wd_after_vals.append(wd_after_bin)

            detail_rows.append(
                {
                    "frac": float(frac),
                    "repeat": int(repeat),
                    "method": method,
                    "method_group": _method_group(method),
                    "mask_seed": _safe_int(mask_seed),
                    "algo_seed": _safe_int(algo_seed),
                    "taxon": taxon,
                    "bin": bin_label,
                    "wd_before_bin": wd_before_bin,
                    "wd_after_bin": wd_after_bin,
                    "euclid_mean_sd_before_bin": euclid_before_bin,
                    "euclid_mean_sd_after_bin": euclid_after_bin,
                }
            )

        mean_true = np.asarray(mean_true, dtype=float)
        mean_before = np.asarray(mean_before, dtype=float)
        mean_after = np.asarray(mean_after, dtype=float)
        sd_true = np.asarray(sd_true, dtype=float)
        sd_before = np.asarray(sd_before, dtype=float)
        sd_after = np.asarray(sd_after, dtype=float)

        traj_mean_l2_before = _l2_distance(mean_true, mean_before)
        traj_mean_l2_after = _l2_distance(mean_true, mean_after)
        traj_sd_l2_before = _l2_distance(sd_true, sd_before)
        traj_sd_l2_after = _l2_distance(sd_true, sd_after)
        traj_joint_before = float(np.mean(joint_before)) if joint_before else np.nan
        traj_joint_after = float(np.mean(joint_after)) if joint_after else np.nan
        taxon_avg_bin_wd_before = float(np.mean(wd_before_vals)) if wd_before_vals else np.nan
        taxon_avg_bin_wd_after = float(np.mean(wd_after_vals)) if wd_after_vals else np.nan

        eval_rows.append(
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": _safe_int(mask_seed),
                "algo_seed": _safe_int(algo_seed),
                "taxon": taxon,
                "traj_mean_l2_before": traj_mean_l2_before,
                "traj_mean_l2_after": traj_mean_l2_after,
                "traj_sd_l2_before": traj_sd_l2_before,
                "traj_sd_l2_after": traj_sd_l2_after,
                "traj_mean_sd_joint_euclid_before": traj_joint_before,
                "traj_mean_sd_joint_euclid_after": traj_joint_after,
                "taxon_avg_bin_wd_before": taxon_avg_bin_wd_before,
                "taxon_avg_bin_wd_after": taxon_avg_bin_wd_after,
                "traj_mean_l2_improve": (
                    traj_mean_l2_before - traj_mean_l2_after
                    if np.isfinite(traj_mean_l2_before) and np.isfinite(traj_mean_l2_after)
                    else np.nan
                ),
                "traj_sd_l2_improve": (
                    traj_sd_l2_before - traj_sd_l2_after
                    if np.isfinite(traj_sd_l2_before) and np.isfinite(traj_sd_l2_after)
                    else np.nan
                ),
                "traj_mean_sd_joint_euclid_improve": (
                    traj_joint_before - traj_joint_after
                    if np.isfinite(traj_joint_before) and np.isfinite(traj_joint_after)
                    else np.nan
                ),
                "taxon_avg_bin_wd_improve": (
                    taxon_avg_bin_wd_before - taxon_avg_bin_wd_after
                    if np.isfinite(taxon_avg_bin_wd_before) and np.isfinite(taxon_avg_bin_wd_after)
                    else np.nan
                ),
            }
        )

    df_eval = pd.DataFrame(eval_rows, columns=eval_cols)
    df_detail = pd.DataFrame(detail_rows, columns=detail_cols)
    metric_cols = [
        "traj_mean_l2_before", "traj_mean_l2_after", "traj_mean_l2_improve",
        "traj_sd_l2_before", "traj_sd_l2_after", "traj_sd_l2_improve",
        "traj_mean_sd_joint_euclid_before", "traj_mean_sd_joint_euclid_after", "traj_mean_sd_joint_euclid_improve",
        "taxon_avg_bin_wd_before", "taxon_avg_bin_wd_after", "taxon_avg_bin_wd_improve",
    ]
    n_taxa_used = int(df_eval[metric_cols].notna().any(axis=1).sum()) if not df_eval.empty else 0
    method_row = {
        "frac": float(frac),
        "repeat": int(repeat),
        "method": method,
        "method_group": _method_group(method),
        "mask_seed": _safe_int(mask_seed),
        "algo_seed": _safe_int(algo_seed),
        "scope": "TaxonTrajectory",
        "n_taxa_used": n_taxa_used,
    }
    for metric in metric_cols:
        method_row[metric] = float(df_eval[metric].mean()) if not df_eval.empty else np.nan
    df_method = pd.DataFrame([method_row], columns=method_cols)
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
    """Evaluate per-bin abundance-structure preservation across taxa."""
    _validate_da(da_truth_abund)
    _validate_da(da_before_abund)
    _validate_da(da_after_abund)

    rows = []
    for bin_label in da_truth_abund.coords["bin"].values:
        truth_slice = da_truth_abund.sel(bin=bin_label).transpose("participant", "taxon").values.astype(float)
        before_slice = da_before_abund.sel(bin=bin_label).transpose("participant", "taxon").values.astype(float)
        after_slice = da_after_abund.sel(bin=bin_label).transpose("participant", "taxon").values.astype(float)

        truth_mean = np.full(truth_slice.shape[1], np.nan, dtype=float)
        truth_sd = np.full(truth_slice.shape[1], np.nan, dtype=float)
        truth_ratio = np.full(truth_slice.shape[1], np.nan, dtype=float)
        before_mean = np.full(before_slice.shape[1], np.nan, dtype=float)
        before_sd = np.full(before_slice.shape[1], np.nan, dtype=float)
        before_ratio = np.full(before_slice.shape[1], np.nan, dtype=float)
        after_mean = np.full(after_slice.shape[1], np.nan, dtype=float)
        after_sd = np.full(after_slice.shape[1], np.nan, dtype=float)
        after_ratio = np.full(after_slice.shape[1], np.nan, dtype=float)

        for idx in range(truth_slice.shape[1]):
            truth_vals = truth_slice[:, idx]
            truth_vals = truth_vals[np.isfinite(truth_vals)]
            before_vals = before_slice[:, idx]
            before_vals = before_vals[np.isfinite(before_vals)]
            after_vals = after_slice[:, idx]
            after_vals = after_vals[np.isfinite(after_vals)]

            if truth_vals.size > 0:
                truth_mean[idx] = float(np.mean(truth_vals))
            if truth_vals.size >= 2:
                truth_sd[idx] = float(np.std(truth_vals, ddof=1))
            truth_ratio[idx] = _safe_ratio(truth_mean[idx], truth_sd[idx])

            if before_vals.size > 0:
                before_mean[idx] = float(np.mean(before_vals))
            if before_vals.size >= 2:
                before_sd[idx] = float(np.std(before_vals, ddof=1))
            before_ratio[idx] = _safe_ratio(before_mean[idx], before_sd[idx])

            if after_vals.size > 0:
                after_mean[idx] = float(np.mean(after_vals))
            if after_vals.size >= 2:
                after_sd[idx] = float(np.std(after_vals, ddof=1))
            after_ratio[idx] = _safe_ratio(after_mean[idx], after_sd[idx])

        ratio_before_mask = np.isfinite(truth_ratio) & np.isfinite(before_ratio)
        ratio_after_mask = np.isfinite(truth_ratio) & np.isfinite(after_ratio)
        bin_ratio_wd_before = _safe_wasserstein(before_ratio[ratio_before_mask], truth_ratio[ratio_before_mask])
        bin_ratio_wd_after = _safe_wasserstein(after_ratio[ratio_after_mask], truth_ratio[ratio_after_mask])
        bin_avg_euclid_before = _mean_euclid_2d(truth_mean, truth_sd, before_mean, before_sd)
        bin_avg_euclid_after = _mean_euclid_2d(truth_mean, truth_sd, after_mean, after_sd)
        used_mask = (
            (np.isfinite(truth_ratio) & (np.isfinite(before_ratio) | np.isfinite(after_ratio)))
            | (
                np.isfinite(truth_mean) & np.isfinite(truth_sd)
                & (
                    (np.isfinite(before_mean) & np.isfinite(before_sd))
                    | (np.isfinite(after_mean) & np.isfinite(after_sd))
                )
            )
        )
        rows.append(
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": _safe_int(mask_seed),
                "algo_seed": _safe_int(algo_seed),
                "bin": bin_label,
                "n_taxa_used": int(np.sum(used_mask)),
                "bin_ratio_wd_before": bin_ratio_wd_before,
                "bin_ratio_wd_after": bin_ratio_wd_after,
                "bin_ratio_wd_improve": (
                    bin_ratio_wd_before - bin_ratio_wd_after
                    if np.isfinite(bin_ratio_wd_before) and np.isfinite(bin_ratio_wd_after)
                    else np.nan
                ),
                "bin_avg_euclid_before": bin_avg_euclid_before,
                "bin_avg_euclid_after": bin_avg_euclid_after,
                "bin_avg_euclid_improve": (
                    bin_avg_euclid_before - bin_avg_euclid_after
                    if np.isfinite(bin_avg_euclid_before) and np.isfinite(bin_avg_euclid_after)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=_bin_distribution_eval_columns())


def build_abundance_eval_repeat_summary(
    df_taxon_marginal_method: pd.DataFrame,
    df_taxon_traj_method: pd.DataFrame,
    df_bin_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Build repeat-level long summary for abundance-space evaluation outputs."""
    rows = []
    if df_taxon_marginal_method is not None and not df_taxon_marginal_method.empty:
        metrics = [
            "n_taxa_used",
            "marginal_ratio_wd_before", "marginal_ratio_wd_after", "marginal_ratio_wd_improve",
            "marginal_avg_euclid_before", "marginal_avg_euclid_after", "marginal_avg_euclid_improve",
        ]
        for _, row in df_taxon_marginal_method.iterrows():
            for metric in metrics:
                rows.append(
                    {
                        "frac": row["frac"],
                        "repeat": row["repeat"],
                        "method": row["method"],
                        "method_group": row["method_group"],
                        "mask_seed": row.get("mask_seed"),
                        "algo_seed": row.get("algo_seed"),
                        "scope": "TaxonMarginal",
                        "metric": metric,
                        "value": row.get(metric, np.nan),
                    }
                )

    if df_taxon_traj_method is not None and not df_taxon_traj_method.empty:
        metrics = [
            "n_taxa_used",
            "traj_mean_l2_before", "traj_mean_l2_after", "traj_mean_l2_improve",
            "traj_sd_l2_before", "traj_sd_l2_after", "traj_sd_l2_improve",
            "traj_mean_sd_joint_euclid_before", "traj_mean_sd_joint_euclid_after", "traj_mean_sd_joint_euclid_improve",
            "taxon_avg_bin_wd_before", "taxon_avg_bin_wd_after", "taxon_avg_bin_wd_improve",
        ]
        for _, row in df_taxon_traj_method.iterrows():
            for metric in metrics:
                rows.append(
                    {
                        "frac": row["frac"],
                        "repeat": row["repeat"],
                        "method": row["method"],
                        "method_group": row["method_group"],
                        "mask_seed": row.get("mask_seed"),
                        "algo_seed": row.get("algo_seed"),
                        "scope": "TaxonTrajectory",
                        "metric": metric,
                        "value": row.get(metric, np.nan),
                    }
                )

    if df_bin_distribution is not None and not df_bin_distribution.empty:
        keys = ["frac", "repeat", "method", "method_group", "mask_seed", "algo_seed"]
        metrics = [
            "n_taxa_used",
            "bin_ratio_wd_before", "bin_ratio_wd_after", "bin_ratio_wd_improve",
            "bin_avg_euclid_before", "bin_avg_euclid_after", "bin_avg_euclid_improve",
        ]
        grouped = df_bin_distribution.groupby(keys, dropna=False)[metrics].mean(numeric_only=True).reset_index()
        for _, row in grouped.iterrows():
            for metric in metrics:
                rows.append(
                    {
                        "frac": row["frac"],
                        "repeat": row["repeat"],
                        "method": row["method"],
                        "method_group": row["method_group"],
                        "mask_seed": row.get("mask_seed"),
                        "algo_seed": row.get("algo_seed"),
                        "scope": "BinDistribution",
                        "metric": metric,
                        "value": row.get(metric, np.nan),
                    }
                )

    return pd.DataFrame(rows, columns=_abundance_eval_summary_columns())


def summarize_abundance_eval_across_repeats(df_repeat_abund: pd.DataFrame) -> pd.DataFrame:
    """Summarize abundance-space repeat-level metrics across repeats."""
    out_cols = ["frac", "scope", "method", "metric", "n", "mean", "sd", "sem", "ci95_low", "ci95_high"]
    if df_repeat_abund.empty:
        return pd.DataFrame(columns=out_cols)

    rows = []
    for (frac, scope, method, metric), g in df_repeat_abund.groupby(["frac", "scope", "method", "metric"], dropna=False):
        stat = _summary_real(g["value"].to_numpy(dtype=float))
        rows.append(
            {
                "frac": float(frac),
                "scope": scope,
                "method": method,
                "metric": metric,
                **stat,
            }
        )
    return pd.DataFrame(rows, columns=out_cols).sort_values(["frac", "scope", "method", "metric"]).reset_index(drop=True)


def merge_abundance_eval_outputs(outdir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge abundance-space repeat outputs without touching legacy summaries."""
    outdir = Path(outdir)
    completed_keys = _completed_repeat_keys(outdir)

    marginal_files = _collect_repeat_files(
        outdir, "taxon_marginal_eval_frac_*_repeat_*.csv", completed_keys
    )
    traj_files = _collect_repeat_files(
        outdir, "taxon_trajectory_eval_frac_*_repeat_*.csv", completed_keys
    )
    bin_files = _collect_repeat_files(
        outdir, "bin_distribution_eval_frac_*_repeat_*.csv", completed_keys
    )
    summary_files = _collect_repeat_files(
        outdir, "abundance_eval_summary_frac_*_repeat_*.csv", completed_keys
    )

    df_marginal = (
        pd.concat([pd.read_csv(p) for p in marginal_files], ignore_index=True)
        if marginal_files else pd.DataFrame(columns=_taxon_marginal_eval_columns())
    )
    df_traj = (
        pd.concat([pd.read_csv(p) for p in traj_files], ignore_index=True)
        if traj_files else pd.DataFrame(columns=_taxon_trajectory_eval_columns())
    )
    df_bin = (
        pd.concat([pd.read_csv(p) for p in bin_files], ignore_index=True)
        if bin_files else pd.DataFrame(columns=_bin_distribution_eval_columns())
    )
    df_repeat = (
        pd.concat([pd.read_csv(p) for p in summary_files], ignore_index=True)
        if summary_files else pd.DataFrame(columns=_abundance_eval_summary_columns())
    )

    marginal_out = outdir / "taxon_marginal_eval_all.csv"
    traj_out = outdir / "taxon_trajectory_eval_all.csv"
    bin_out = outdir / "bin_distribution_eval_all.csv"
    repeat_out = outdir / "abundance_repeat_level_metrics.csv"
    errorbar_out = outdir / "abundance_errorbar_summary.csv"

    if not df_marginal.empty:
        df_marginal = df_marginal.drop_duplicates(subset=["frac", "repeat", "method", "taxon"], keep="last")
        df_marginal = df_marginal.sort_values(["frac", "repeat", "method", "taxon"]).reset_index(drop=True)
        _atomic_to_csv(df_marginal, marginal_out, index=False)
    else:
        marginal_out.unlink(missing_ok=True)
    if not df_traj.empty:
        df_traj = df_traj.drop_duplicates(subset=["frac", "repeat", "method", "taxon"], keep="last")
        df_traj = df_traj.sort_values(["frac", "repeat", "method", "taxon"]).reset_index(drop=True)
        _atomic_to_csv(df_traj, traj_out, index=False)
    else:
        traj_out.unlink(missing_ok=True)
    if not df_bin.empty:
        df_bin = df_bin.drop_duplicates(subset=["frac", "repeat", "method", "bin"], keep="last")
        df_bin["_bin_order"] = df_bin["bin"].map(_bin_key)
        df_bin = (
            df_bin.sort_values(["frac", "repeat", "method", "_bin_order"])
            .drop(columns=["_bin_order"])
            .reset_index(drop=True)
        )
        _atomic_to_csv(df_bin, bin_out, index=False)
    else:
        bin_out.unlink(missing_ok=True)
    if not df_repeat.empty:
        df_repeat = df_repeat.drop_duplicates(subset=["frac", "repeat", "method", "scope", "metric"], keep="last")
        df_repeat = df_repeat.sort_values(["frac", "repeat", "method", "scope", "metric"]).reset_index(drop=True)
        _atomic_to_csv(df_repeat, repeat_out, index=False)
    else:
        repeat_out.unlink(missing_ok=True)

    df_errorbar = summarize_abundance_eval_across_repeats(df_repeat)
    if not df_errorbar.empty:
        _atomic_to_csv(df_errorbar, errorbar_out, index=False)
    else:
        errorbar_out.unlink(missing_ok=True)

    return df_marginal, df_traj, df_bin, df_repeat


if __name__ == "__main__":
    main()
