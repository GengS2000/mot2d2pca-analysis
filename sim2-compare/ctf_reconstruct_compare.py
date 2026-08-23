from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import svd
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from competitor_methods import (  # noqa: E402
    COMPETITOR_GROUPS,
    available_requested_methods,
    impute_competitor,
)

DEFAULT_METHODS = ["penalized_2d2_pca", "ctf", "mean", "nearest_positive", "temporal_knn"]
SIM2_SUPPORTED_METHODS = frozenset(DEFAULT_METHODS)
METHOD_GROUP_MAP = {
    "penalized_2d2_pca": "matrix_factorization",
    "mean": "naive",
    "nearest_positive": "local_time",
    "temporal_knn": "local_time",
    "ctf": COMPETITOR_GROUPS["ctf"],
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
        from gemelli import factorization
    except Exception:
        return None

    rclr_da = kwargs["rclr_da"]
    observed_mask = kwargs["observed_mask"]
    rank = int(kwargs["rank"])
    max_iter = int(kwargs["max_iter"])
    seed = int(kwargs["seed"])
    transformed = np.where(observed_mask.values.astype(bool), rclr_da.values.astype(float), 0.0)
    mask = observed_mask.values.astype(float)

    original_rtpm = factorization.robust_tensor_power_method

    def seeded_rtpm(tensor, n_components, n_initializations=50,
                    max_rtpm_iterations=50, tol_rtpm=1e-10, random_state=42):
        return original_rtpm(
            tensor,
            n_components,
            n_initializations=n_initializations,
            max_rtpm_iterations=max_rtpm_iterations,
            tol_rtpm=tol_rtpm,
            random_state=seed,
        )

    factorization.robust_tensor_power_method = seeded_rtpm
    try:
        loadings, eigvals, _ = factorization.tenals(
            transformed,
            mask,
            n_components=max(1, min(rank, min(transformed.shape))),
            n_initializations=10,
            max_als_iterations=max_iter,
            max_rtpm_iterations=max_iter,
            tol_als=float(kwargs["tol"]),
            tol_rtpm=float(kwargs["tol"]),
            fillna=1.0,
        )
        component_weights = np.diag(eigvals) if np.ndim(eigvals) == 2 else np.asarray(eigvals)
        t_hat = np.asarray(factorization.construct_tensor(component_weights, loadings), dtype=float)
        factor_list = [np.asarray(value, dtype=float) for value in loadings]
        if len(factor_list) != 3:
            raise RuntimeError(f"Gemelli returned {len(factor_list)} loading matrices; expected 3")
        observed_loss = _observed_loss(
            rclr_da.values.astype(float), observed_mask.values.astype(bool), t_hat
        )
        return {
            "T_hat": t_hat,
            "A": factor_list[0],
            "B": factor_list[1],
            "C": factor_list[2],
            "sigma": np.asarray(component_weights, dtype=float),
            "n_iter": max_iter,
            "final_loss": float(observed_loss),
            "converged": True,
        }
    except Exception as exc:
        warnings.warn(f"Official Gemelli tenals path failed: {exc}", RuntimeWarning)
        return None
    finally:
        factorization.robust_tensor_power_method = original_rtpm


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
        elif method == "gemelli":
            raise RuntimeError("Official Gemelli tenals path is unavailable or failed; fallback is disabled.")

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
        pen_weight = 0.0
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
                pen_weight += float(wij)
        if pen_weight > 0.0:
            pen /= pen_weight
        astar = astar - lam * pen
        x = _top_eigvecs(astar, d_eff)

        xxt = x @ x.T
        astarstar = np.zeros((m, m), dtype=float)
        for i in range(m_time):
            bi = ac[i]
            astarstar += bi @ xxt @ bi.T
        astarstar /= max(m_time, 1)

        pen2 = np.zeros((m, m), dtype=float)
        pen2_weight = 0.0
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
                pen2_weight += float(wij)
        if pen2_weight > 0.0:
            pen2 /= pen2_weight
        astarstar = astarstar - lam * pen2
        z = _top_eigvecs(astarstar, q_eff)

    c = np.zeros((m_time, q_eff, d_eff), dtype=float)
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
    require_existing: bool = False,
):
    truth_path = Path(truth_path)
    if require_existing and not truth_path.is_file():
        raise FileNotFoundError(
            f"Frozen Sim2 truth does not exist: {truth_path}. "
            "Array shards are read-only and are not allowed to rebuild truth."
        )
    if truth_path.exists() and (not force_rebuild):
        da_gt = _load_dataarray(truth_path, var_name="y_true")
        da_gt.name = "y_true"
        print(f"[truth] loaded existing truth from {truth_path}")
        return da_gt
    if require_existing:
        raise RuntimeError("Internal error: required frozen truth was not loaded")

    out = ctf_complete_tensor_from_log_da(
        da=base_da,
        observed_mask=base_observed_mask,
        eps=eps,
        rank=rank,
        method=method,
        seed=ctf_seed,
        verbose=ctf_verbose,
    )
    da_gt = out["ctf_imp"].copy(deep=True)
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


def _apply_masked_candidate(
    da_input: xr.DataArray,
    candidate_da: xr.DataArray,
    replace_mask: xr.DataArray,
) -> xr.DataArray:
    """
    Apply candidate values only on masked cells for the current random cell-mask task.

    Legacy comparators may internally construct full participant-bin candidate vectors, but
    only `replace_mask=True` cells are allowed to change in the final returned tensor.
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


def _to_relative_abundance_space(
    da_log: xr.DataArray,
    eps: float,
    name: str = "relative_abundance_space",
) -> xr.DataArray:
    """
    Convert log10(percent+eps) into per-participant/per-bin relative abundance.

    This is used by legacy comparator methods whose reference definitions operate on
    day vectors in relative-abundance space. The external task remains random cell-mask;
    the conversion only changes the internal modeling scale.
    """
    da_abund = to_abundance_space(da_log, eps=eps, name=f"{name}_abundance")
    vals = da_abund.values.astype(float)
    denom = np.sum(vals, axis=1, keepdims=True)
    rel = np.divide(vals, denom, out=np.zeros_like(vals), where=denom > 0)
    return xr.DataArray(rel, coords=da_log.coords, dims=da_log.dims, name=name)


def _relative_to_log_abundance_space(
    da_rel: xr.DataArray,
    eps: float,
    name: str = "log_abundance_imputed",
) -> xr.DataArray:
    """Convert relative abundance back to the script's log10(percent+eps) scale."""
    vals = np.asarray(da_rel.values, dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    vals = np.where(vals < 0.0, 0.0, vals)
    pct = 100.0 * vals
    log_vals = np.log10(pct + float(eps))
    return xr.DataArray(log_vals, coords=da_rel.coords, dims=da_rel.dims, name=name)


def _mean_initialize_relative_tensor(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    eps: float,
) -> xr.DataArray:
    """
    Mean-initialize masked cells in relative-abundance space for internal legacy modeling.

    The task remains random cell-mask imputation. This initialization is only used to build
    complete participant trajectories for legacy dynamic methods and does not change which
    cells are ultimately eligible for write-back.
    """
    da_rel = _to_relative_abundance_space(da_input, eps=eps, name="relative_abundance_init")
    arr = da_rel.transpose("participant", "taxon", "bin").values.astype(float).copy()
    mask = replace_mask.transpose("participant", "taxon", "bin").values.astype(bool)
    n_p, n_j, _ = arr.shape
    for p in range(n_p):
        for j in range(n_j):
            obs = arr[p, j, ~mask[p, j, :]]
            finite = obs[np.isfinite(obs)]
            fill = float(np.mean(finite)) if finite.size > 0 else 0.0
            arr[p, j, mask[p, j, :]] = fill
    denom = np.sum(arr, axis=1, keepdims=True)
    arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)
    return xr.DataArray(
        arr,
        coords=da_rel.coords,
        dims=da_rel.dims,
        name="relative_abundance_mean_initialized",
    )


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


def _fill_nearest_positive(times: np.ndarray, observed_idx: np.ndarray, observed_vals: np.ndarray, target_time: float, fallback_value: float) -> float:
    if observed_idx.size == 0:
        return float(fallback_value)
    obs_times = times[observed_idx]
    distance = np.abs(obs_times - float(target_time))
    min_distance = float(np.min(distance))
    tied = np.flatnonzero(np.isclose(distance, min_distance, rtol=0.0, atol=1e-12))
    chosen = int(tied[np.argmin(obs_times[tied])])
    return float(observed_vals[chosen])


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
    return _fill_nearest_positive(times, observed_idx, observed_vals, target_time, fallback_value)


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
        elif method == "median":
            out[idx] = _trajectory_fill_constant(obs_vals, "median", default_value=zero_value)
        elif method == "nearest_positive":
            out[idx] = _fill_nearest_positive(times, obs_idx, obs_vals, t, fallback_value=target_fallback)
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


def _legacy_predict_relative_step(
    x: np.ndarray,
    delta_t: float,
    self_growth: np.ndarray,
    interaction_matrix: np.ndarray,
    perturbation_matrix: Optional[np.ndarray] = None,
    perturbation_incidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Predict one forward relative-abundance step using the legacy gLV discretization.

    This is mathematically aligned with the reference file's day-by-day prediction rule, but
    implemented in a directly runnable form and restricted to the no-perturbation path for now.
    """
    x_vec = np.asarray(x, dtype=float).reshape(-1, 1)
    mue = np.asarray(self_growth, dtype=float).reshape(-1, 1)
    mat = np.asarray(interaction_matrix, dtype=float)
    if perturbation_matrix is not None or perturbation_incidence is not None:
        raise NotImplementedError("Legacy perturbation paths are not supported in this script.")
    dx = mue + mat @ x_vec
    nxt = np.exp(float(delta_t) * dx) * x_vec
    nxt = np.where(np.isfinite(nxt) & (nxt > 0.0), nxt, 0.0)
    total = float(np.sum(nxt))
    if total <= 0.0:
        return np.full((x_vec.shape[0], 1), np.nan, dtype=float)
    return nxt / total


def _legacy_create_y_matrix(X: np.ndarray, pert_incidence: Optional[np.ndarray] = None) -> np.ndarray:
    ones = np.ones((1, X.shape[1] - 1), dtype=float)
    y_mat = np.concatenate((X[:, 1:], ones), axis=0)
    if pert_incidence is not None:
        y_mat = np.concatenate((y_mat, pert_incidence), axis=0)
    return y_mat


def _legacy_create_F_matrix(X: np.ndarray, time_points: np.ndarray) -> np.ndarray:
    if X.shape[1] < 2:
        return np.zeros((X.shape[0], 0), dtype=float)
    cols = []
    for idx in range(X.shape[1] - 1):
        dt = float(time_points[idx + 1] - time_points[idx])
        if dt <= 0:
            raise ValueError("time_points must be strictly increasing for legacy gLV methods.")
        ratio = np.divide(
            X[:, idx + 1],
            X[:, idx],
            out=np.ones(X.shape[0], dtype=float),
            where=X[:, idx] > 0,
        )
        f_vec = np.log(ratio).reshape(-1, 1)
        f_vec = np.nan_to_num(f_vec, nan=0.0, posinf=0.0, neginf=0.0) / dt
        cols.append(f_vec)
    return np.concatenate(cols, axis=1) if cols else np.zeros((X.shape[0], 0), dtype=float)


def _legacy_mse_glv(F: np.ndarray, Y_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    gram = Y_mat @ Y_mat.T
    coef = np.linalg.solve(gram, Y_mat @ F.T)
    m = F.shape[0]
    M = coef[:m, :].T
    mue = coef[m: m + 1, :].T
    pert = coef[m + 1:, :].T if Y_mat.shape[0] > m + 1 else None
    return M, mue, pert


def _legacy_coef_matrix_by_paper_trick_ridge(
    Y: np.ndarray,
    F: np.ndarray,
    lambda_M: float,
    lambda_mue: float,
    lambda_E: float,
) -> np.ndarray:
    num_spec = F.shape[0]
    num_pert = Y.shape[0] - num_spec - 1
    diag = [float(lambda_M)] * num_spec + [float(lambda_mue)] + [float(lambda_E)] * max(num_pert, 0)
    D = np.diagflat(diag)
    return F @ (Y.T @ np.linalg.solve((Y @ Y.T) + D, np.eye(Y.shape[0])))


def _legacy_parse_ridge_coef(
    coef_mat: np.ndarray,
    num_species: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    M = coef_mat[:, :num_species]
    mue = coef_mat[:, num_species:num_species + 1]
    pert = coef_mat[:, num_species + 1:] if coef_mat.shape[1] > num_species + 1 else None
    return M, mue, pert


def _legacy_fit_line_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _legacy_limits(
    F: np.ndarray,
    data: np.ndarray,
    error_threshold: float,
    repeats: int,
    med_or_avg: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runnable LIMITS implementation matching the reference math while fixing obvious Python issues.

    The reference file contains implementation bugs such as invalid random shuffling and loop forms.
    This version preserves the greedy feature-selection logic and aggregation rule.
    """
    rng = np.random.default_rng(seed)
    n_spec, n_obs = F.shape
    if n_obs < 2:
        raise ValueError("LIMITS requires at least two transitions.")
    all_M, all_mue = [], []
    for _ in range(max(int(repeats), 1)):
        idx = np.arange(n_obs)
        rng.shuffle(idx)
        split = max(1, n_obs // 2)
        if split >= n_obs:
            split = n_obs - 1
        train_idx = np.sort(idx[:split])
        test_idx = np.sort(idx[split:])
        train_F = F[:, train_idx]
        train_data = data[:, train_idx]
        test_F = F[:, test_idx]
        test_data = data[:, test_idx]
        tmp_mue = np.zeros((n_spec, 1), dtype=float)
        tmp_M = np.zeros((n_spec, n_spec), dtype=float)
        for main_spec in range(n_spec):
            target_train = train_F[main_spec, :].astype(float).copy()
            target_test = test_F[main_spec, :].astype(float)
            inactive = [i for i in range(n_spec)]

            ones_train = np.ones_like(target_train)
            tmp_mue[main_spec, 0] = _legacy_fit_line_no_intercept(ones_train, target_train)
            target_train = target_train - tmp_mue[main_spec, 0]
            pred_test = np.full(test_F.shape[1], tmp_mue[main_spec, 0], dtype=float)

            inactive.remove(main_spec)
            x_train = train_data[main_spec, :].astype(float)
            self_coef = _legacy_fit_line_no_intercept(x_train, target_train)
            tmp_M[main_spec, main_spec] = self_coef
            target_train = target_train - x_train * self_coef
            pred_test = pred_test + test_data[main_spec, :].astype(float) * self_coef
            error = float(np.mean((pred_test - target_test) ** 2))

            improve = True
            while improve and inactive:
                best_error = np.inf
                best_spec = None
                best_coeff = 0.0
                for spec_idx in inactive:
                    x_train = train_data[spec_idx, :].astype(float)
                    coeff = _legacy_fit_line_no_intercept(x_train, target_train)
                    tmp_pred = pred_test + test_data[spec_idx, :].astype(float) * coeff
                    tmp_error = float(np.mean((tmp_pred - target_test) ** 2))
                    if tmp_error < best_error:
                        best_error = tmp_error
                        best_spec = spec_idx
                        best_coeff = coeff
                if best_spec is None:
                    break
                improve_ratio = (error - best_error) / max(abs(error), 1e-12)
                if improve_ratio >= float(error_threshold):
                    tmp_M[main_spec, best_spec] = best_coeff
                    inactive.remove(best_spec)
                    error = best_error
                    target_train = target_train - train_data[best_spec, :].astype(float) * best_coeff
                    pred_test = pred_test + test_data[best_spec, :].astype(float) * best_coeff
                else:
                    improve = False
        all_mue.append(tmp_mue)
        all_M.append(tmp_M)

    all_mue_arr = np.stack(all_mue, axis=0)
    all_M_arr = np.stack(all_M, axis=0)
    reducer = np.median if int(med_or_avg) == 0 else np.mean
    final_mue = reducer(all_mue_arr, axis=0)
    final_M = reducer(all_M_arr, axis=0)
    return final_mue, final_M


def _apply_legacy_day_vector_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    method: str,
    params: Dict,
    eps: float,
) -> xr.DataArray:
    """
    Legacy local-time comparators for the random cell-mask task.

    Each method may construct a full candidate vector for a participant-bin day, but final
    write-back is restricted to `replace_mask=True` cells only.
    """
    da_rel = _to_relative_abundance_space(da_input, eps=eps, name="legacy_local_rel")
    rel = da_rel.transpose("participant", "taxon", "bin").values.astype(float)
    mask = replace_mask.transpose("participant", "taxon", "bin").values.astype(bool)
    time_vals = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
    candidate = np.full_like(rel, np.nan, dtype=float)

    for p in range(rel.shape[0]):
        participant_matrix = rel[p, :, :]
        participant_mask = mask[p, :, :]
        for t_idx in range(rel.shape[2]):
            if not np.any(participant_mask[:, t_idx]):
                continue
            obs_cols = np.flatnonzero(~np.any(participant_mask, axis=0))
            if method == "weighted_avg_legacy":
                if t_idx == 0 or t_idx == rel.shape[2] - 1:
                    continue
                if (t_idx - 1 not in obs_cols) or (t_idx + 1 not in obs_cols):
                    continue
                prev_time = float(time_vals[t_idx - 1])
                cur_time = float(time_vals[t_idx])
                next_time = float(time_vals[t_idx + 1])
                denom = next_time - prev_time
                if denom <= 0:
                    continue
                prev_vec = participant_matrix[:, t_idx - 1]
                next_vec = participant_matrix[:, t_idx + 1]
                cand = (
                    next_vec * (cur_time - prev_time)
                    + prev_vec * (next_time - cur_time)
                ) / denom
            elif method == "spline_legacy":
                if obs_cols.size < max(int(params.get("min_points", 4)), 4):
                    continue
                cand = np.full(participant_matrix.shape[0], np.nan, dtype=float)
                train_times = time_vals[obs_cols]
                if np.unique(train_times).size < 4:
                    continue
                try:
                    for j in range(participant_matrix.shape[0]):
                        f = interp1d(train_times, participant_matrix[j, obs_cols], kind="cubic")
                        pred = float(f(time_vals[t_idx]))
                        cand[j] = max(0.0, pred)
                except Exception:
                    continue
            elif method == "knn_epanechnikov_legacy":
                if obs_cols.size == 0:
                    continue
                k = max(1, int(params.get("k", 3)))
                obs_times = time_vals[obs_cols]
                dist_sorted = np.sort(np.abs(obs_times - time_vals[t_idx]))
                kth_index = min(k, dist_sorted.size - 1)
                bandwidth = float(dist_sorted[kth_index])
                if bandwidth <= 0:
                    continue
                cand = np.zeros(participant_matrix.shape[0], dtype=float)
                used = False
                for col in obs_cols:
                    dist = abs(float(time_vals[col] - time_vals[t_idx]))
                    if dist <= bandwidth:
                        ker_norm = dist / bandwidth
                        kernel = 0.75 * (1.0 - ker_norm**2)
                        cand += kernel * participant_matrix[:, col]
                        used = True
                if not used:
                    continue
            else:
                raise ValueError(f"Unsupported legacy local-time method: {method}")

            if not np.all(np.isfinite(cand)):
                continue
            total = float(np.sum(cand))
            if total <= 0:
                continue
            candidate[p, :, t_idx] = cand / total

    da_candidate_rel = xr.DataArray(
        candidate,
        coords=da_rel.coords,
        dims=da_rel.dims,
        name="legacy_local_candidate_rel",
    )
    da_candidate_log = _relative_to_log_abundance_space(da_candidate_rel, eps=eps)
    return _apply_masked_candidate(da_input, da_candidate_log, replace_mask)


def _apply_legacy_glv_family_method(
    da_input: xr.DataArray,
    replace_mask: xr.DataArray,
    method: str,
    params: Dict,
    eps: float,
    model_seed: int,
) -> xr.DataArray:
    """
    Legacy gLV-family comparators for the random cell-mask task.

    The model is fitted on mean-initialized relative-abundance trajectories, but only masked
    cells are overwritten in the final tensor.
    """
    da_rel_init = _mean_initialize_relative_tensor(da_input, replace_mask, eps=eps)
    rel_init = da_rel_init.transpose("participant", "taxon", "bin").values.astype(float)
    time_vals = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
    candidate = np.full_like(rel_init, np.nan, dtype=float)

    for p in range(rel_init.shape[0]):
        X = rel_init[p, :, :]
        if X.shape[1] < 2:
            continue
        try:
            F = _legacy_create_F_matrix(X, time_vals)
            Y = _legacy_create_y_matrix(X, pert_incidence=None)
            if F.shape[1] == 0:
                continue
            if method == "glv_mse_legacy":
                M, mue, _pert = _legacy_mse_glv(F, Y)
                pert = None
            elif method == "glv_ridge_legacy":
                coef_mat = _legacy_coef_matrix_by_paper_trick_ridge(
                    Y=Y,
                    F=F,
                    lambda_M=float(params.get("lambda_M", 1e-3)),
                    lambda_mue=float(params.get("lambda_mue", 1e-3)),
                    lambda_E=float(params.get("lambda_E", 1e-3)),
                )
                M, mue, pert = _legacy_parse_ridge_coef(coef_mat, num_species=X.shape[0])
            elif method == "limits_legacy":
                Mue, M = _legacy_limits(
                    F=F,
                    data=X[:, :-1],
                    error_threshold=float(params.get("error_threshold", 0.05)),
                    repeats=int(params.get("repeats", 20)),
                    med_or_avg=int(params.get("med_or_avg", 0)),
                    seed=int(model_seed) + p,
                )
                mue = Mue
                pert = None
            else:
                raise ValueError(f"Unsupported legacy dynamic method: {method}")

            pred = np.full_like(X, np.nan, dtype=float)
            pred[:, 0] = X[:, 0]
            for t_idx in range(1, X.shape[1]):
                dt = float(time_vals[t_idx] - time_vals[t_idx - 1])
                if dt <= 0:
                    raise ValueError("time_points must be strictly increasing for legacy gLV prediction.")
                pred_step = _legacy_predict_relative_step(
                    x=pred[:, t_idx - 1],
                    delta_t=dt,
                    self_growth=mue,
                    interaction_matrix=M,
                    perturbation_matrix=pert,
                    perturbation_incidence=None,
                )
                pred[:, t_idx] = pred_step.ravel()
            candidate[p, :, :] = pred
        except Exception:
            continue

    da_candidate_rel = xr.DataArray(
        candidate,
        coords=da_rel_init.coords,
        dims=da_rel_init.dims,
        name="legacy_dynamic_candidate_rel",
    )
    da_candidate_log = _relative_to_log_abundance_space(da_candidate_rel, eps=eps)
    return _apply_masked_candidate(da_input, da_candidate_log, replace_mask)


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
    eps: float = 1e-3,
    method_params: Optional[Dict] = None,
    pca_params: Optional[Dict] = None,
    observed_mode: str = "positive_only_except_replace",
    verbose: bool = False,
) -> Tuple[xr.DataArray, np.ndarray]:
    method = str(method)
    method_params = {} if method_params is None else dict(method_params)
    bins_use = np.array(da_input.coords["bin"].values)
    if method == "penalized_2d2_pca":
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
    elif method == "ctf":
        observed_mask = build_observed_mask(
            da_input=da_input,
            zero_value=zero_value,
            mode=observed_mode,
            replace_mask=replace_mask,
        )
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
        bins_trained = bins_use
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
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    da_imp = da_imp.transpose(*da_input.dims)
    off_mask = ~replace_mask.transpose(*da_input.dims).values.astype(bool)
    if not np.array_equal(da_imp.values[off_mask], da_input.values[off_mask]):
        raise AssertionError(f"method={method} modified an off-mask cell")
    da_imp.name = "log_abundance_imputed"
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
    methods = list(methods) if methods is not None else list(DEFAULT_METHODS)
    pca_params = {"d": d, "q": q, "lam": lam, "sigma": sigma, "T0": T0, "T": T}
    impute_method_params = {} if impute_method_params is None else impute_method_params

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
            eps=float(eps),
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

        init_method = "mean" if method in {"dbn_dense", "dbn_sparse", "glv", "glv_mse_legacy", "glv_ridge_legacy", "limits_legacy"} else ""
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

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frac_str = _format_frac(frac)
    repeat_str = f"{int(repeat):03d}"
    df_bin_one = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    df_repeat_one = pd.concat(repeat_parts, ignore_index=True) if repeat_parts else pd.DataFrame()

    bin_path = outdir / f"repeat_metrics_frac_{frac_str}_repeat_{repeat_str}.csv"
    repeat_path = outdir / f"repeat_summary_frac_{frac_str}_repeat_{repeat_str}.csv"
    df_bin_one.to_csv(bin_path, index=False)
    df_repeat_one.to_csv(repeat_path, index=False)

    if save_imputed_matrix:
        for method, da_impf in imputed_outputs.items():
            _save_dataarray(
                da_impf, "y_imp",
                outdir / f"imputed_{method}_frac_{frac_str}_repeat_{repeat_str}",
                save_format=save_format,
                compress_level=compress_level,
            )

    abund_marginal_stats_parts = []
    abund_marginal_eval_parts = []
    abund_bin_stats_parts = []
    abund_traj_eval_parts = []
    abund_traj_detail_parts = []
    abund_bin_dist_parts = []
    abund_repeat_summary_parts = []
    da_gt_abund = to_abundance_space(da_gt, eps=eps, name="truth_abundance_space")
    da_before_abund = to_abundance_space(da_zi, eps=eps, name="before_abundance_space")

    for method in methods:
        da_after_abund = to_abundance_space(
            imputed_outputs[method],
            eps=eps,
            name=f"{method}_abundance_space",
        )
        try:
            df_taxon_stats = pd.concat(
                [
                    compute_taxon_marginal_stats(
                        da_gt_abund,
                        frac=float(frac),
                        repeat=int(repeat),
                        method=method,
                        dataset_role="truth",
                        mask_seed=int(mask_seed),
                        algo_seed=int(algo_seed),
                    ),
                    compute_taxon_marginal_stats(
                        da_before_abund,
                        frac=float(frac),
                        repeat=int(repeat),
                        method=method,
                        dataset_role="before",
                        mask_seed=int(mask_seed),
                        algo_seed=int(algo_seed),
                    ),
                    compute_taxon_marginal_stats(
                        da_after_abund,
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
            df_taxon_eval, df_taxon_method = evaluate_taxon_marginal_preservation(
                df_taxon_stats=df_taxon_stats,
                frac=float(frac),
                repeat=int(repeat),
                method=method,
                mask_seed=int(mask_seed),
                algo_seed=int(algo_seed),
            )

            df_taxon_bin_stats = pd.concat(
                [
                    compute_taxon_bin_stats(
                        da_gt_abund,
                        frac=float(frac),
                        repeat=int(repeat),
                        method=method,
                        dataset_role="truth",
                        mask_seed=int(mask_seed),
                        algo_seed=int(algo_seed),
                    ),
                    compute_taxon_bin_stats(
                        da_before_abund,
                        frac=float(frac),
                        repeat=int(repeat),
                        method=method,
                        dataset_role="before",
                        mask_seed=int(mask_seed),
                        algo_seed=int(algo_seed),
                    ),
                    compute_taxon_bin_stats(
                        da_after_abund,
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
            df_traj_eval, df_traj_detail, df_traj_method = evaluate_taxon_trajectory_preservation(
                df_taxon_bin_stats=df_taxon_bin_stats,
                da_truth_abund=da_gt_abund,
                da_before_abund=da_before_abund,
                da_after_abund=da_after_abund,
                frac=float(frac),
                repeat=int(repeat),
                method=method,
                mask_seed=int(mask_seed),
                algo_seed=int(algo_seed),
            )
            df_bin_dist = evaluate_bin_distribution_preservation(
                da_truth_abund=da_gt_abund,
                da_before_abund=da_before_abund,
                da_after_abund=da_after_abund,
                frac=float(frac),
                repeat=int(repeat),
                method=method,
                mask_seed=int(mask_seed),
                algo_seed=int(algo_seed),
            )
            df_abund_repeat = build_abundance_eval_repeat_summary(
                df_taxon_marginal_method=df_taxon_method,
                df_taxon_traj_method=df_traj_method,
                df_bin_distribution=df_bin_dist,
            )

            abund_marginal_stats_parts.append(df_taxon_stats)
            abund_marginal_eval_parts.append(df_taxon_eval)
            abund_bin_stats_parts.append(df_taxon_bin_stats)
            abund_traj_eval_parts.append(df_traj_eval)
            abund_traj_detail_parts.append(df_traj_detail)
            abund_bin_dist_parts.append(df_bin_dist)
            abund_repeat_summary_parts.append(df_abund_repeat)
        except Exception as e:
            warnings.warn(
                f"abundance evaluation failed for method={method}, frac={frac}, repeat={repeat}: {e}",
                RuntimeWarning,
            )

    df_taxon_marginal_stats = (
        pd.concat(abund_marginal_stats_parts, ignore_index=True)
        if abund_marginal_stats_parts else _empty_taxon_marginal_stats_df()
    )
    df_taxon_marginal_eval = (
        pd.concat(abund_marginal_eval_parts, ignore_index=True)
        if abund_marginal_eval_parts else _empty_taxon_marginal_eval_df()
    )
    df_taxon_bin_stats = (
        pd.concat(abund_bin_stats_parts, ignore_index=True)
        if abund_bin_stats_parts else _empty_taxon_bin_stats_df()
    )
    df_taxon_traj_eval = (
        pd.concat(abund_traj_eval_parts, ignore_index=True)
        if abund_traj_eval_parts else _empty_taxon_trajectory_eval_df()
    )
    df_taxon_traj_detail = (
        pd.concat(abund_traj_detail_parts, ignore_index=True)
        if abund_traj_detail_parts else _empty_taxon_trajectory_detail_df()
    )
    df_bin_distribution = (
        pd.concat(abund_bin_dist_parts, ignore_index=True)
        if abund_bin_dist_parts else _empty_bin_distribution_eval_df()
    )
    df_abundance_eval_summary = (
        pd.concat(abund_repeat_summary_parts, ignore_index=True)
        if abund_repeat_summary_parts else _empty_abundance_eval_summary_df()
    )

    df_taxon_marginal_eval.to_csv(
        outdir / f"taxon_marginal_eval_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    df_taxon_traj_eval.to_csv(
        outdir / f"taxon_trajectory_eval_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    df_bin_distribution.to_csv(
        outdir / f"bin_distribution_eval_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    df_abundance_eval_summary.to_csv(
        outdir / f"abundance_eval_summary_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    if not compact_output:
        df_taxon_marginal_stats.to_csv(
            outdir / f"taxon_marginal_stats_frac_{frac_str}_repeat_{repeat_str}.csv",
            index=False,
        )
        df_taxon_bin_stats.to_csv(
            outdir / f"taxon_bin_stats_frac_{frac_str}_repeat_{repeat_str}.csv",
            index=False,
        )
        df_taxon_traj_detail.to_csv(
            outdir / f"taxon_trajectory_detail_frac_{frac_str}_repeat_{repeat_str}.csv",
            index=False,
        )

    injected_eval = int(df_repeat_one.loc[df_repeat_one["scope"].eq("InjectedCells"), "n_eval"].sum())
    print(
        f"[repeat done] frac={frac:.4f} repeat={repeat} mask_seed={mask_seed} "
        f"algo_seed={algo_seed} sampled={sampled_cells} eval_cells={injected_eval} methods={','.join(methods)}"
    )
    return df_bin_one, df_repeat_one


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
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bin_parts, repeat_parts, fail_rows = [], [], []
    for r in range(repeat_start, repeat_start + n_repeats):
        try:
            df_bin_one, df_repeat_one = run_one_repeat(
                da_gt=da_gt,
                frac=frac,
                repeat=r,
                mask_seed=mask_seed_base + r,
                algo_seed=algo_seed,
                outdir=outdir,
                d=d, q=q, lam=lam, sigma=sigma, T0=T0, T=T,
                save_imputed_matrix=save_imputed_matrix,
                save_format=save_format,
                compress_level=compress_level,
                verbose=verbose,
                eps=eps,
                methods=methods,
                impute_method_params=impute_method_params,
            )
            bin_parts.append(df_bin_one)
            repeat_parts.append(df_repeat_one)
        except Exception as e:
            err_text = traceback.format_exc()
            frac_str = _format_frac(frac)
            repeat_str = f"{int(r):03d}"
            fail_path = outdir / f"repeat_failed_frac_{frac_str}_repeat_{repeat_str}.log"
            fail_path.write_text(err_text, encoding="utf-8")
            fail_rows.append(
                {
                    "frac": float(frac),
                    "repeat": int(r),
                    "mask_seed": int(mask_seed_base + r),
                    "algo_seed": int(algo_seed),
                    "error": str(e),
                    "log_path": str(fail_path),
                }
            )
            print(f"[repeat failed] frac={frac:.4f} repeat={r}, log={fail_path}")

    df_bin_all = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    df_repeat_all = pd.concat(repeat_parts, ignore_index=True) if repeat_parts else pd.DataFrame()
    df_fail = pd.DataFrame(fail_rows)

    frac_str = _format_frac(frac)
    if not df_bin_all.empty:
        df_bin_all.to_csv(outdir / f"frac_{frac_str}_bin_metrics_all_repeats.csv", index=False)
    if not df_repeat_all.empty:
        df_repeat_all.to_csv(outdir / f"frac_{frac_str}_repeat_metrics_all_repeats.csv", index=False)
    if not df_fail.empty:
        df_fail.to_csv(outdir / f"frac_{frac_str}_repeat_failures.csv", index=False)

    return df_bin_all, df_repeat_all, df_fail


def merge_repeat_outputs(outdir):
    outdir = Path(outdir)
    repeat_files = sorted(outdir.glob("repeat_summary_frac_*_repeat_*.csv"))
    bin_files = sorted(outdir.glob("repeat_metrics_frac_*_repeat_*.csv"))

    df_repeat = pd.concat([pd.read_csv(p) for p in repeat_files], ignore_index=True) if repeat_files else pd.DataFrame()
    df_bin = pd.concat([pd.read_csv(p) for p in bin_files], ignore_index=True) if bin_files else pd.DataFrame()
    if not df_repeat.empty and "method" not in df_repeat.columns:
        df_repeat["method"] = "penalized_2d2_pca"
    if not df_bin.empty and "method" not in df_bin.columns:
        df_bin["method"] = "penalized_2d2_pca"
    if not df_repeat.empty:
        df_repeat = df_repeat.drop_duplicates(subset=["frac", "repeat", "method", "scope"], keep="last")
    if not df_bin.empty:
        df_bin = df_bin.drop_duplicates(subset=["frac", "repeat", "method", "scope", "bin"], keep="last")

    df_errorbar = summarize_across_repeats(df_repeat)

    repeat_path = outdir / "repeat_level_metrics.csv"
    errorbar_path = outdir / "errorbar_summary.csv"
    bin_path = outdir / "repeat_bin_metrics_all.csv"
    if not df_repeat.empty:
        df_repeat.to_csv(repeat_path, index=False)
    if not df_errorbar.empty:
        df_errorbar.to_csv(errorbar_path, index=False)
    if not df_bin.empty:
        df_bin.to_csv(bin_path, index=False)

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
):
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
    df_errorbar = summarize_across_repeats(df_repeat_all)

    save_dir = Path(save_dir)
    if not df_bin_all.empty:
        df_bin_all.to_csv(save_dir / "repeat_bin_metrics_all.csv", index=False)
    if not df_repeat_all.empty:
        df_repeat_all.to_csv(save_dir / "repeat_level_metrics.csv", index=False)
    if not df_errorbar.empty:
        df_errorbar.to_csv(save_dir / "errorbar_summary.csv", index=False)
    if not df_fail_all.empty:
        df_fail_all.to_csv(save_dir / "repeat_failures_all.csv", index=False)
    merge_abundance_eval_outputs(save_dir)

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
    allowed = set(METHOD_GROUP_MAP.keys())
    if methods_arg is None:
        return list(DEFAULT_METHODS)
    raw_items = methods_arg if isinstance(methods_arg, (list, tuple)) else [methods_arg]
    methods = []
    for raw in raw_items:
        for method in [x.strip() for x in str(raw).split(",") if x.strip()]:
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
        "weighted_avg_legacy": {},
        "spline_legacy": {"min_points": int(args.spline_min_points)},
        "knn_epanechnikov_legacy": {"k": int(args.legacy_knn_k)},
        "equal": {"fill_value": float(args.equal_fill_value)},
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
        "penalized_2d2_pca": {},
        "ctf": {
            "rank": int(args.competitor_ctf_rank),
            "n_initializations": int(args.competitor_ctf_initializations),
            "max_iterations_als": int(args.competitor_ctf_max_iterations),
            "max_iterations_rptm": int(args.competitor_ctf_max_iterations),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="CTF truth building + mask-aware penalized 2D2-PCA imputation simulation."
    )
    parser.add_argument("--tax_path", type=str, default="hmp2_mgx_taxonomy.tsv")
    parser.add_argument("--meta_path", type=str, default="metagenomics.xlsx")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--min_pos_cells", type=int, default=10)
    parser.add_argument("--min_participants", type=int, default=5)

    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--n_repeats", type=int, default=100)
    parser.add_argument("--fracs", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--truth_path", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="artifacts/simulation3")
    parser.add_argument("--mask_seed", type=int, default=123)
    parser.add_argument("--algo_seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Retained methods: penalized_2d2_pca, ctf, mean, nearest_positive, temporal_knn.",
    )
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
    parser.add_argument(
        "--ctf_method", type=str, default="fallback", choices=["auto", "gemelli", "fallback"],
        help="Truth generator only. The retained comparison method 'ctf' always uses Gemelli's API.",
    )
    parser.add_argument("--ctf_seed", type=int, default=0)
    parser.add_argument("--competitor-ctf-rank", type=int, default=3)
    parser.add_argument("--competitor-ctf-initializations", type=int, default=10)
    parser.add_argument("--competitor-ctf-max-iterations", type=int, default=50)
    parser.add_argument("--force_rebuild_truth", action="store_true")
    parser.add_argument("--build_truth_only", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--defer_merge", action="store_true", help="Skip shared-output merge in a single array shard.")
    parser.add_argument(
        "--require_existing_truth", "--require-existing-truth",
        dest="require_existing_truth", action="store_true",
        help="Fail if --truth_path is absent; array shards never rebuild frozen truth.",
    )
    parser.add_argument(
        "--compact_output", "--compact-output", dest="compact_output", action="store_true",
        help="Omit large intermediate taxon stats/detail CSVs while retaining summary metrics.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--temporal_knn_k", "--temporal-knn-k", dest="temporal_knn_k", type=int, default=3)
    parser.add_argument("--temporal_knn_sigma", "--temporal-knn-sigma", dest="temporal_knn_sigma", type=float, default=4.0)
    parser.add_argument("--legacy_knn_k", "--legacy-knn-k", dest="legacy_knn_k", type=int, default=3, help="K for legacy knn_epanechnikov_legacy.")
    parser.add_argument("--spline_min_points", type=int, default=4)
    parser.add_argument("--equal_fill_value", type=float, default=0.0)
    parser.add_argument("--dbn_dense_alpha", type=float, default=1.0)
    parser.add_argument("--dbn_sparse_alpha", type=float, default=0.05)
    parser.add_argument("--dbn_sparse_n_iter", type=int, default=150)
    parser.add_argument("--glv_alpha", type=float, default=0.5)
    parser.add_argument("--glv_ridge_lambda_M", "--glv-ridge-lambda-M", dest="glv_ridge_lambda_M", type=float, default=1e-3, help="Legacy glv_ridge_legacy lambda for interaction matrix M.")
    parser.add_argument("--glv_ridge_lambda_mue", "--glv-ridge-lambda-mue", dest="glv_ridge_lambda_mue", type=float, default=1e-3, help="Legacy glv_ridge_legacy lambda for self-growth mue.")
    parser.add_argument("--glv_ridge_lambda_E", "--glv-ridge-lambda-E", dest="glv_ridge_lambda_E", type=float, default=1e-3, help="Legacy glv_ridge_legacy lambda for perturbation terms; currently perturbations are not used.")
    parser.add_argument("--limits_error_threshold", "--limits-error-threshold", dest="limits_error_threshold", type=float, default=0.05, help="Legacy limits_legacy greedy improvement threshold.")
    parser.add_argument("--limits_repeats", "--limits-repeats", dest="limits_repeats", type=int, default=20, help="Legacy limits_legacy bootstrap repetitions.")
    parser.add_argument("--limits_med_or_avg", "--limits-med-or-avg", dest="limits_med_or_avg", type=int, default=0, choices=[0, 1], help="Legacy limits_legacy aggregation rule: 0 median, 1 mean.")

    args = parser.parse_args()
    verbose = not args.quiet
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.methods)
    zero_value = np.log10(args.eps)
    impute_method_params = _build_impute_method_params(args, zero_value=zero_value)
    methods, skipped = available_requested_methods(methods, impute_method_params)
    if skipped:
        details = "; ".join(f"{method}: {reason}" for method, reason in skipped.items())
        raise RuntimeError(f"Requested retained competitor is unavailable: {details}")
    if not methods:
        raise RuntimeError("No requested methods are available")

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
        truth_path = outdir / ("truth_ctf.zarr" if args.save_format == "zarr" else "truth_ctf.nc")
    else:
        truth_path = Path(args.truth_path)

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
        run_repeats_for_frac(
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
        )
        merge_repeat_outputs(outdir=outdir)
        merge_abundance_eval_outputs(outdir=outdir)
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
    )


# =====================
# Abundance-space distribution / trajectory evaluation
# =====================

def _empty_taxon_marginal_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
            "dataset_role", "taxon", "n_obs", "abund_mean", "abund_sd", "abund_mean_over_sd",
        ]
    )


def _empty_taxon_marginal_eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "taxon",
            "delta_mean_before", "delta_mean_after",
            "delta_sd_before", "delta_sd_after",
            "delta_mean_over_sd_before", "delta_mean_over_sd_after",
        ]
    )


def _empty_taxon_marginal_method_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
            "scope", "n_taxa_used",
            "marginal_ratio_wd_before", "marginal_ratio_wd_after", "marginal_ratio_wd_improve",
            "marginal_avg_euclid_before", "marginal_avg_euclid_after", "marginal_avg_euclid_improve",
        ]
    )


def _empty_taxon_bin_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
            "dataset_role", "taxon", "bin", "n_obs",
            "abund_mean_bin", "abund_sd_bin", "abund_mean_over_sd_bin",
        ]
    )


def _empty_taxon_trajectory_eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "taxon",
            "traj_mean_l2_before", "traj_mean_l2_after",
            "traj_sd_l2_before", "traj_sd_l2_after",
            "traj_mean_sd_joint_euclid_before", "traj_mean_sd_joint_euclid_after",
            "taxon_avg_bin_wd_before", "taxon_avg_bin_wd_after",
            "traj_mean_l2_improve", "traj_sd_l2_improve",
            "traj_mean_sd_joint_euclid_improve", "taxon_avg_bin_wd_improve",
        ]
    )


def _empty_taxon_trajectory_detail_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
            "taxon", "bin",
            "wd_before_bin", "wd_after_bin",
            "euclid_mean_sd_before_bin", "euclid_mean_sd_after_bin",
        ]
    )


def _empty_taxon_trajectory_method_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "scope", "n_taxa_used",
            "traj_mean_l2_before", "traj_mean_l2_after", "traj_mean_l2_improve",
            "traj_sd_l2_before", "traj_sd_l2_after", "traj_sd_l2_improve",
            "traj_mean_sd_joint_euclid_before", "traj_mean_sd_joint_euclid_after",
            "traj_mean_sd_joint_euclid_improve",
            "taxon_avg_bin_wd_before", "taxon_avg_bin_wd_after", "taxon_avg_bin_wd_improve",
        ]
    )


def _empty_bin_distribution_eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
            "bin", "n_taxa_used",
            "bin_ratio_wd_before", "bin_ratio_wd_after", "bin_ratio_wd_improve",
            "bin_avg_euclid_before", "bin_avg_euclid_after", "bin_avg_euclid_improve",
        ]
    )


def _empty_abundance_eval_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frac", "repeat", "method", "method_group", "mask_seed", "algo_seed",
            "scope", "metric", "value",
        ]
    )


def _sample_sd(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    return float(np.std(vals, ddof=1))


def _safe_ratio(mean_value: float, sd_value: float) -> float:
    if not np.isfinite(mean_value) or not np.isfinite(sd_value) or sd_value <= 0:
        return np.nan
    return float(mean_value / sd_value)


def _mean_euclid_2d(
    x_true_mean: np.ndarray,
    x_true_sd: np.ndarray,
    x_pred_mean: np.ndarray,
    x_pred_sd: np.ndarray,
) -> float:
    true_mean = np.asarray(x_true_mean, dtype=float)
    true_sd = np.asarray(x_true_sd, dtype=float)
    pred_mean = np.asarray(x_pred_mean, dtype=float)
    pred_sd = np.asarray(x_pred_sd, dtype=float)
    mask = (
        np.isfinite(true_mean) & np.isfinite(true_sd)
        & np.isfinite(pred_mean) & np.isfinite(pred_sd)
    )
    if not np.any(mask):
        return np.nan
    dist = np.sqrt((pred_mean[mask] - true_mean[mask]) ** 2 + (pred_sd[mask] - true_sd[mask]) ** 2)
    return float(np.mean(dist)) if dist.size > 0 else np.nan


def _l2_on_valid_pairs(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    true_vals = np.asarray(x_true, dtype=float)
    pred_vals = np.asarray(x_pred, dtype=float)
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if not np.any(mask):
        return np.nan
    return float(np.linalg.norm(pred_vals[mask] - true_vals[mask]))


def to_abundance_space(
    da_log: xr.DataArray,
    eps: float = 1e-3,
    name: str = "abundance_space",
) -> xr.DataArray:
    """Convert log10(percent + eps) tensor back to abundance / percent space."""
    _validate_da(da_log)
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


def compute_taxon_marginal_stats(
    da_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    dataset_role: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute per-taxon abundance marginal statistics over participant x bin cells."""
    _validate_da(da_abund)
    rows = []
    values = da_abund.values.astype(float)
    taxa = da_abund.coords["taxon"].values
    for tax_idx, taxon in enumerate(taxa):
        flat = values[:, tax_idx, :].reshape(-1)
        finite = np.isfinite(flat)
        n_obs = int(finite.sum())
        abund_mean = float(np.mean(flat[finite])) if n_obs > 0 else np.nan
        abund_sd = _sample_sd(flat[finite]) if n_obs > 0 else np.nan
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
                "abund_mean_over_sd": _safe_ratio(abund_mean, abund_sd),
            }
        )
    return pd.DataFrame(rows, columns=_empty_taxon_marginal_stats_df().columns)


def evaluate_taxon_marginal_preservation(
    df_taxon_stats: pd.DataFrame,
    frac: float,
    repeat: int,
    method: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate taxon-level marginal distribution preservation in abundance space."""
    if df_taxon_stats.empty:
        return _empty_taxon_marginal_eval_df(), _empty_taxon_marginal_method_df()

    use_cols = ["taxon", "abund_mean", "abund_sd", "abund_mean_over_sd"]
    truth = (
        df_taxon_stats.loc[df_taxon_stats["dataset_role"].eq("truth"), use_cols]
        .drop_duplicates(subset=["taxon"], keep="last")
        .rename(
            columns={
                "abund_mean": "abund_mean_truth",
                "abund_sd": "abund_sd_truth",
                "abund_mean_over_sd": "abund_mean_over_sd_truth",
            }
        )
    )
    before = (
        df_taxon_stats.loc[df_taxon_stats["dataset_role"].eq("before"), use_cols]
        .drop_duplicates(subset=["taxon"], keep="last")
        .rename(
            columns={
                "abund_mean": "abund_mean_before",
                "abund_sd": "abund_sd_before",
                "abund_mean_over_sd": "abund_mean_over_sd_before",
            }
        )
    )
    after = (
        df_taxon_stats.loc[df_taxon_stats["dataset_role"].eq("after"), use_cols]
        .drop_duplicates(subset=["taxon"], keep="last")
        .rename(
            columns={
                "abund_mean": "abund_mean_after",
                "abund_sd": "abund_sd_after",
                "abund_mean_over_sd": "abund_mean_over_sd_after",
            }
        )
    )
    merged = truth.merge(before, on="taxon", how="outer").merge(after, on="taxon", how="outer")
    if merged.empty:
        return _empty_taxon_marginal_eval_df(), _empty_taxon_marginal_method_df()

    df_taxon_eval = pd.DataFrame(
        {
            "frac": float(frac),
            "repeat": int(repeat),
            "method": method,
            "method_group": _method_group(method),
            "mask_seed": mask_seed,
            "algo_seed": algo_seed,
            "taxon": merged["taxon"],
            "delta_mean_before": merged["abund_mean_before"] - merged["abund_mean_truth"],
            "delta_mean_after": merged["abund_mean_after"] - merged["abund_mean_truth"],
            "delta_sd_before": merged["abund_sd_before"] - merged["abund_sd_truth"],
            "delta_sd_after": merged["abund_sd_after"] - merged["abund_sd_truth"],
            "delta_mean_over_sd_before": merged["abund_mean_over_sd_before"] - merged["abund_mean_over_sd_truth"],
            "delta_mean_over_sd_after": merged["abund_mean_over_sd_after"] - merged["abund_mean_over_sd_truth"],
        }
    )

    ratio_truth = merged["abund_mean_over_sd_truth"].to_numpy(dtype=float)
    ratio_before = merged["abund_mean_over_sd_before"].to_numpy(dtype=float)
    ratio_after = merged["abund_mean_over_sd_after"].to_numpy(dtype=float)
    ratio_mask_before = np.isfinite(ratio_truth) & np.isfinite(ratio_before)
    ratio_mask_after = np.isfinite(ratio_truth) & np.isfinite(ratio_after)
    marginal_ratio_wd_before = _safe_wasserstein(ratio_truth[ratio_mask_before], ratio_before[ratio_mask_before])
    marginal_ratio_wd_after = _safe_wasserstein(ratio_truth[ratio_mask_after], ratio_after[ratio_mask_after])

    true_mean = merged["abund_mean_truth"].to_numpy(dtype=float)
    true_sd = merged["abund_sd_truth"].to_numpy(dtype=float)
    before_mean = merged["abund_mean_before"].to_numpy(dtype=float)
    before_sd = merged["abund_sd_before"].to_numpy(dtype=float)
    after_mean = merged["abund_mean_after"].to_numpy(dtype=float)
    after_sd = merged["abund_sd_after"].to_numpy(dtype=float)
    marginal_avg_euclid_before = _mean_euclid_2d(true_mean, true_sd, before_mean, before_sd)
    marginal_avg_euclid_after = _mean_euclid_2d(true_mean, true_sd, after_mean, after_sd)

    taxa_used_mask = (
        (np.isfinite(ratio_truth) & (np.isfinite(ratio_before) | np.isfinite(ratio_after)))
        | (
            np.isfinite(true_mean) & np.isfinite(true_sd)
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
                "mask_seed": mask_seed,
                "algo_seed": algo_seed,
                "scope": "TaxonMarginal",
                "n_taxa_used": int(taxa_used_mask.sum()),
                "marginal_ratio_wd_before": marginal_ratio_wd_before,
                "marginal_ratio_wd_after": marginal_ratio_wd_after,
                "marginal_ratio_wd_improve": marginal_ratio_wd_before - marginal_ratio_wd_after,
                "marginal_avg_euclid_before": marginal_avg_euclid_before,
                "marginal_avg_euclid_after": marginal_avg_euclid_after,
                "marginal_avg_euclid_improve": marginal_avg_euclid_before - marginal_avg_euclid_after,
            }
        ],
        columns=_empty_taxon_marginal_method_df().columns,
    )
    return df_taxon_eval, df_method


def compute_taxon_bin_stats(
    da_abund: xr.DataArray,
    frac: float,
    repeat: int,
    method: str,
    dataset_role: str,
    mask_seed: Optional[int] = None,
    algo_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute per-taxon, per-bin abundance statistics across participants."""
    _validate_da(da_abund)
    rows = []
    values = da_abund.values.astype(float)
    taxa = da_abund.coords["taxon"].values
    bins = da_abund.coords["bin"].values
    for tax_idx, taxon in enumerate(taxa):
        for bin_idx, bin_label in enumerate(bins):
            vec = values[:, tax_idx, bin_idx]
            finite = np.isfinite(vec)
            n_obs = int(finite.sum())
            abund_mean = float(np.mean(vec[finite])) if n_obs > 0 else np.nan
            abund_sd = _sample_sd(vec[finite]) if n_obs > 0 else np.nan
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
                    "abund_mean_bin": abund_mean,
                    "abund_sd_bin": abund_sd,
                    "abund_mean_over_sd_bin": _safe_ratio(abund_mean, abund_sd),
                }
        )
    return pd.DataFrame(rows, columns=_empty_taxon_bin_stats_df().columns)


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
    """Evaluate per-taxon trajectories and per-taxon-bin distribution preservation."""
    if df_taxon_bin_stats.empty:
        return (
            _empty_taxon_trajectory_eval_df(),
            _empty_taxon_trajectory_detail_df(),
            _empty_taxon_trajectory_method_df(),
        )

    _validate_da(da_truth_abund)
    _validate_da(da_before_abund)
    _validate_da(da_after_abund)
    ordered_bins = list(da_truth_abund.coords["bin"].values)
    ordered_taxa = list(da_truth_abund.coords["taxon"].values)
    taxon_to_idx = {taxon: idx for idx, taxon in enumerate(ordered_taxa)}

    use_cols = ["taxon", "bin", "abund_mean_bin", "abund_sd_bin", "abund_mean_over_sd_bin"]
    truth = (
        df_taxon_bin_stats.loc[df_taxon_bin_stats["dataset_role"].eq("truth"), use_cols]
        .drop_duplicates(subset=["taxon", "bin"], keep="last")
        .rename(
            columns={
                "abund_mean_bin": "abund_mean_bin_truth",
                "abund_sd_bin": "abund_sd_bin_truth",
                "abund_mean_over_sd_bin": "abund_mean_over_sd_bin_truth",
            }
        )
    )
    before = (
        df_taxon_bin_stats.loc[df_taxon_bin_stats["dataset_role"].eq("before"), use_cols]
        .drop_duplicates(subset=["taxon", "bin"], keep="last")
        .rename(
            columns={
                "abund_mean_bin": "abund_mean_bin_before",
                "abund_sd_bin": "abund_sd_bin_before",
                "abund_mean_over_sd_bin": "abund_mean_over_sd_bin_before",
            }
        )
    )
    after = (
        df_taxon_bin_stats.loc[df_taxon_bin_stats["dataset_role"].eq("after"), use_cols]
        .drop_duplicates(subset=["taxon", "bin"], keep="last")
        .rename(
            columns={
                "abund_mean_bin": "abund_mean_bin_after",
                "abund_sd_bin": "abund_sd_bin_after",
                "abund_mean_over_sd_bin": "abund_mean_over_sd_bin_after",
            }
        )
    )
    merged = truth.merge(before, on=["taxon", "bin"], how="outer").merge(after, on=["taxon", "bin"], how="outer")
    if merged.empty:
        return (
            _empty_taxon_trajectory_eval_df(),
            _empty_taxon_trajectory_detail_df(),
            _empty_taxon_trajectory_method_df(),
        )

    merged["bin"] = pd.Categorical(merged["bin"], categories=ordered_bins, ordered=True)
    merged = merged.sort_values(["taxon", "bin"]).reset_index(drop=True)
    detail_rows = []
    eval_rows = []
    truth_vals = da_truth_abund.values.astype(float)
    before_vals = da_before_abund.values.astype(float)
    after_vals = da_after_abund.values.astype(float)

    for taxon in ordered_taxa:
        g = merged.loc[merged["taxon"].eq(taxon)].copy()
        if g.empty:
            continue
        g = g.set_index("bin").reindex(ordered_bins).reset_index()
        mean_truth = g["abund_mean_bin_truth"].to_numpy(dtype=float)
        mean_before = g["abund_mean_bin_before"].to_numpy(dtype=float)
        mean_after = g["abund_mean_bin_after"].to_numpy(dtype=float)
        sd_truth = g["abund_sd_bin_truth"].to_numpy(dtype=float)
        sd_before = g["abund_sd_bin_before"].to_numpy(dtype=float)
        sd_after = g["abund_sd_bin_after"].to_numpy(dtype=float)

        traj_mean_l2_before = _l2_on_valid_pairs(mean_truth, mean_before)
        traj_mean_l2_after = _l2_on_valid_pairs(mean_truth, mean_after)
        traj_sd_l2_before = _l2_on_valid_pairs(sd_truth, sd_before)
        traj_sd_l2_after = _l2_on_valid_pairs(sd_truth, sd_after)
        traj_joint_before = _mean_euclid_2d(mean_truth, sd_truth, mean_before, sd_before)
        traj_joint_after = _mean_euclid_2d(mean_truth, sd_truth, mean_after, sd_after)

        tax_idx = taxon_to_idx[taxon]
        bin_wd_before_vals = []
        bin_wd_after_vals = []
        for bin_idx, bin_label in enumerate(ordered_bins):
            v_truth = truth_vals[:, tax_idx, bin_idx]
            v_before = before_vals[:, tax_idx, bin_idx]
            v_after = after_vals[:, tax_idx, bin_idx]
            v_truth = v_truth[np.isfinite(v_truth)]
            v_before = v_before[np.isfinite(v_before)]
            v_after = v_after[np.isfinite(v_after)]
            wd_before_bin = _safe_wasserstein(v_truth, v_before)
            wd_after_bin = _safe_wasserstein(v_truth, v_after)
            euclid_before_bin = _mean_euclid_2d(
                np.array([mean_truth[bin_idx]]),
                np.array([sd_truth[bin_idx]]),
                np.array([mean_before[bin_idx]]),
                np.array([sd_before[bin_idx]]),
            )
            euclid_after_bin = _mean_euclid_2d(
                np.array([mean_truth[bin_idx]]),
                np.array([sd_truth[bin_idx]]),
                np.array([mean_after[bin_idx]]),
                np.array([sd_after[bin_idx]]),
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
            if np.isfinite(wd_before_bin):
                bin_wd_before_vals.append(wd_before_bin)
            if np.isfinite(wd_after_bin):
                bin_wd_after_vals.append(wd_after_bin)

        taxon_avg_bin_wd_before = float(np.mean(bin_wd_before_vals)) if bin_wd_before_vals else np.nan
        taxon_avg_bin_wd_after = float(np.mean(bin_wd_after_vals)) if bin_wd_after_vals else np.nan
        eval_rows.append(
            {
                "frac": float(frac),
                "repeat": int(repeat),
                "method": method,
                "method_group": _method_group(method),
                "mask_seed": mask_seed,
                "algo_seed": algo_seed,
                "taxon": taxon,
                "traj_mean_l2_before": traj_mean_l2_before,
                "traj_mean_l2_after": traj_mean_l2_after,
                "traj_sd_l2_before": traj_sd_l2_before,
                "traj_sd_l2_after": traj_sd_l2_after,
                "traj_mean_sd_joint_euclid_before": traj_joint_before,
                "traj_mean_sd_joint_euclid_after": traj_joint_after,
                "taxon_avg_bin_wd_before": taxon_avg_bin_wd_before,
                "taxon_avg_bin_wd_after": taxon_avg_bin_wd_after,
                "traj_mean_l2_improve": traj_mean_l2_before - traj_mean_l2_after,
                "traj_sd_l2_improve": traj_sd_l2_before - traj_sd_l2_after,
                "traj_mean_sd_joint_euclid_improve": traj_joint_before - traj_joint_after,
                "taxon_avg_bin_wd_improve": taxon_avg_bin_wd_before - taxon_avg_bin_wd_after,
            }
        )

    df_eval = pd.DataFrame(eval_rows, columns=_empty_taxon_trajectory_eval_df().columns)
    df_detail = pd.DataFrame(detail_rows, columns=_empty_taxon_trajectory_detail_df().columns)
    if df_eval.empty:
        return df_eval, df_detail, _empty_taxon_trajectory_method_df()

    metric_cols = [
        "traj_mean_l2_before", "traj_mean_l2_after", "traj_mean_l2_improve",
        "traj_sd_l2_before", "traj_sd_l2_after", "traj_sd_l2_improve",
        "traj_mean_sd_joint_euclid_before", "traj_mean_sd_joint_euclid_after",
        "traj_mean_sd_joint_euclid_improve",
        "taxon_avg_bin_wd_before", "taxon_avg_bin_wd_after", "taxon_avg_bin_wd_improve",
    ]
    valid_taxa = np.isfinite(df_eval[metric_cols].to_numpy(dtype=float)).any(axis=1)
    row = {
        "frac": float(frac),
        "repeat": int(repeat),
        "method": method,
        "method_group": _method_group(method),
        "mask_seed": mask_seed,
        "algo_seed": algo_seed,
        "scope": "TaxonTrajectory",
        "n_taxa_used": int(valid_taxa.sum()),
    }
    for col in metric_cols:
        vals = df_eval[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        row[col] = float(np.mean(vals)) if vals.size > 0 else np.nan
    df_method = pd.DataFrame([row], columns=_empty_taxon_trajectory_method_df().columns)
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
    """Evaluate per-bin abundance structure preservation across taxa."""
    _validate_da(da_truth_abund)
    _validate_da(da_before_abund)
    _validate_da(da_after_abund)
    rows = []
    bins = da_truth_abund.coords["bin"].values
    taxa = da_truth_abund.coords["taxon"].values
    truth_vals = da_truth_abund.values.astype(float)
    before_vals = da_before_abund.values.astype(float)
    after_vals = da_after_abund.values.astype(float)

    for bin_idx, bin_label in enumerate(bins):
        truth_mean, truth_sd, truth_ratio = [], [], []
        before_mean, before_sd, before_ratio = [], [], []
        after_mean, after_sd, after_ratio = [], [], []
        for tax_idx, _taxon in enumerate(taxa):
            vec_truth = truth_vals[:, tax_idx, bin_idx]
            vec_before = before_vals[:, tax_idx, bin_idx]
            vec_after = after_vals[:, tax_idx, bin_idx]
            vec_truth = vec_truth[np.isfinite(vec_truth)]
            vec_before = vec_before[np.isfinite(vec_before)]
            vec_after = vec_after[np.isfinite(vec_after)]

            mt = float(np.mean(vec_truth)) if vec_truth.size > 0 else np.nan
            st = _sample_sd(vec_truth)
            mb = float(np.mean(vec_before)) if vec_before.size > 0 else np.nan
            sb = _sample_sd(vec_before)
            ma = float(np.mean(vec_after)) if vec_after.size > 0 else np.nan
            sa = _sample_sd(vec_after)

            truth_mean.append(mt)
            truth_sd.append(st)
            truth_ratio.append(_safe_ratio(mt, st))
            before_mean.append(mb)
            before_sd.append(sb)
            before_ratio.append(_safe_ratio(mb, sb))
            after_mean.append(ma)
            after_sd.append(sa)
            after_ratio.append(_safe_ratio(ma, sa))

        truth_mean = np.asarray(truth_mean, dtype=float)
        truth_sd = np.asarray(truth_sd, dtype=float)
        truth_ratio = np.asarray(truth_ratio, dtype=float)
        before_mean = np.asarray(before_mean, dtype=float)
        before_sd = np.asarray(before_sd, dtype=float)
        before_ratio = np.asarray(before_ratio, dtype=float)
        after_mean = np.asarray(after_mean, dtype=float)
        after_sd = np.asarray(after_sd, dtype=float)
        after_ratio = np.asarray(after_ratio, dtype=float)

        ratio_mask_before = np.isfinite(truth_ratio) & np.isfinite(before_ratio)
        ratio_mask_after = np.isfinite(truth_ratio) & np.isfinite(after_ratio)
        bin_ratio_wd_before = _safe_wasserstein(truth_ratio[ratio_mask_before], before_ratio[ratio_mask_before])
        bin_ratio_wd_after = _safe_wasserstein(truth_ratio[ratio_mask_after], after_ratio[ratio_mask_after])
        bin_avg_euclid_before = _mean_euclid_2d(truth_mean, truth_sd, before_mean, before_sd)
        bin_avg_euclid_after = _mean_euclid_2d(truth_mean, truth_sd, after_mean, after_sd)
        taxa_used_mask = (
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
                "mask_seed": mask_seed,
                "algo_seed": algo_seed,
                "bin": bin_label,
                "n_taxa_used": int(taxa_used_mask.sum()),
                "bin_ratio_wd_before": bin_ratio_wd_before,
                "bin_ratio_wd_after": bin_ratio_wd_after,
                "bin_ratio_wd_improve": bin_ratio_wd_before - bin_ratio_wd_after,
                "bin_avg_euclid_before": bin_avg_euclid_before,
                "bin_avg_euclid_after": bin_avg_euclid_after,
                "bin_avg_euclid_improve": bin_avg_euclid_before - bin_avg_euclid_after,
            }
        )
    return pd.DataFrame(rows, columns=_empty_bin_distribution_eval_df().columns)


def build_abundance_eval_repeat_summary(
    df_taxon_marginal_method: pd.DataFrame,
    df_taxon_traj_method: pd.DataFrame,
    df_bin_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Build long-form per-repeat abundance evaluation summary across scopes."""
    parts = []
    id_cols = ["frac", "repeat", "method", "method_group", "mask_seed", "algo_seed", "scope"]

    def _melt_summary(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return _empty_abundance_eval_summary_df()
        value_cols = [c for c in df_in.columns if c not in id_cols]
        long_df = df_in.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")
        return long_df[_empty_abundance_eval_summary_df().columns]

    if not df_taxon_marginal_method.empty:
        parts.append(_melt_summary(df_taxon_marginal_method))
    if not df_taxon_traj_method.empty:
        parts.append(_melt_summary(df_taxon_traj_method))
    if not df_bin_distribution.empty:
        metric_cols = [
            "bin_ratio_wd_before", "bin_ratio_wd_after", "bin_ratio_wd_improve",
            "bin_avg_euclid_before", "bin_avg_euclid_after", "bin_avg_euclid_improve",
        ]
        rows = []
        group_cols = ["frac", "repeat", "method", "method_group", "mask_seed", "algo_seed"]
        for group_vals, g in df_bin_distribution.groupby(group_cols):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            kv = dict(zip(group_cols, group_vals))
            row = {**kv, "scope": "BinDistribution", "n_bins_used": int(g["bin"].nunique())}
            valid_taxa = g["n_taxa_used"].to_numpy(dtype=float)
            row["bin_mean_n_taxa_used"] = (
                float(np.mean(valid_taxa[np.isfinite(valid_taxa)]))
                if np.isfinite(valid_taxa).any() else np.nan
            )
            for metric in metric_cols:
                vals = g[metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                row[metric] = float(np.mean(vals)) if vals.size > 0 else np.nan
            rows.append(row)
        parts.append(_melt_summary(pd.DataFrame(rows)))

    if not parts:
        return _empty_abundance_eval_summary_df()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["frac", "repeat", "method", "scope", "metric"]).reset_index(drop=True)


def summarize_abundance_eval_across_repeats(
    df_repeat_abund: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize abundance evaluation metrics across repeats with real-valued summary stats."""
    cols = ["frac", "scope", "method", "metric", "n", "mean", "sd", "sem", "ci95_low", "ci95_high"]
    if df_repeat_abund.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (frac, scope, method, metric), g in df_repeat_abund.groupby(["frac", "scope", "method", "metric"]):
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
    return pd.DataFrame(rows, columns=cols).sort_values(["frac", "scope", "method", "metric"]).reset_index(drop=True)


def merge_abundance_eval_outputs(outdir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge abundance-space repeat outputs without affecting legacy reconstruction files."""
    outdir = Path(outdir)
    taxon_marginal_files = sorted(outdir.glob("taxon_marginal_eval_frac_*_repeat_*.csv"))
    taxon_trajectory_files = sorted(outdir.glob("taxon_trajectory_eval_frac_*_repeat_*.csv"))
    bin_distribution_files = sorted(outdir.glob("bin_distribution_eval_frac_*_repeat_*.csv"))
    abundance_summary_files = sorted(outdir.glob("abundance_eval_summary_frac_*_repeat_*.csv"))

    df_taxon_marginal = (
        pd.concat([pd.read_csv(p) for p in taxon_marginal_files], ignore_index=True)
        if taxon_marginal_files else _empty_taxon_marginal_eval_df()
    )
    df_taxon_trajectory = (
        pd.concat([pd.read_csv(p) for p in taxon_trajectory_files], ignore_index=True)
        if taxon_trajectory_files else _empty_taxon_trajectory_eval_df()
    )
    df_bin_distribution = (
        pd.concat([pd.read_csv(p) for p in bin_distribution_files], ignore_index=True)
        if bin_distribution_files else _empty_bin_distribution_eval_df()
    )
    df_abund_repeat = (
        pd.concat([pd.read_csv(p) for p in abundance_summary_files], ignore_index=True)
        if abundance_summary_files else _empty_abundance_eval_summary_df()
    )

    if not df_taxon_marginal.empty:
        df_taxon_marginal = df_taxon_marginal.drop_duplicates(subset=["frac", "repeat", "method", "taxon"], keep="last")
    if not df_taxon_trajectory.empty:
        df_taxon_trajectory = df_taxon_trajectory.drop_duplicates(subset=["frac", "repeat", "method", "taxon"], keep="last")
    if not df_bin_distribution.empty:
        df_bin_distribution = df_bin_distribution.drop_duplicates(subset=["frac", "repeat", "method", "bin"], keep="last")
    if not df_abund_repeat.empty:
        df_abund_repeat = df_abund_repeat.drop_duplicates(subset=["frac", "repeat", "method", "scope", "metric"], keep="last")

    df_abund_errorbar = summarize_abundance_eval_across_repeats(df_abund_repeat)

    if not df_taxon_marginal.empty:
        df_taxon_marginal.to_csv(outdir / "taxon_marginal_eval_all.csv", index=False)
    if not df_taxon_trajectory.empty:
        df_taxon_trajectory.to_csv(outdir / "taxon_trajectory_eval_all.csv", index=False)
    if not df_bin_distribution.empty:
        df_bin_distribution.to_csv(outdir / "bin_distribution_eval_all.csv", index=False)
    if not df_abund_repeat.empty:
        df_abund_repeat.to_csv(outdir / "abundance_repeat_level_metrics.csv", index=False)
    if not df_abund_errorbar.empty:
        df_abund_errorbar.to_csv(outdir / "abundance_errorbar_summary.csv", index=False)

    return df_taxon_marginal, df_taxon_trajectory, df_bin_distribution, df_abund_repeat


if __name__ == "__main__":
    main()
