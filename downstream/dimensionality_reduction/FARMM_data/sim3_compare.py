from __future__ import annotations

import argparse
import json
import traceback
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import svd
from scipy.interpolate import CubicSpline
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance


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
    s = str(bin_label).strip()
    if s.upper() == "PS":
        return 16.0
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            return 0.5 * (float(a) + float(b))
        except ValueError as exc:
            raise ValueError(f"Invalid time-bin label {bin_label!r}; expected numeric, 'PS', or 'start-end'.") from exc
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"Invalid time-bin label {bin_label!r}; expected numeric, 'PS', or 'start-end'.") from exc


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
    groups = {
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
    return groups.get(str(method), "other")


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

    abar = np.where(obs, a, 0.0).sum(axis=0) / np.maximum(obs.sum(axis=0), 1.0)
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

    meta = {"bin": time, "t": tvals, "W": w_time, "Abar": abar}
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
):
    truth_path = Path(truth_path)
    effect_artifact_dir = truth_path.parent if effect_artifact_dir is None else Path(effect_artifact_dir)
    effect_artifacts_ready = (
        (effect_artifact_dir / "effect_pseudo_group.csv").exists()
        and (effect_artifact_dir / "effect_taxon_effect.csv").exists()
        and (effect_artifact_dir / "effect_meta.json").exists()
        and _effect_truth_path(effect_artifact_dir, save_format).exists()
    )
    if truth_path.exists() and (not force_rebuild) and ((not enable_effect) or effect_artifacts_ready):
        da_gt = _load_dataarray(truth_path, var_name="y_true")
        da_gt.name = "y_true"
        print(f"[truth] loaded existing truth from {truth_path}")
        return da_gt
    if truth_path.exists() and (not force_rebuild) and enable_effect and (not effect_artifacts_ready):
        print("[truth] cached effect truth exists but effect artifacts are incomplete; rebuilding.")

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
) -> np.ndarray:
    out = y.copy()
    obs_idx = np.flatnonzero(~mask)
    obs_vals = y[obs_idx]
    mean_fallback = _trajectory_mean(obs_vals, default_value=zero_value)
    fill_value = params.get("fill_value", zero_value)
    for idx in np.flatnonzero(mask):
        t = float(times[idx])
        if method == "mean":
            out[idx] = _trajectory_fill_constant(obs_vals, "mean", default_value=zero_value)
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
                fallback_value=mean_fallback,
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
        arr = da_input.transpose("participant", "taxon", "bin").values.astype(float)
        mask = replace_mask.transpose("participant", "taxon", "bin").values.astype(bool)
        time_vals = np.asarray([_bin_to_time_value(b) for b in da_input.coords["bin"].values], dtype=float)
        out = arr.copy()
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
                    )
        da_imp = xr.DataArray(out, coords=da_input.coords, dims=da_input.dims, name="log_abundance_imputed")
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
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    da_imp = da_imp.transpose(*da_input.dims)
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
):
    bins_use = np.array(da_gt.coords["bin"].values)
    zero_value = np.log10(eps)
    methods = list(methods) if methods is not None else ["penalized_2d2_pca"]
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

        init_method = "mean" if method in {"dbn_dense", "dbn_sparse", "glv"} else ""
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

    df_taxon_marginal_stats.to_csv(
        outdir / f"taxon_marginal_stats_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    df_taxon_marginal_eval.to_csv(
        outdir / f"taxon_marginal_eval_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    df_taxon_bin_stats.to_csv(
        outdir / f"taxon_bin_stats_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    df_taxon_trajectory_eval.to_csv(
        outdir / f"taxon_trajectory_eval_frac_{frac_str}_repeat_{repeat_str}.csv",
        index=False,
    )
    df_taxon_trajectory_detail.to_csv(
        outdir / f"taxon_trajectory_detail_frac_{frac_str}_repeat_{repeat_str}.csv",
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


def _parse_methods(methods_arg: Optional[str]) -> List[str]:
    allowed = {
        "penalized_2d2_pca", "mean", "median", "last", "weighted_time_average",
        "temporal_knn", "spline", "dbn_dense", "dbn_sparse", "glv", "equal",
    }
    if methods_arg is None or str(methods_arg).strip() == "":
        return ["penalized_2d2_pca"]
    methods = []
    for method in [x.strip() for x in str(methods_arg).split(",") if x.strip()]:
        if method not in allowed:
            raise ValueError(f"Unsupported method in --methods: {method}")
        if method not in methods:
            methods.append(method)
    return methods if methods else ["penalized_2d2_pca"]


def _build_impute_method_params(args, zero_value: float) -> Dict[str, Dict]:
    return {
        "mean": {},
        "median": {},
        "last": {},
        "weighted_time_average": {},
        "temporal_knn": {"k": int(args.temporal_knn_k), "kernel_sigma": float(args.temporal_knn_sigma)},
        "spline": {"min_points": int(args.spline_min_points)},
        "equal": {"fill_value": float(args.equal_fill_value)},
        "dbn_dense": {"alpha": float(args.dbn_dense_alpha)},
        "dbn_sparse": {"alpha": float(args.dbn_sparse_alpha), "n_iter": int(args.dbn_sparse_n_iter)},
        "glv": {"alpha": float(args.glv_alpha), "init_method": "mean"},
        "penalized_2d2_pca": {},
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
    parser.add_argument("--methods", type=str, default="penalized_2d2_pca")
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
    parser.add_argument("--enable_effect", action="store_true")
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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--temporal_knn_k", type=int, default=3)
    parser.add_argument("--temporal_knn_sigma", type=float, default=4.0)
    parser.add_argument("--spline_min_points", type=int, default=4)
    parser.add_argument("--equal_fill_value", type=float, default=0.0)
    parser.add_argument("--dbn_dense_alpha", type=float, default=1.0)
    parser.add_argument("--dbn_sparse_alpha", type=float, default=0.05)
    parser.add_argument("--dbn_sparse_n_iter", type=int, default=150)
    parser.add_argument("--glv_alpha", type=float, default=0.5)

    args = parser.parse_args()
    verbose = not args.quiet
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.methods)
    if args.eps <= 0:
        raise ValueError(f"--eps must be > 0, got {args.eps}")
    if not (0.0 <= float(args.diff_frac) <= 1.0):
        raise ValueError(f"--diff_frac must be in [0, 1], got {args.diff_frac}")
    zero_value = np.log10(args.eps)
    impute_method_params = _build_impute_method_params(args, zero_value=zero_value)

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
        effect_artifact_dir=outdir,
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
        )
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

    marginal_files = sorted(outdir.glob("taxon_marginal_eval_frac_*_repeat_*.csv"))
    traj_files = sorted(outdir.glob("taxon_trajectory_eval_frac_*_repeat_*.csv"))
    bin_files = sorted(outdir.glob("bin_distribution_eval_frac_*_repeat_*.csv"))
    summary_files = sorted(outdir.glob("abundance_eval_summary_frac_*_repeat_*.csv"))

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

    if not df_marginal.empty:
        df_marginal = df_marginal.drop_duplicates(subset=["frac", "repeat", "method", "taxon"], keep="last")
        df_marginal.to_csv(outdir / "taxon_marginal_eval_all.csv", index=False)
    if not df_traj.empty:
        df_traj = df_traj.drop_duplicates(subset=["frac", "repeat", "method", "taxon"], keep="last")
        df_traj.to_csv(outdir / "taxon_trajectory_eval_all.csv", index=False)
    if not df_bin.empty:
        df_bin = df_bin.drop_duplicates(subset=["frac", "repeat", "method", "bin"], keep="last")
        df_bin.to_csv(outdir / "bin_distribution_eval_all.csv", index=False)
    if not df_repeat.empty:
        df_repeat = df_repeat.drop_duplicates(subset=["frac", "repeat", "method", "scope", "metric"], keep="last")
        df_repeat.to_csv(outdir / "abundance_repeat_level_metrics.csv", index=False)

    df_errorbar = summarize_abundance_eval_across_repeats(df_repeat)
    if not df_errorbar.empty:
        df_errorbar.to_csv(outdir / "abundance_errorbar_summary.csv", index=False)

    return df_marginal, df_traj, df_bin, df_repeat


if __name__ == "__main__":
    main()
