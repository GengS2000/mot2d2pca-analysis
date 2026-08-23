from __future__ import annotations
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# -----------------------
# 1) Load the data
# -----------------------
tax_path = "hmp2_mgx_taxonomy.tsv"
meta_path = "metagenomics.xlsx"

tax = pd.read_csv(tax_path, sep="\t")
tax = tax.rename(columns={tax.columns[0]: "taxon"}).set_index("taxon")

meta = pd.read_excel(meta_path)

# -----------------------
# 2) Clean and align External ID
# -----------------------
tax.columns = tax.columns.str.replace(r"_profile$", "", regex=True)

meta_mgx = (
    meta.loc[meta["data_type"].eq("metagenomics"), ["External ID", "Participant ID", "week_num"]]
        .dropna(subset=["External ID", "Participant ID", "week_num"])
        .rename(columns={"External ID": "external_id", "Participant ID": "participant", "week_num": "week"})
)

meta_mgx["week"] = meta_mgx["week"].astype(int)
meta_mgx["participant"] = meta_mgx["participant"].astype(str)
# -----------------------
# 2.5) Taxon filter (for the downstream mechanistic analysis)
# -----------------------
idx = tax.index.astype(str)

has_species = idx.str.contains("s__", na=False)
not_euk = ~idx.str.contains("k__Eukaryota", na=False)

# Drop entries that are unclassified and carry no CAG number, which usually
# indicates poor alignment quality.
has_unclassified = idx.str.contains("unclassified", na=False)
has_cag = idx.str.contains(r"CAG[_-]?\d+", regex=True, na=False)  # matches CAG_110 / CAG-110
bad_unclassified = has_unclassified & (~has_cag)

mask = has_species & not_euk & (~bad_unclassified)

tax_before = tax.shape[0]
tax = tax.loc[mask].copy()
tax_after = tax.shape[0]
print(f"[taxon filter] taxa: {tax_before} -> {tax_after}")

# Optional hard check that every retained feature is s__
assert tax.index.to_series().str.contains("s__", na=False).all()

# -----------------------
# 3) Intersect the samples
# -----------------------
common = sorted(set(tax.columns).intersection(set(meta_mgx["external_id"])))
tax = tax[common]
meta_mgx = meta_mgx.drop_duplicates("external_id").set_index("external_id").loc[common]

# -----------------------
# 4) Sample x Taxon table joined to the metadata
# -----------------------
X = tax.T  # index=external_id, columns=taxon
X.index.name = "external_id"
X = X.join(meta_mgx, how="left")

assert X["participant"].isna().sum() == 0, "External IDs that cannot be mapped to participant/week are present"

taxa = tax.index.astype(str).tolist()

# -----------------------
# 5) Define the time windows
#    baseline: week = 0
#    thereafter one bin per 4 weeks: 1-4, 5-8, ..., 33-36
# -----------------------
def week_to_bin(week: int) -> str | None:
    if week == 0:
        return "0"
    if 1 <= week <= 36:
        start = ((week - 1) // 4) * 4 + 1
        end = start + 3
        return f"{start}-{end}"
    return None  # outside the range: discard

bin_order = ["0"] + [f"{s}-{s+3}" for s in range(1, 37, 4)]

# -----------------------
# 6) Align to the time windows and aggregate by mean
#    - repeated measurements in one participant-bin: take the taxon mean
#    - missing within a bin: keep NaN
# -----------------------
X_taxa = X[taxa].copy()
X_taxa["participant"] = X["participant"].astype(str)
X_taxa["week"] = X["week"].astype(int)
X_taxa["bin"] = X_taxa["week"].map(week_to_bin)

# Keep only samples that fall inside a defined window
X_taxa = X_taxa.dropna(subset=["bin"])

X_taxa = X_taxa.set_index(["participant", "bin"])[taxa]
X_agg = X_taxa.groupby(level=["participant", "bin"]).mean()

# -----------------------
# 7) Build the complete participant x bin grid, padding with NaN
# -----------------------
participants = sorted(X_agg.index.get_level_values("participant").unique().tolist())
bins = bin_order

full_index = pd.MultiIndex.from_product([participants, bins], names=["participant", "bin"])
X_full = X_agg.reindex(full_index)  # unmeasured cells become NaN

#### Taxon prevalence filter
# -----------------------
# 7.5) Prevalence filter, applied on the participant x bin grid
#-----------------------
# Occurrence is defined by positivity rather than mean abundance, which suits
# the observed-only CTF setting better.
pos_mask_full = X_full.gt(0).fillna(False)   # index=(participant, bin), columns=taxa

# Condition 1: at least min_pos_cells positive cells on the full participant x bin grid.
# Condition 2: present in at least min_participants participants.
min_pos_cells = 10      ### lenient: min_pos_cells = 5, min_participants = 3
min_participants = 5    ## balanced: min_pos_cells = 10, min_participants = 5; strict: min_pos_cells = 20, min_participants = 10
pos_cells_per_taxon = pos_mask_full.sum(axis=0)

pos_by_participant = pos_mask_full.groupby(level="participant").any()
pos_participants_per_taxon = pos_by_participant.sum(axis=0)

keep_taxa_mask = (
    (pos_cells_per_taxon >= min_pos_cells) &
    (pos_participants_per_taxon >= min_participants)
)

tax_before_prev = X_full.shape[1]
X_full = X_full.loc[:, keep_taxa_mask]
tax_after_prev = X_full.shape[1]

taxa = X_full.columns.astype(str).tolist()

print(
    f"[prevalence filter] taxa: {tax_before_prev} -> {tax_after_prev} "
    f"(min_pos_cells={min_pos_cells}, min_participants={min_participants})"
)

# Optional diagnostics
print("[diag] pos_cells_per_taxon summary:")
print(pos_cells_per_taxon[keep_taxa_mask].describe())

print("[diag] pos_participants_per_taxon summary:")
print(pos_participants_per_taxon[keep_taxa_mask].describe())

# -----------------------
# 6.5) Log transform: Y = log(X + eps)
#     - NaN is preserved as 0
#     - eps may be 1e-2 (=0.01) or 1e-3 (=0.001)
# -----------------------
eps = 1e-3

# 1) Keep genuine missingness as NaN; do not fillna(0) first
X_full_log = np.log10(X_full + eps)

# 2) The three masks
mask_missing_df = X_full.isna()                          # genuinely missing
mask_zero_df    = X_full.eq(0.0).fillna(False)          # observed zeros
observed_positive_df = X_full.gt(0.0).fillna(False)     # cells that actually enter the CTF fit
mask_to_impute_df = (~observed_positive_df)             # cells to impute = missing or zero
# 8) reshape -> 3D tensor: participant × taxon × bin
# -----------------------
arr = X_full_log.to_numpy().reshape(len(participants), len(bins), len(taxa))
tensor = np.transpose(arr, (0, 2, 1))

obs_arr = observed_positive_df.to_numpy().reshape(len(participants), len(bins), len(taxa))
obs_tensor = np.transpose(obs_arr, (0, 2, 1))

imp_arr = mask_to_impute_df.to_numpy().reshape(len(participants), len(bins), len(taxa))
imp_tensor = np.transpose(imp_arr, (0, 2, 1))

da = xr.DataArray(
    tensor,
    dims=("participant", "taxon", "bin"),
    coords={"participant": participants, "taxon": taxa, "bin": bins},
    name="abundance_binned"
)

da_observed_mask = xr.DataArray(
    obs_tensor,
    dims=("participant", "taxon", "bin"),
    coords={"participant": participants, "taxon": taxa, "bin": bins},
    name="observed_positive_mask"
)

da_impute_mask = xr.DataArray(
    imp_tensor,
    dims=("participant", "taxon", "bin"),
    coords={"participant": participants, "taxon": taxa, "bin": bins},
    name="mask_to_impute"
)

# ctf_reconstruct.py
# -*- coding: utf-8 -*-


import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import xarray as xr
from scipy.linalg import svd


def _validate_da(da: xr.DataArray) -> None:
    """Basic sanity checks for input tensor in log(percent + eps) space."""
    assert da.dims == ("participant", "taxon", "bin"), (
        f"Expected dims ('participant','taxon','bin'), got {da.dims}"
    )
    shape = da.shape
    for d, n in zip(da.dims, shape):
        if d not in da.coords:
            raise ValueError(f"Missing coord for dim '{d}'")
        if da.coords[d].size != n:
            raise ValueError(
                f"Coord length mismatch on dim '{d}': coord={da.coords[d].size}, shape={n}"
            )


def _unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding."""
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def _khatrirao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Column-wise Kronecker product.
    A: (m, r), B: (n, r) -> (m*n, r)
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have same number of columns")
    r = A.shape[1]
    return np.vstack([np.kron(A[:, i], B[:, i]) for i in range(r)]).T


def _init_factors_svd(
    X_filled: np.ndarray, rank: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD-based init from zero-filled tensor (init only, not fitting objective)."""
    rng = np.random.default_rng(seed)
    I, J, K = X_filled.shape
    A = np.zeros((I, rank), dtype=float)
    B = np.zeros((J, rank), dtype=float)
    C = np.zeros((K, rank), dtype=float)

    for mode, size, out in [(0, I, A), (1, J, B), (2, K, C)]:
        M = _unfold(X_filled, mode)
        try:
            U, _, _ = svd(M, full_matrices=False)
            r_take = min(rank, U.shape[1])
            out[:, :r_take] = U[:, :r_take]
            if r_take < rank:
                out[:, r_take:] = rng.normal(0, 1e-2, size=(size, rank - r_take))
        except Exception:
            out[:] = rng.normal(0, 1e-2, size=(size, rank))
    return A, B, C


def _init_factors_random(shape: Tuple[int, int, int], rank: int, seed: int):
    rng = np.random.default_rng(seed)
    I, J, K = shape
    A = rng.normal(0.0, 0.1, size=(I, rank))
    B = rng.normal(0.0, 0.1, size=(J, rank))
    C = rng.normal(0.0, 0.1, size=(K, rank))
    return A, B, C


def _normalize_factors(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, sigma: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Absorb scale into sigma to improve stability."""
    rank = A.shape[1]
    for r in range(rank):
        na = np.linalg.norm(A[:, r]) + 1e-12
        nb = np.linalg.norm(B[:, r]) + 1e-12
        nc = np.linalg.norm(C[:, r]) + 1e-12
        A[:, r] /= na
        B[:, r] /= nb
        C[:, r] /= nc
        sigma[r] *= na * nb * nc
    return A, B, C, sigma


def _cp_reconstruct(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Full tensor reconstruction from CP factors."""
    return np.einsum("ir,jr,kr,r->ijk", A, B, C, sigma)


def _observed_loss(T: np.ndarray, M: np.ndarray, T_hat: np.ndarray) -> float:
    diff = (T_hat - T)[M]
    if diff.size == 0:
        return np.nan
    return float(np.mean(diff * diff))


def rclr_transform_tensor(da_ra_pct: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    rclr transform in composition (% proxy) space.

    Input space:
    - da_ra_pct: percent abundance proxy (>=0), dims=('participant','taxon','bin')

    For each (participant, bin):
    - observed taxa are entries > 0
    - geometric mean uses only observed taxa
    - rclr = log(x) - mean(log(x_observed))
    - unobserved positions (x<=0 or NaN) -> NaN

    Returns
    -------
    rclr_da : xr.DataArray
        rclr tensor with NaN on unobserved entries
    observed_mask : xr.DataArray
        bool mask (True where rclr is observed/valid)
    """
    if da_ra_pct.dims != ("participant", "taxon", "bin"):
        raise ValueError("da_ra_pct dims must be ('participant','taxon','bin')")

    X = da_ra_pct.values.astype(float)
    I, J, K = X.shape
    R = np.full((I, J, K), np.nan, dtype=float)
    M = np.zeros((I, J, K), dtype=bool)

    for i in range(I):
        for k in range(K):
            v = X[i, :, k]
            obs = np.isfinite(v) & (v > 0)
            if not np.any(obs):
                continue
            lv = np.log10(v[obs])
            gm_log = np.mean(lv)
            out = np.full(J, np.nan, dtype=float)
            out[obs] = np.log10(v[obs]) - gm_log
            R[i, :, k] = out
            M[i, :, k] = obs

    rclr_da = xr.DataArray(
        R,
        dims=da_ra_pct.dims,
        coords=da_ra_pct.coords,
        name="rclr_da",
    )
    observed_mask = xr.DataArray(
        M,
        dims=da_ra_pct.dims,
        coords=da_ra_pct.coords,
        name="observed_mask",
    )
    return rclr_da, observed_mask


def _update_A(T, M, A, B, C, sigma, ridge=1e-6):
    I, J, K = T.shape
    rank = A.shape[1]
    for i in range(I):
        idx = np.argwhere(M[i, :, :])  # columns: j,k
        if idx.size == 0:
            A[i, :] = 0.0
            continue
        j_idx = idx[:, 0]
        k_idx = idx[:, 1]
        Z = (B[j_idx, :] * C[k_idx, :]) * sigma[None, :]
        y = T[i, j_idx, k_idx]
        G = Z.T @ Z + ridge * np.eye(rank)
        rhs = Z.T @ y
        A[i, :] = np.linalg.solve(G, rhs)
    return A


def _update_B(T, M, A, B, C, sigma, ridge=1e-6):
    I, J, K = T.shape
    rank = B.shape[1]
    for j in range(J):
        idx = np.argwhere(M[:, j, :])  # columns: i,k
        if idx.size == 0:
            B[j, :] = 0.0
            continue
        i_idx = idx[:, 0]
        k_idx = idx[:, 1]
        Z = (A[i_idx, :] * C[k_idx, :]) * sigma[None, :]
        y = T[i_idx, j, k_idx]
        G = Z.T @ Z + ridge * np.eye(rank)
        rhs = Z.T @ y
        B[j, :] = np.linalg.solve(G, rhs)
    return B


def _update_C(T, M, A, B, C, sigma, ridge=1e-6):
    I, J, K = T.shape
    rank = C.shape[1]
    for k in range(K):
        idx = np.argwhere(M[:, :, k])  # columns: i,j
        if idx.size == 0:
            C[k, :] = 0.0
            continue
        i_idx = idx[:, 0]
        j_idx = idx[:, 1]
        Z = (A[i_idx, :] * B[j_idx, :]) * sigma[None, :]
        y = T[i_idx, j_idx, k]
        G = Z.T @ Z + ridge * np.eye(rank)
        rhs = Z.T @ y
        C[k, :] = np.linalg.solve(G, rhs)
    return C


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
    """
    Observed-only CP-ALS on rclr space.

    Objective:
    min || P_Omega(T) - P_Omega(sum_r sigma[r] * a_r (x) b_r (x) c_r) ||_F^2

    This is a CTF-style observed-only low-rank tensor reconstruction fallback,
    not a direct gemelli reconstructed tensor export.
    """
    T = rclr_da.values.astype(float)
    M = observed_mask.values.astype(bool)
    I, J, K = T.shape

    if rank <= 0:
        raise ValueError("rank must be > 0")
    if rank > min(I, J, K):
        warnings.warn(
            f"rank={rank} > min(shape)={min(I,J,K)} may be unstable.", RuntimeWarning
        )

    X_filled = np.where(M, T, 0.0)
    if init == "svd":
        A, B, C = _init_factors_svd(X_filled, rank, seed)
    else:
        A, B, C = _init_factors_random(T.shape, rank, seed)

    sigma = np.ones(rank, dtype=float)
    A, B, C, sigma = _normalize_factors(A, B, C, sigma)

    prev_loss = np.inf
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        A = _update_A(T, M, A, B, C, sigma)
        B = _update_B(T, M, A, B, C, sigma)
        C = _update_C(T, M, A, B, C, sigma)
        A, B, C, sigma = _normalize_factors(A, B, C, sigma)

        T_hat = _cp_reconstruct(A, B, C, sigma)
        loss = _observed_loss(T, M, T_hat)
        rel = abs(prev_loss - loss) / (abs(prev_loss) + 1e-12)

        if verbose:
            print(f"[fallback-ALS] iter={it:03d} loss={loss:.8e} rel_change={rel:.3e}")

        n_iter = it
        if np.isfinite(prev_loss) and rel < tol:
            converged = True
            break
        prev_loss = loss

    T_hat = _cp_reconstruct(A, B, C, sigma)
    final_loss = _observed_loss(T, M, T_hat)

    return {
        "T_hat": T_hat,
        "A": A,
        "B": B,
        "C": C,
        "sigma": sigma,
        "n_iter": n_iter,
        "final_loss": final_loss,
        "converged": converged,
    }


def _try_gemelli_path(*args, **kwargs) -> Optional[Dict[str, object]]:
    """
    Optional gemelli path.
    Returns None if no stable low-level API is available for factor export/reconstruction.
    """
    try:
        import gemelli  # noqa: F401
    except Exception:
        return None

    # Intentionally conservative:
    # if factor/singular-value export is not directly available, do not reverse-engineer
    # from ordination object. Use fallback instead.
    return None


def ctf_complete_tensor_from_log_da(
    da: xr.DataArray,
    observed_mask: xr.DataArray | None = None,
    eps: float = 1e-3,
    rank: int = 3,
    max_iter: int = 200,
    tol: float = 1e-5,
    init: str = "svd",
    seed: int = 0,
    method: str = "auto",
    verbose: bool = True,
):
    """
    CTF-style observed-only low-rank tensor reconstruction from log(percent + eps) tensor.

    Data spaces:
    1) Input `da`: log(percent + eps) space
    2) Recover `da_ra_pct`: percent abundance proxy
       da_ra_pct = clip(exp(da) - eps, lower=0)
    3) rclr space:
       rclr on each (participant, bin) taxon vector, observed taxa only (>0), else NaN
    4) observed-only CP low-rank completion in rclr space
    5) Back to composition:
       W=exp(T_hat), P=W/sum_taxon(W), ctf_comp_pct=100*P
    6) Back to log-space working tensor:
       ctf_imp = log(ctf_comp_pct + eps)

    Returns
    -------
    out : dict
        {
            "da_ra_pct": xr.DataArray,
            "rclr_da": xr.DataArray,
            "observed_mask": xr.DataArray,
            "ctf_rclr_hat": xr.DataArray,
            "ctf_comp_pct": xr.DataArray,
            "ctf_imp": xr.DataArray,
            "factors": {"A","B","C","sigma"},
            "meta": {"method_used","rank","n_iter","final_loss","converged"}
        }
    """
    _validate_da(da)
    if method not in {"auto", "gemelli", "fallback"}:
        raise ValueError("method must be one of {'auto','gemelli','fallback'}")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    # 1) log10(percent+eps) -> percent abundance proxy
    # Keep transform pair strictly consistent with np.log10(...).
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

    da_ra_pct = xr.DataArray(
        raw,
        dims=da.dims,
        coords=da.coords,
        name="da_ra_pct",
    )

    # 2) rclr + observed mask
    # An explicit mask takes precedence, so floating-point error never decides
    # what counts as observed.
    if observed_mask is None:
      rclr_da, observed_mask = rclr_transform_tensor(da_ra_pct)
    else:
      if observed_mask.dims != da.dims:
        raise ValueError("observed_mask dims must match da.dims")
      if observed_mask.shape != da.shape:
        raise ValueError("observed_mask shape must match da.shape")

      X = da_ra_pct.values.astype(float)
      M = observed_mask.values.astype(bool)

      R = np.full_like(X, np.nan, dtype=float)
      I, J, K = X.shape
      for i in range(I):
        for k in range(K):
            obs = M[i, :, k]
            if not np.any(obs):
                continue
            v = X[i, obs, k]
            lv = np.log10(v)
            gm_log = np.mean(lv)
            R[i, obs, k] = lv - gm_log

      rclr_da = xr.DataArray(
        R, dims=da.dims, coords=da.coords, name="rclr_da"
         )
    obs_ratio = float(observed_mask.values.mean())
    if verbose:
        print(f"[sanity] observed ratio in rclr tensor: {obs_ratio:.6f}")

    # 3) decomposition
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

    T_hat = fit["T_hat"]

    ctf_rclr_hat = xr.DataArray(
        T_hat,
        dims=da.dims,
        coords=da.coords,
        name="ctf_rclr_hat",
    )

    # 4) rclr -> composition (%)
    # stable softmax over taxon axis
    Tmax = np.max(T_hat, axis=1, keepdims=True)
    W = np.exp(T_hat - Tmax)
    Wsum = np.sum(W, axis=1, keepdims=True)
    Wsum = np.where(Wsum <= 0, 1.0, Wsum)
    P = W / Wsum  # sum over taxon == 1

    # Robust handling for sample-time with no observed taxa in original data:
    # force uniform composition to avoid unconstrained arbitrary profile.
    M = observed_mask.values
    sample_time_observed = np.any(M, axis=1)  # (participant, bin)
    no_obs = ~sample_time_observed
    if np.any(no_obs):
        J = da.sizes["taxon"]
        P[:, :, :] = np.where(
            no_obs[:, None, :],
            1.0 / J,
            P,
        )

    ctf_comp_pct_np = 100.0 * P
    ctf_comp_pct_np = np.where(ctf_comp_pct_np < 0, 0.0, ctf_comp_pct_np)

    ctf_comp_pct = xr.DataArray(
        ctf_comp_pct_np,
        dims=da.dims,
        coords=da.coords,
        name="ctf_comp_pct",
    )

    # 5) composition -> log(percent + eps)
    ctf_imp_np = np.log10(ctf_comp_pct_np + eps)
    ctf_imp_np = np.where(np.isfinite(ctf_imp_np), ctf_imp_np, np.log10(eps))

    ctf_imp = xr.DataArray(
        ctf_imp_np,
        dims=da.dims,
        coords=da.coords,
        name="ctf_imp",
    )

    # 6) final sanity summaries
    sum_taxon = ctf_comp_pct.sum("taxon")
    if verbose:
        vals = sum_taxon.values.ravel()
        print(
            "[sanity] ctf_comp_pct.sum('taxon') summary: "
            f"min={np.nanmin(vals):.6f}, mean={np.nanmean(vals):.6f}, "
            f"max={np.nanmax(vals):.6f}, std={np.nanstd(vals):.6e}"
        )

    if not np.all(np.isfinite(ctf_imp.values)):
        raise RuntimeError("ctf_imp contains non-finite values (NaN/inf/-inf).")

    out = {
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
    return out


# ---------------------------
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance
import argparse
import traceback


def _bin_to_time_value(bin_label):
    s = str(bin_label)
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-")
        return 0.5 * (float(a) + float(b))
    return float(s)


def _bin_key(x):
    s = str(x)
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return float(s)


def _format_frac(frac: float) -> str:
    return f"{float(frac):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _orthonormalize(M):
    Q, _ = np.linalg.qr(M)
    return Q


def _top_eigvecs(S, k):
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1][:k]
    U = vecs[:, idx]
    return _orthonormalize(U)


def _build_time_weights(t, sigma=3.0):
    t = np.asarray(t, dtype=float)
    dt2 = (t[:, None] - t[None, :]) ** 2
    W = np.exp(-dt2 / (2.0 * sigma**2))
    return W


def penalized_2d2_pca(
    da,
    time=None,
    d=10,
    q=10,
    lam=1.0,
    sigma=3.0,
    T0=10,
    T=30,
    seed=0,
    verbose=True
):
    rng = np.random.default_rng(seed)

    if time is None:
        time = da.coords["bin"].values
    time = np.array(time)
    M = len(time)

    A_list = []
    for w in time:
        Ai = da.sel(bin=w).values
        A_list.append(Ai.astype(float))
    A = np.stack(A_list, axis=0)
    M_, m, n = A.shape
    assert M_ == M

    t = np.array([_bin_to_time_value(x) for x in time], dtype=float)
    W = _build_time_weights(t, sigma=sigma)

    Abar = A.mean(axis=0)
    Ac = A - Abar

    Z = rng.standard_normal((m, q))
    Z = _orthonormalize(Z)

    if verbose:
        print(f"[Phase 1] lambda=0, iter={T0}, M={M}, m={m}, n={n}, q={q}, d={d}")

    for _ in range(T0):
        ZZt = Z @ Z.T
        Astar0 = np.zeros((n, n), dtype=float)
        for i in range(M):
            Bi = Ac[i]
            Astar0 += Bi.T @ ZZt @ Bi
        Astar0 /= M
        X = _top_eigvecs(Astar0, d)

        XXt = X @ X.T
        Astarstar0 = np.zeros((m, m), dtype=float)
        for i in range(M):
            Bi = Ac[i]
            Astarstar0 += Bi @ XXt @ Bi.T
        Astarstar0 /= M
        Z = _top_eigvecs(Astarstar0, q)

    if verbose:
        print(f"[Phase 2] lambda={lam}, sigma={sigma}, iter={T}")

    for _ in range(T):
        ZZt = Z @ Z.T
        Astar = np.zeros((n, n), dtype=float)

        for i in range(M):
            Bi = Ac[i]
            Astar += Bi.T @ ZZt @ Bi
        Astar /= M

        Pen = np.zeros((n, n), dtype=float)
        for i in range(M):
            for j in range(M):
                wij = W[i, j]
                if wij == 0.0:
                    continue
                Dij = A[i] - A[j]
                Pen += wij * (Dij.T @ ZZt @ Dij)
        Astar = Astar - lam * Pen
        X = _top_eigvecs(Astar, d)

        XXt = X @ X.T
        Astarstar = np.zeros((m, m), dtype=float)
        for i in range(M):
            Bi = Ac[i]
            Astarstar += Bi @ XXt @ Bi.T
        Astarstar /= M

        Pen2 = np.zeros((m, m), dtype=float)
        for i in range(M):
            for j in range(M):
                wij = W[i, j]
                if wij == 0.0:
                    continue
                Dij = A[i] - A[j]
                Pen2 += wij * (Dij @ XXt @ Dij.T)
        Astarstar = Astarstar - lam * Pen2
        Z = _top_eigvecs(Astarstar, q)

    C = np.zeros((M, q, d), dtype=float)
    for i in range(M):
        C[i] = Z.T @ A[i] @ X

    meta = {"bin": time, "t": t, "W": W, "Abar": Abar}
    return X, Z, C, meta


def _safe_pearson(x, y):
    if x.size < 2:
        return np.nan
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_wasserstein(x, y):
    if x.size == 0 or y.size == 0:
        return np.nan
    return float(wasserstein_distance(x, y))


def evaluate_per_bin(da_true, da_before, da_after, bins, eval_mask=None, bin_dim="bin"):
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

        rows.append({
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
        })
    return pd.DataFrame(rows)


def _weighted_avg(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.average(v[m], weights=w[m]))


def _build_weighted_summary(df_metrics):
    # Repeat is the statistical unit. This is only within-repeat bin aggregation.
    rows = []
    keys = ["scope"]
    for key in ["frac", "repeat"]:
        if key in df_metrics.columns:
            keys = [key] + keys

    for group_vals, g in df_metrics.groupby(keys):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        kv = dict(zip(keys, group_vals))
        w = g["n_eval"].to_numpy(dtype=float)
        rows.append({
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
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["frac", "repeat", "scope"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


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


def build_truth_once(
    truth_path,
    force_rebuild=False,
    save_format="netcdf",
    compress_level=4,
    rank=3,
    method="fallback",
    ctf_seed=0,
    ctf_verbose=True
):
    truth_path = Path(truth_path)
    if truth_path.exists() and (not force_rebuild):
        da_gt = _load_dataarray(truth_path, var_name="y_true")
        da_gt.name = "y_true"
        print(f"[truth] loaded existing truth from {truth_path}")
        return da_gt

    out = ctf_complete_tensor_from_log_da(
        da=da,
        observed_mask=da_observed_mask,
        eps=eps,
        rank=rank,
        method=method,
        seed=ctf_seed,
        verbose=ctf_verbose,
    )
    da_gt = out["ctf_imp"].copy(deep=True)
    da_gt.name = "y_true"

    truth_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = _save_dataarray(
        da_gt, "y_true", truth_path, save_format=save_format, compress_level=compress_level
    )
    print(f"[truth] built and saved truth to {saved_path}")
    return da_gt


def make_random_mask_strict(da_gt, bins, frac=0.10, seed=0, eps=1e-3):
    rng = np.random.default_rng(seed)
    da_sub = da_gt.sel(bin=bins)
    vals = da_sub.values

    # Strict injection region: only truth > log10(eps).
    eligible = np.isfinite(vals) & (vals > np.log10(eps))
    idx = np.flatnonzero(eligible)
    n_eligible = int(idx.size)
    if n_eligible == 0 or frac <= 0:
        mask = np.zeros(vals.shape, dtype=bool)
        return xr.DataArray(mask, coords=da_sub.coords, dims=da_sub.dims, name="rand_mask_to_impute")

    n_pick = int(np.round(frac * n_eligible))
    if frac > 0 and n_pick == 0:
        n_pick = 1
    n_pick = min(n_pick, n_eligible)

    pick = rng.choice(idx, size=n_pick, replace=False)
    mask = np.zeros(vals.size, dtype=bool)
    mask[pick] = True
    mask = mask.reshape(vals.shape)
    return xr.DataArray(mask, coords=da_sub.coords, dims=da_sub.dims, name="rand_mask_to_impute")


def make_random_mask(da_ref, bins, frac=0.10, seed=0):
    return make_random_mask_strict(da_ref, bins=bins, frac=frac, seed=seed, eps=eps)


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
):
    bins_use = np.array(da_gt.coords["bin"].values)
    zero_value = np.log10(eps)

    da_rand_mask = make_random_mask_strict(da_gt, bins_use, frac=frac, seed=mask_seed, eps=eps)
    injected_cells = int(da_rand_mask.values.sum())

    da_zi = da_gt.copy(deep=True)
    vals = da_zi.sel(bin=bins_use).values.copy()
    vals[da_rand_mask.values.astype(bool)] = zero_value
    da_zi.loc[dict(bin=bins_use)] = vals
    da_zi.name = "log_abundance_zero_inflated"

    Xf, Zf, Cf, metaf = penalized_2d2_pca(
        da=da_zi, time=bins_use,
        d=d, q=q,
        lam=lam, sigma=sigma,
        T0=T0, T=T,
        seed=algo_seed, verbose=verbose
    )

    bins_trained = np.array(metaf["bin"])
    Abarf = metaf["Abar"]
    da_impf = da_zi.copy(deep=True)
    replaced = 0

    for ti, b in enumerate(bins_trained):
        Ai = da_impf.sel(bin=b).values
        Ahat = Abarf + (Zf @ Cf[ti] @ Xf.T)
        mk = da_rand_mask.sel(bin=b).values.astype(bool)
        nrep = int(mk.sum())
        replaced += nrep
        if nrep > 0:
            Ai_new = Ai.copy()
            Ai_new[mk] = Ahat[mk]
            da_impf.loc[dict(bin=b)] = Ai_new

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

    df_bin_one = pd.concat([df_injected, df_all], ignore_index=True)
    df_bin_one["frac"] = float(frac)
    df_bin_one["repeat"] = int(repeat)
    df_bin_one["mask_seed"] = int(mask_seed)
    df_bin_one["algo_seed"] = int(algo_seed)
    df_bin_one["injected_cells"] = int(injected_cells)
    df_bin_one["replaced_cells"] = int(replaced)
    df_bin_one["bin_order"] = df_bin_one["bin"].map(_bin_key)
    df_bin_one = df_bin_one.sort_values(["scope", "bin_order"]).drop(columns=["bin_order"])

    df_repeat_one = _build_weighted_summary(df_bin_one)
    df_repeat_one["frac"] = float(frac)
    df_repeat_one["repeat"] = int(repeat)
    df_repeat_one["mask_seed"] = int(mask_seed)
    df_repeat_one["algo_seed"] = int(algo_seed)
    df_repeat_one["injected_cells"] = int(injected_cells)
    df_repeat_one["replaced_cells"] = int(replaced)
    df_repeat_one = df_repeat_one[
        [
            "frac", "repeat", "scope", "n_eval",
            "mse_before", "mse_after", "mse_improve",
            "pearson_before", "pearson_after", "pearson_improve",
            "wd_before", "wd_after", "wd_improve",
            "mask_seed", "algo_seed", "injected_cells", "replaced_cells",
        ]
    ]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frac_str = _format_frac(frac)
    repeat_str = f"{int(repeat):03d}"

    bin_path = outdir / f"repeat_metrics_frac_{frac_str}_repeat_{repeat_str}.csv"
    repeat_path = outdir / f"repeat_summary_frac_{frac_str}_repeat_{repeat_str}.csv"
    df_bin_one.to_csv(bin_path, index=False)
    df_repeat_one.to_csv(repeat_path, index=False)

    if save_imputed_matrix:
        _save_dataarray(
            da_impf, "y_imp",
            outdir / f"imputed_frac_{frac_str}_repeat_{repeat_str}",
            save_format=save_format,
            compress_level=compress_level,
        )

    injected_eval = int(df_repeat_one.loc[df_repeat_one["scope"].eq("InjectedCells"), "n_eval"].sum())
    print(
        f"[repeat done] frac={frac:.4f} repeat={repeat} mask_seed={mask_seed} "
        f"algo_seed={algo_seed} injected={injected_cells} eval_cells={injected_eval}"
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
            )
            bin_parts.append(df_bin_one)
            repeat_parts.append(df_repeat_one)
        except Exception as e:
            err_text = traceback.format_exc()
            frac_str = _format_frac(frac)
            repeat_str = f"{int(r):03d}"
            fail_path = outdir / f"repeat_failed_frac_{frac_str}_repeat_{repeat_str}.log"
            fail_path.write_text(err_text, encoding="utf-8")
            fail_rows.append({
                "frac": float(frac),
                "repeat": int(r),
                "mask_seed": int(mask_seed_base + r),
                "algo_seed": int(algo_seed),
                "error": str(e),
                "log_path": str(fail_path),
            })
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


def summarize_across_repeats(df_repeat):
    metrics = [
        "mse_before", "mse_after", "mse_improve",
        "pearson_before", "pearson_after", "pearson_improve",
        "wd_before", "wd_after", "wd_improve",
    ]
    if df_repeat.empty:
        return pd.DataFrame(columns=["frac", "scope", "metric", "n", "mean", "sd", "sem", "ci95_low", "ci95_high"])

    rows = []
    for (frac, scope), g in df_repeat.groupby(["frac", "scope"]):
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            if metric in {"pearson_before", "pearson_after"}:
                stat = _summary_pearson_fisher(vals)
            else:
                stat = _summary_real(vals)
            rows.append({
                "frac": float(frac),
                "scope": scope,
                "metric": metric,
                **stat,
            })
    return pd.DataFrame(rows).sort_values(["frac", "scope", "metric"]).reset_index(drop=True)


def merge_repeat_outputs(outdir):
    outdir = Path(outdir)
    repeat_files = sorted(outdir.glob("repeat_summary_frac_*_repeat_*.csv"))
    bin_files = sorted(outdir.glob("repeat_metrics_frac_*_repeat_*.csv"))

    df_repeat = pd.concat([pd.read_csv(p) for p in repeat_files], ignore_index=True) if repeat_files else pd.DataFrame()
    df_bin = pd.concat([pd.read_csv(p) for p in bin_files], ignore_index=True) if bin_files else pd.DataFrame()
    if not df_repeat.empty:
        df_repeat = df_repeat.drop_duplicates(subset=["frac", "repeat", "scope"], keep="last")
    if not df_bin.empty:
        df_bin = df_bin.drop_duplicates(subset=["frac", "repeat", "scope", "bin"], keep="last")

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

    return df_bin_all, df_repeat_all, df_errorbar, df_fail_all


def _parse_fracs(frac_arg, fracs_arg):
    if frac_arg is not None:
        return [float(frac_arg)]
    if fracs_arg is None or str(fracs_arg).strip() == "":
        return [0.1, 0.3, 0.5, 0.7, 0.9]
    vals = [float(x.strip()) for x in str(fracs_arg).split(",") if x.strip() != ""]
    return vals if vals else [0.1, 0.3, 0.5, 0.7, 0.9]


def main():
    parser = argparse.ArgumentParser(
        description="CTF reconstruction simulation with repeat-level aggregation and error-bar summary."
    )
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--n_repeats", type=int, default=100)
    parser.add_argument("--fracs", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--truth_path", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="artifacts/simulation3")
    parser.add_argument("--mask_seed", type=int, default=123)
    parser.add_argument("--algo_seed", type=int, default=0)
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
    parser.add_argument("--force_rebuild_truth", action="store_true")
    parser.add_argument("--build_truth_only", action="store_true")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_repeat_outputs(outdir=outdir)
        return

    if args.truth_path is None:
        truth_path = outdir / ("truth_ctf.zarr" if args.save_format == "zarr" else "truth_ctf.nc")
    else:
        truth_path = Path(args.truth_path)

    da_gt = build_truth_once(
        truth_path=truth_path,
        force_rebuild=args.force_rebuild_truth,
        save_format=args.save_format,
        compress_level=args.compress_level,
        rank=args.ctf_rank,
        method=args.ctf_method,
        ctf_seed=args.ctf_seed,
        ctf_verbose=(not args.quiet),
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
            verbose=(not args.quiet),
        )
        merge_repeat_outputs(outdir=outdir)
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
            verbose=(not args.quiet),
        )
        merge_repeat_outputs(outdir=outdir)
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
        verbose=(not args.quiet),
    )

from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
# The preprocessing above defines the variables this section consumes:
#   da               log(percent + eps) DataArray, dims participant x taxon x bin
#   da_observed_mask observation mask, same dims as da
#   eps              log-transform offset (e.g. 1e-3)

# ===================== CTF reconstruction and export =====================
def save_ctf_imp(da, da_observed_mask, eps, save_path, rank=3, method="fallback"):
    """Build ctf_imp and write it to a NetCDF file.

    Parameters
    ----------
    da : DataArray in log(percent + eps) space
    da_observed_mask : observation-mask DataArray
    eps : log-transform offset
    save_path : output path, e.g. "ctf_imp.nc"
    rank : rank of the CTF reconstruction
    method : reconstruction backend (fallback / gemelli / auto)
    """
    # Run the CTF tensor reconstruction
    ctf_result = ctf_complete_tensor_from_log_da(
        da=da,
        observed_mask=da_observed_mask,
        eps=eps,
        rank=rank,
        method=method,
        verbose=True
    )
    
    # Extract the ctf_imp DataArray
    ctf_imp_da = ctf_result["ctf_imp"]
    
    # NetCDF encoding: compress to keep the file small
    encoding = {
        ctf_imp_da.name: {
            "zlib": True,        # enable compression
            "complevel": 4,      # compression level 1-9; 4 is a balanced choice
            "dtype": "float32"   # reduced precision; smaller file, no effect on the analysis
        }
    }
    
    # Make sure the output directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the NetCDF file
    ctf_imp_da.to_netcdf(
        path=save_path,
        encoding=encoding,
        engine="netcdf4"
    )
    
    print(f"ctf_imp written to {save_path.absolute()}")
    return ctf_imp_da

# Run the export
ctf_imp_da = save_ctf_imp(
    da=da,                  # log-space DataArray built above
    da_observed_mask=da_observed_mask,  # observation mask built above
    eps=1e-3,               # must match the eps used in the log transform above
    save_path="ctf_imp_new.nc",
    rank=3,
    method="fallback"
)



if __name__ == "__main__":
    main()
