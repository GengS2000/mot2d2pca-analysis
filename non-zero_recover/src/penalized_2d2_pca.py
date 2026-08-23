from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from .preprocessing import EXPECTED_DIMS, validate_tensor


def _bin_to_time_value(bin_label) -> float:
    """Convert a bin label such as '5-8' to a numeric midpoint."""
    s = str(bin_label)
    if s == "0":
        return 0.0
    if "-" in s:
        a, b = s.split("-", 1)
        return 0.5 * (float(a) + float(b))
    return float(s)


def _orthonormalize(m: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis spanning the columns of m."""
    q, _ = np.linalg.qr(m, mode="reduced")
    return q


def _top_eigvecs(s: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors of a symmetric matrix, robust to small effective rank."""
    k_eff = min(int(k), s.shape[0])
    vals, vecs = np.linalg.eigh(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0))
    idx = np.argsort(vals)[::-1][:k_eff]
    return _orthonormalize(vecs[:, idx])


def _build_time_weights(t: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian time-neighborhood weights with a zero diagonal."""
    t = np.asarray(t, dtype=float)
    sigma = max(float(sigma), 1e-8)
    w = np.exp(-((t[:, None] - t[None, :]) ** 2) / (2.0 * sigma**2))
    np.fill_diagonal(w, 0.0)
    return w


def build_observed_mask(
    da_input: xr.DataArray,
    zero_value: float,
    mode: str = "positive_only_except_replace",
    replace_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Build a boolean training mask for imputation models."""
    validate_tensor(da_input, "da_input")
    vals = da_input.values
    finite = np.isfinite(vals)
    if mode == "positive_only":
        obs = finite & (vals > float(zero_value))
    elif mode == "all_finite":
        obs = finite
    elif mode == "all_finite_except_replace":
        if replace_mask is None:
            raise ValueError("replace_mask is required")
        obs = finite & ~replace_mask.values.astype(bool)
    elif mode == "positive_only_except_replace":
        if replace_mask is None:
            raise ValueError("replace_mask is required")
        obs = finite & (vals > float(zero_value)) & ~replace_mask.values.astype(bool)
    else:
        raise ValueError(f"Unsupported observed mask mode: {mode}")
    return xr.DataArray(obs, dims=EXPECTED_DIMS, coords=da_input.coords, name="observed_mask")


def penalized_2d2_pca(
    da: xr.DataArray,
    time=None,
    d: int = 1,
    q: int = 1,
    lam: float = 0.1,
    sigma: float = 4.0,
    T0: int = 30,
    T: int = 30,
    seed: int = 0,
    observed_mask: Optional[xr.DataArray] = None,
    verbose: bool = False,
):
    """Fit the mask-aware penalized 2D2-PCA model on a log-profile tensor."""
    validate_tensor(da, "da")
    if observed_mask is not None:
        validate_tensor(observed_mask, "observed_mask")
        if observed_mask.shape != da.shape:
            raise ValueError("observed_mask shape must match da")
    rng = np.random.default_rng(int(seed))
    if time is None:
        time = da.coords["bin"].values
    time = np.asarray(time)
    a = np.stack([da.sel(bin=b).values.astype(float) for b in time], axis=0)
    if observed_mask is None:
        obs = np.isfinite(a)
    else:
        obs = np.stack([observed_mask.sel(bin=b).values.astype(bool) for b in time], axis=0)
    a = np.where(np.isfinite(a), a, 0.0)
    m_time, n_participant, n_taxon = a.shape
    d_eff = min(max(int(d), 1), n_taxon)
    q_eff = min(max(int(q), 1), n_participant)
    tvals = np.asarray([_bin_to_time_value(x) for x in time], dtype=float)
    w_time = _build_time_weights(tvals, sigma=sigma)

    denom = np.maximum(obs.sum(axis=0), 1.0)
    abar = np.where(obs, a, 0.0).sum(axis=0) / denom
    ac = np.where(obs, a - abar[None, :, :], 0.0)
    z = _orthonormalize(rng.standard_normal((n_participant, q_eff)))

    if verbose:
        print(f"[penalized_2d2_pca] phase1 T0={T0}, phase2 T={T}, shape={da.shape}, d={d_eff}, q={q_eff}")
    for _ in range(max(int(T0), 0)):
        zzt = z @ z.T
        s_taxon = sum(ac[i].T @ zzt @ ac[i] for i in range(m_time)) / max(m_time, 1)
        x = _top_eigvecs(s_taxon, d_eff)
        xxt = x @ x.T
        s_participant = sum(ac[i] @ xxt @ ac[i].T for i in range(m_time)) / max(m_time, 1)
        z = _top_eigvecs(s_participant, q_eff)

    for _ in range(max(int(T), 0)):
        zzt = z @ z.T
        s_taxon = sum(ac[i].T @ zzt @ ac[i] for i in range(m_time)) / max(m_time, 1)
        pen_taxon = np.zeros((n_taxon, n_taxon), dtype=float)
        for i in range(m_time):
            for j in range(m_time):
                wij = w_time[i, j]
                if wij == 0:
                    continue
                oij = obs[i] & obs[j]
                if np.any(oij):
                    dij = np.where(oij, a[i] - a[j], 0.0)
                    pen_taxon += wij * (dij.T @ zzt @ dij)
        x = _top_eigvecs(s_taxon - float(lam) * pen_taxon, d_eff)

        xxt = x @ x.T
        s_participant = sum(ac[i] @ xxt @ ac[i].T for i in range(m_time)) / max(m_time, 1)
        pen_participant = np.zeros((n_participant, n_participant), dtype=float)
        for i in range(m_time):
            for j in range(m_time):
                wij = w_time[i, j]
                if wij == 0:
                    continue
                oij = obs[i] & obs[j]
                if np.any(oij):
                    dij = np.where(oij, a[i] - a[j], 0.0)
                    pen_participant += wij * (dij @ xxt @ dij.T)
        z = _top_eigvecs(s_participant - float(lam) * pen_participant, q_eff)

    c = np.zeros((m_time, q_eff, d_eff), dtype=float)
    for i in range(m_time):
        ai_eff = np.where(obs[i], a[i], abar)
        c[i] = z.T @ ai_eff @ x
    return x, z, c, {"bin": time, "t": tvals, "W": w_time, "Abar": abar, "observed_n": int(obs.sum())}

