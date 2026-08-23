from __future__ import annotations

import numpy as np

from .common import nrmse, pearson
from .model_adapter import fit_transform


def make_mask(data, fraction, mechanism, seed, minimum=5):
    """Deterministic artificial-missing mask; identical seed means identical mask."""
    rng = np.random.default_rng(int(seed))
    eligible = np.isfinite(data)
    out = np.zeros_like(eligible, dtype=bool)
    T, R, C = data.shape
    for t in range(T):
        for r in range(R):
            ids = np.flatnonzero(eligible[t, r])
            n = min(int(round(float(fraction) * ids.size)), max(ids.size - int(minimum), 0))
            if n == 0:
                continue
            if mechanism == "low_abundance":
                values = data[t, r, ids]
                weights = np.max(values) - values + 1e-8
                weights /= weights.sum()
                selected = rng.choice(ids, n, replace=False, p=weights)
            else:
                selected = rng.choice(ids, n, replace=False)
            out[t, r, selected] = True
    if mechanism in {"time_point", "time_window"}:
        # Retain the same overall fraction but concentrate one half in a
        # consecutive time window; this models a missed visit period.
        out[:] = False
        width = 1 if mechanism == "time_point" else max(1, min(T, int(round(T * fraction))))
        start = int(rng.integers(0, T - width + 1))
        window = np.arange(start, start + width)
        for t in window:
            for r in range(R):
                ids = np.flatnonzero(eligible[t, r])
                # A full block of visits is missing; the window width controls
                # the global missing fraction, while `minimum` preserves fit data.
                n = max(ids.size - int(minimum), 0)
                if n:
                    out[t, r, rng.choice(ids, n, replace=False)] = True
    elif mechanism == "subject_dropout":
        out[:] = False
        rows = rng.permutation(R)
        n_rows = max(1, int(round(R * fraction)))
        for r in rows[:n_rows]:
            for t in range(T):
                ids = np.flatnonzero(eligible[t, r])
                n = max(ids.size - int(minimum), 0)
                if n:
                    out[t, r, rng.choice(ids, n, replace=False)] = True
    return out


def evaluate(truth, prediction, mask):
    y, yhat = truth[mask], prediction[mask]
    if y.size == 0:
        raise ValueError("Empty evaluation mask")
    # Mean trajectory correlation across time, ignoring degenerate slices.
    corrs = [
        pearson(truth[ti][mask[ti]], prediction[ti][mask[ti]])
        for ti in range(mask.shape[0])
        if mask[ti].any()
    ]
    corrs = [x for x in corrs if np.isfinite(x)]
    return {
        "n_masked": int(mask.sum()),
        "nrmse": nrmse(y, yhat),
        "pearson": pearson(y, yhat),
        "trajectory_correlation": float(np.mean(corrs)) if corrs else float("nan"),
    }


def run_variant(data, observed, times, params, seed, variant="full"):
    d, q, lam, sigma = int(params["d"]), int(params["q"]), float(params["lam"]), float(params["sigma"])
    if variant == "temporal_only":
        out = data.copy()
        for r in range(data.shape[1]):
            for c in range(data.shape[2]):
                good = observed[:, r, c]
                if good.any():
                    out[:, r, c] = np.interp(times, times[good], data[:, r, c][good])
                else:
                    out[:, r, c] = 0.0
        return {"imputed": np.where(observed, data, out), "runtime_seconds": 0.0, "converged": True, "n_iter": 0}
    if variant == "no_temporal" or variant == "low_rank_only":
        lam = 0.0
    if variant == "plain_2d2pca":
        lam = 0.0
        work = np.where(observed, data, 0.0)
        obs = np.ones_like(observed, dtype=bool)
    elif variant == "no_mask_awareness":
        work = np.where(observed, data, 0.0)
        obs = np.ones_like(observed, dtype=bool)
    else:
        work, obs = data, observed
    if variant == "row_2dpca":
        d = data.shape[2]
    elif variant == "column_2dpca":
        q = data.shape[1]
    return fit_transform(work, obs, times, d, q, lam, sigma, seed)
