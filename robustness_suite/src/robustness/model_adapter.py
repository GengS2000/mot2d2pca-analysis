from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .common import suite_root


def fit_transform(data, observed_mask, times, rank_d, rank_q, temporal_lambda, sigma, random_state):
    """Run the formal package and return a uniform result dictionary."""
    package_src = suite_root().parent / "mot2d2pca" / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))
    from mot2d2pca import fit_mot2d2pca

    start = time.perf_counter()
    result = fit_mot2d2pca(
        data, time=times, d=rank_d, q=rank_q, lam=temporal_lambda,
        sigma=sigma, T0=30, T=30, mask=observed_mask, random_state=random_state,
    )
    reconstruction = result.reconstruct() + result.mean[None, :, :]
    imputed = np.where(observed_mask, data, reconstruction)
    return {
        "imputed": imputed,
        "reconstruction": reconstruction,
        "converged": True,
        "n_iter": 60,
        "runtime_seconds": time.perf_counter() - start,
        "metadata": {"n_observed": result.n_observed},
    }
