import unittest

import numpy as np
import xarray as xr

from sim3_compare.sim3_compare import (
    DEFAULT_METHODS,
    _build_taxon_positive_fallbacks,
    _impute_local_trajectory,
    penalized_2d2_pca,
)


class Sim3RetainedMethodsTest(unittest.TestCase):
    def test_retained_method_set_is_aligned(self):
        self.assertEqual(
            DEFAULT_METHODS,
            [
                "penalized_2d2_pca",
                "ctf",
                "mean",
                "nearest_positive",
                "temporal_knn",
            ],
        )

    def test_nearest_positive_uses_actual_time_and_never_chains(self):
        floor = -3.0
        times = np.array([0.0, 2.5, 10.5])
        y = np.array([1.0, floor, floor])
        mask = np.array([False, True, True])
        out = _impute_local_trajectory(
            y=y,
            mask=mask,
            times=times,
            method="nearest_positive",
            params={},
            zero_value=floor,
            fallback_values=np.full(3, 0.25),
        )
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0])

    def test_no_positive_trajectory_uses_target_specific_config_b_fallback(self):
        floor = -3.0
        times = np.array([0.0, 2.5, 6.5])
        fallback = np.array([0.1, 0.2, 0.3])
        y = np.full(3, floor)
        mask = np.ones(3, dtype=bool)
        for method in ("mean", "nearest_positive", "temporal_knn"):
            out = _impute_local_trajectory(
                y=y,
                mask=mask,
                times=times,
                method=method,
                params={"k": 3, "kernel_sigma": 4.0},
                zero_value=floor,
                fallback_values=fallback,
            )
            np.testing.assert_allclose(out, fallback)

    def test_taxon_fallback_uses_nearest_supported_time_median(self):
        floor = -3.0
        times = np.array([0.0, 2.5, 6.5])
        values = np.array(
            [
                [[floor, floor, floor]],
                [[1.0, floor, 3.0]],
                [[2.0, floor, 5.0]],
            ]
        )
        replace = np.zeros_like(values, dtype=bool)
        replace[0, 0, :] = True
        fallback = _build_taxon_positive_fallbacks(values, replace, times, floor)
        np.testing.assert_allclose(fallback[0], [1.5, 1.5, 4.0])

    def test_p2d2_unsupported_abar_is_log_floor(self):
        floor = -3.0
        da = xr.DataArray(
            np.full((2, 1, 2), floor),
            dims=("participant", "taxon", "bin"),
            coords={"participant": ["p0", "p1"], "taxon": ["t0"], "bin": ["0", "1-4"]},
        )
        observed = xr.zeros_like(da, dtype=bool)
        _, _, _, meta = penalized_2d2_pca(
            da,
            d=1,
            q=1,
            lam=0.03,
            sigma=4.0,
            T0=1,
            T=1,
            seed=1,
            observed_mask=observed,
            floor_value=floor,
        )
        np.testing.assert_allclose(meta["Abar"], floor)
        self.assertEqual(meta["unsupported_trajectory_count"], 2)


if __name__ == "__main__":
    unittest.main()
