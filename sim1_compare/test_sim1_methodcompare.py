import unittest

import numpy as np
import xarray as xr

from sim1_compare.sim1_methodcompare import (
    _fill_nearest_positive,
    _impute_local_method,
    penalized_2d2_pca,
)


FLOOR = -3.0
TIMES = np.array([0.0, 2.5, 6.5, 10.5])


def _da(values, name):
    values = np.asarray(values)
    return xr.DataArray(
        values,
        dims=("participant", "taxon", "bin"),
        coords={
            "participant": [f"p{i}" for i in range(values.shape[0])],
            "taxon": [f"x{j}" for j in range(values.shape[1])],
            "bin": ["0", "1-4", "5-8", "9-12"][: values.shape[2]],
        },
        name=name,
    )


class Sim1FallbackTests(unittest.TestCase):
    def test_unsupported_abar_is_log_floor_and_centered_residual_is_zero(self):
        values = np.array(
            [
                [
                    [FLOOR, FLOOR, FLOOR, FLOOR],
                    [-1.0, -1.0, -1.0, -1.0],
                ]
            ]
        )
        da = _da(values, "log_abundance")
        observed = da > FLOOR
        _, _, c, meta = penalized_2d2_pca(
            da,
            d=1,
            q=1,
            lam=0.03,
            sigma=4.0,
            t0=1,
            t=1,
            seed=7,
            observed_mask=observed,
            floor_value=FLOOR,
        )
        self.assertEqual(meta["Abar"][0, 0], FLOOR)
        self.assertEqual(meta["unsupported_trajectory_count"], 1)
        np.testing.assert_allclose(c, 0.0, atol=1e-12)

    def test_nearest_positive_uses_actual_time_and_never_chains(self):
        values = np.array([-1.0, FLOOR, FLOOR, -2.0])
        observed = np.array([True, False, False, True])
        left = _fill_nearest_positive(values, observed, TIMES, 1, FLOOR)
        right = _fill_nearest_positive(values, observed, TIMES, 2, FLOOR)
        self.assertEqual(left, -1.0)
        self.assertEqual(right, -2.0)

    def test_empty_trajectory_uses_nearest_time_taxon_positive_median(self):
        values = np.array(
            [
                [[FLOOR, FLOOR, FLOOR, FLOOR]],
                [[-1.0, FLOOR, FLOOR, -2.0]],
            ]
        )
        replace = np.zeros_like(values, dtype=bool)
        replace[0, 0, 1:3] = True
        observed = (values > FLOOR) & (~replace)
        result = _impute_local_method(
            da_input=_da(values, "input"),
            replace_mask=_da(replace, "replace"),
            observed_mask=_da(observed, "observed"),
            method="nearest_positive",
            method_params={},
            zero_value=FLOOR,
        )
        self.assertEqual(result.values[0, 0, 1], -1.0)
        self.assertEqual(result.values[0, 0, 2], -2.0)
        self.assertEqual(result.attrs["positive_trajectory_fallback_count"], 2)
        self.assertEqual(result.attrs["no_positive_support_count"], 0)

    def test_temporal_knn_does_not_use_floor_as_neighbor(self):
        values = np.array([[[-1.0, FLOOR, FLOOR, -2.0]]])
        replace = np.zeros_like(values, dtype=bool)
        replace[0, 0, 1:3] = True
        observed = (values > FLOOR) & (~replace)
        result = _impute_local_method(
            da_input=_da(values, "input"),
            replace_mask=_da(replace, "replace"),
            observed_mask=_da(observed, "observed"),
            method="temporal_knn",
            method_params={"k": 3, "bandwidth": 4.0},
            zero_value=FLOOR,
        )
        self.assertTrue(np.all(result.values[0, 0, 1:3] > FLOOR))

    def test_wholly_unsupported_taxon_remains_floor_and_is_counted(self):
        values = np.full((2, 1, 4), FLOOR)
        replace = np.zeros_like(values, dtype=bool)
        replace[0, 0, 2] = True
        observed = (values > FLOOR) & (~replace)
        result = _impute_local_method(
            da_input=_da(values, "input"),
            replace_mask=_da(replace, "replace"),
            observed_mask=_da(observed, "observed"),
            method="nearest_positive",
            method_params={},
            zero_value=FLOOR,
        )
        self.assertEqual(result.values[0, 0, 2], FLOOR)
        self.assertEqual(result.attrs["no_positive_support_count"], 1)


if __name__ == "__main__":
    unittest.main()
