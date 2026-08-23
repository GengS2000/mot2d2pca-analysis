from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


class _Coord:
    """Minimal coordinate wrapper used by the local DataArray fallback."""

    def __init__(self, values):
        self.values = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)

    @property
    def size(self):
        return self.values.size

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]


class DataArray:
    """Small subset of xarray.DataArray required by the experiment scripts.

    This fallback is intentionally narrow. If real xarray is installed outside
    this repository, remove this file and the same project code will use it.
    """

    def __init__(self, data, dims: Sequence[str], coords: Mapping[str, Iterable] | None = None, name: str | None = None):
        self.values = np.asarray(data)
        self.dims = tuple(dims)
        self.name = name
        if self.values.ndim != len(self.dims):
            raise ValueError("number of dims must match data ndim")
        coords = {} if coords is None else dict(coords)
        self.coords: Dict[str, _Coord] = {}
        for axis, dim in enumerate(self.dims):
            vals = coords.get(dim, np.arange(self.values.shape[axis]))
            coord = vals if isinstance(vals, _Coord) else _Coord(vals)
            if coord.size != self.values.shape[axis]:
                raise ValueError(f"coord {dim!r} length does not match data shape")
            self.coords[dim] = coord

    @property
    def shape(self):
        return self.values.shape

    @property
    def size(self):
        return self.values.size

    @property
    def sizes(self):
        return {dim: self.values.shape[i] for i, dim in enumerate(self.dims)}

    def copy(self, deep: bool = True):
        data = self.values.copy() if deep else self.values
        coords = {k: v.values.copy() for k, v in self.coords.items()}
        return DataArray(data, dims=self.dims, coords=coords, name=self.name)

    def sel(self, **indexers):
        data = self.values
        dims = list(self.dims)
        coords = {k: v.values for k, v in self.coords.items()}
        for dim, label in indexers.items():
            if dim not in dims:
                raise KeyError(dim)
            axis = dims.index(dim)
            coord_vals = self.coords[dim].values
            matches = np.flatnonzero(coord_vals == label)
            if matches.size == 0:
                raise KeyError(label)
            data = np.take(data, int(matches[0]), axis=axis)
            dims.pop(axis)
            coords.pop(dim, None)
        return DataArray(data, dims=tuple(dims), coords={d: coords[d] for d in dims}, name=self.name)

    def transpose(self, *dims):
        if not dims:
            dims = tuple(reversed(self.dims))
        if tuple(dims) == self.dims:
            return self.copy(deep=False)
        axes = [self.dims.index(d) for d in dims]
        data = np.transpose(self.values, axes=axes)
        coords = {d: self.coords[d].values for d in dims}
        return DataArray(data, dims=tuple(dims), coords=coords, name=self.name)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self):
        return f"DataArray(name={self.name!r}, dims={self.dims!r}, shape={self.shape!r})"

