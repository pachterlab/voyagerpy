#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from voyagerpy import metrics as mtr
from voyagerpy import plotting as plt
from voyagerpy import spatial as spt
from voyagerpy import utils as utl
from voyagerpy.read import read_10x_visium
from voyagerpy.read import read_10x_counts

from anndata import AnnData
from geopandas import GeoDataFrame
from pandas import Index

from typing import Union, Tuple, TypeVar
from scipy import sparse
import numpy as np

from anndata import AnnData
import anndata


class Voyager(AnnData):
    def __init__(self, *args, **kwargs):
        X = args[0] if len(args) else kwargs.get("X", None)

        dtype_present = len(args) > 8 or "dtype" in kwargs
        dtype = None

        if not dtype_present:
            if isinstance(X, (AnnData, Voyager)):
                dtype = X.X.dtype
            elif X is None:
                dtype = None
            else:
                dtype = X.dtype

        _kwargs = dict(dtype=dtype)
        _kwargs.update(kwargs)

        super().__init__(*args, **_kwargs)
        barcode_geo = self.obsm.setdefault("geometry", gpd.GeoDataFrame(index=self.obs_names))

        if not isinstance(barcode_geo, anndata._core.views.DataFrameView) and not isinstance(barcode_geo, gpd.GeoDataFrame):
            self.obsm["geometry"] = gpd.GeoDataFrame(barcode_geo)

    def copy(self, *args) -> "Voyager":
        adata = super().copy(*args)
        return Voyager(adata)

    def __getitem__(self, index):
        oidx, vidx = self._normalize_indices(index)
        return Voyager(self, oidx=oidx, vidx=vidx, asview=True)


__all__ = [
    "metrics",
    "plotting",
    "spatial",
    "utils",
    "read_10x_counts",
    "read_10x_visium",
]
