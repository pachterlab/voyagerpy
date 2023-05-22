from typing import Tuple, TypeVar, Union

import anndata
import numpy as np
from anndata import AnnData
from geopandas import GeoDataFrame
from pandas import Index
from scipy import sparse


class VoyagerData(AnnData):
    def __init__(self, *args, **kwargs):
        X = args[0] if len(args) else kwargs.get("X", None)

        dtype_present = len(args) > 8 or "dtype" in kwargs
        dtype = None

        if not dtype_present:
            if isinstance(X, (AnnData, VoyagerData)):
                dtype = X.X.dtype
            elif X is None:
                dtype = None
            else:
                dtype = X.dtype

        _kwargs = dict(dtype=dtype)
        _kwargs.update(kwargs)

        super().__init__(*args, **_kwargs)
        barcode_geo = self.obsm.setdefault("geometry", GeoDataFrame(index=self.obs_names))

        if not isinstance(barcode_geo, anndata._core.views.DataFrameView) and not isinstance(barcode_geo, GeoDataFrame):
            self.obsm["geometry"] = GeoDataFrame(barcode_geo)

    def copy(self, *args) -> "VoyagerData":
        adata = super().copy(*args)
        return VoyagerData(adata)

    def __getitem__(self, index):
        oidx, vidx = self._normalize_indices(index)
        return VoyagerData(self, oidx=oidx, vidx=vidx, asview=True)
