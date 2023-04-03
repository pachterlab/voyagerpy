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

# SingleIndexable = TypeVar("SingleIndexable", int, slice, np.int64, np.int32, np.ndarray)
# _single_indexable_ = Union[int, slice, np.int64, np.int32, np.ndarray]
# Indexable = TypeVar("Indexable", _single_indexable_, Tuple[_single_indexable_, _single_indexable_], sparse.spmatrix)

# class VoyagerData(AnnData):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.obsm["geometry"] = GeoDataFrame(index=self.obs_names)
#         self.uns["spatial"] = {}

#     def copy(self, *args, **kwargs):
#         vdat = super().copy(*args, **kwargs)
#         vdat.obsm["geometry"] = self.obsm["geometry"].copy()
#         return vdat

#     def __getitem__(self, index: Indexable) -> "VoyagerData":
#         adata = super().__getitem__(index)



__all__ = [
    "metrics",
    "plotting",
    "spatial",
    "utils",
    "read_10x_counts",
    "read_10x_visium",
]
