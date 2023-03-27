#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def is_highres(adata: AnnData) -> bool:
    if "hires" in adata.uns["spatial"]["img"]:
        return True
    if "lowres" in adata.uns["spatial"]["img"]:
        return False
    raise ValueError("Cannot find image data in .uns['spatial']")


def is_lowres(adata: AnnData) -> bool:
    if "lowres" in adata.uns["spatial"]["img"]:
        return True
    if "hires" in adata.uns["spatial"]["img"]:
        return False
    raise ValueError("Cannot find image data in .uns['spatial']")


def make_unique(items: List) -> List:
    items = items[:]
    for i in range(len(items) - 1, -1, -1):
        if items.count(items[i]) > 1:
            items.pop(i)
    return items


def get_scale(adata: AnnData, res: Optional[str] = None) -> float:
    if res not in [None, "hi", "hires", "lo", "lowres"]:
        raise ValueError(f"Unrecognized value {res} for res.")

    hires = is_highres(adata) and res not in ["lowres", "lo"]
    scale_key = "tissue_hires_scalef" if hires else "tissue_lowres_scalef"
    return adata.uns["spatial"]["scale"][scale_key]


def calculate_metrics(adata: AnnData) -> AnnData:
    adata.var_names_make_unique()
    # forcells
    # n_genes_by_counts
    adata.obs["n_genes_by_counts"] = np.diff(adata.X.tocsr().indptr)  # type:ignore
    # total_counts
    adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).reshape((adata.X.shape[0]))  # type:ignore
    # prop_mito

    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    adata.obs["pct_counts_mt"] = np.sum(adata[:, adata.var["mt"]].X, axis=1).A1 / np.sum(adata.X, axis=1).A1  # type:ignore

    # for_genes
    # detected in nr of barcodes
    adata.var["n_cells_by_counts"] = np.diff(adata.X.tocsc().indptr)  # type: ignore
    # barcodes per feature
    adata.var["total_counts"] = np.array(adata.X.sum(axis=0)).reshape((adata.X.shape[1]))  # type: ignore
    return adata


def add_per_cell_qcmetrics(adata: AnnData, subsets: Dict[str, np.ndarray], force: bool = False) -> None:
    if "sum" not in adata.obs.keys() or force:
        adata.obs["sum"] = adata.X.sum(axis=1)

    if "detected" not in adata.obs.keys() or force:
        adata.obs["detected"] = (adata.X > 0).sum(axis=1)

    for key, arr in subsets.items():

        if arr.dtype in ("bool", "O"):
            arr = arr.astype(adata.X.dtype)

        sum_key = f"subsets_{key}_sum"
        det_key = f"subsets_{key}_detected"
        prc_key = f"subsets_{key}_percent"

        if sum_key not in adata.obs.keys() or force:
            adata.obs[sum_key] = adata.X.dot(arr)

        if det_key not in adata.obs.keys() or force:
            adata.obs[det_key] = (adata.X > 0).dot(arr)

        if prc_key not in adata.obs.keys() or force:
            adata.obs[prc_key] = adata.obs[sum_key] / adata.obs["sum"]


def log_norm_counts(adata: AnnData, layer: Optional[str] = None, inplace: bool = True, base: Optional[int] = 2, pseudocount: int = 1):

    # Equivalent to:
    # target_sum = adata.X.sum(axis=1).mean()
    # sc.pp.normalize_total(adata, target_sum=target_sum)
    # sc.pp.log1p(adata, base=2)

    if not inplace:
        adata = adata.copy()

    X = adata.X if layer is None else adata.layers[layer]

    cell_sums = sp.csr_matrix(X.sum(axis=1)).toarray()
    cell_sums /= cell_sums.mean()

    X = sp.csr_matrix(X / cell_sums)

    log_funcs = {2: np.log2, 10: np.log10, None: np.log}
    other_log = lambda x: np.log(x) / np.log(base)  # type: ignore # noqa: E731
    log = log_funcs.get(base, other_log)

    if pseudocount == 1:
        # This ensures we don't interact with the zeros in X
        X.data = log(X.data + pseudocount)
    else:
        X = log(X + pseudocount)
    adata.X = X
    return adata


def scale(X, center=True, unit_variance: bool = True, center_before_scale: bool = True, ddof: int = 1):
    is_sparse = isinstance(X, sp.csr_matrix)
    if is_sparse:
        X = X.todense()
    else:
        X = X.copy()

    kwargs = dict(axis=0, keepdims=True)
    if isinstance(X, np.matrix):
        kwargs.pop("keepdims")

    if center and center_before_scale:
        X -= X.mean(**kwargs)

    if unit_variance:
        std = X.std(ddof=ddof, **kwargs)
        w = np.where(std < 1e-8)
        std[w] = 1
        X = np.divide(X, std)

    if center and not center_before_scale:
        X -= X.mean(axis=0)

    return X


def normalize_csr(X: sp.csr_matrix, byrow: bool = True) -> sp.csr_matrix:
    axis = int(byrow)
    sum_ = sp.csr_matrix(X.sum(axis=axis))
    sum_.eliminate_zeros()
    sum_.data = 1 / sum_.data
    sum_ = sp.diags(sum_.toarray().ravel())
    return sum_.dot(X) if byrow else X.dot(sum_)
