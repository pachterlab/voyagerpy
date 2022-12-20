#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
from anndata import AnnData


def is_highres(adata: AnnData) -> bool:
    if "hires" in adata.uns["spatial"]["img"]:
        return True
    if "lowres" in adata.uns["spatial"]["img"]:
        return False
    raise ValueError("Cannot find image data in .uns['spatial']")


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
        sum_key = f"subsets_{key}_sum"
        det_key = f"subsets_{key}_detected"
        prc_key = f"subsets_{key}_percent"

        if sum_key not in adata.obs.keys() or force:
            adata.obs[sum_key] = adata.X.dot(arr)

        if det_key not in adata.obs.keys() or force:
            adata.obs[det_key] = (adata.X > 0).dot(arr)

        if prc_key not in adata.obs.keys() or force:
            adata.obs[prc_key] = adata.obs[sum_key] / adata.obs["sum"]
