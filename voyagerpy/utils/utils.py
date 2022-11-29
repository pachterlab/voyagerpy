#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from anndata import AnnData
import numpy as np

def is_highres(adata: AnnData) -> bool:
    if("hires" in adata.uns["spatial"]["img"]):
        return True
    if("lowres" in adata.uns["spatial"]["img"]):
        return False
    raise ValueError(
        "Cannot find image data in .uns['spatial']"
    )


    
def calculate_metrics(adata: AnnData) -> AnnData:
    adata.var_names_make_unique()
    #forcells
    #n_genes_by_counts
    adata.obs["n_genes_by_counts"] = np.diff(adata.X.tocsr().indptr)
    #total_counts
    adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).reshape((adata.X.shape[0]))
    #prop_mito
    
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    adata.obs['pct_counts_mt'] = np.sum(adata[:,adata.var["mt"]].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    
    
    #for_genes
    #detected in nr of barcodes
    adata.var["n_cells_by_counts"] = np.diff(adata.X.tocsc().indptr)
    #barcodes per feature
    adata.var["total_counts"] = np.array(adata.X.sum(axis=0)).reshape((adata.X.shape[1]))
    return adata
    

