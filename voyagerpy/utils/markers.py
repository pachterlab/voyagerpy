#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy import sparse


def get_statistics(adata, clust1, clust2=None, test="mw", alternative="greater", skip=None, clust_str: str = "cluster"):
    """Run the mann whitney test for genes in a given anndata object,between cells in clust1 and clust2.

    Parameters
    ----------
    adata : anndata object
        Anndata object with that has already been given a leiden clustering, or custom one in adata.obs["cluster"]
    clust1 : String
        A single group of cells, as defined by the clustering.
    clust2 : String
        String defining the second group that is to be tested against the first,
        if this is none then the the test is cluster1 vs rest. The default is None.
    test : String, optional
        Type of test. For now only mann whitney is possible. The default is "mw".
    alternative : String, optional
        Direction of test. The default is "greater".
    skip : List or array of ints, optional
        This is a parameter to speed up calculations. The default method for combining p values,
        takes the largest one for each cluster, therefore 1 has already been seen in the row, that row is skipped,
        because that row will always be designated a p value of 1.
        The default is None.

    Returns
    -------
    arr_final : Array of shape n where n is adata.shape[1]
        The p values for each gene.

    """
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    # if "highly_variable" in adata.var:
    #    genes = np.array(adata.var.index.get_indexer(adata.var.loc[adata.var["highly_variable"] == True].index))
    # else:
    # genes = np.array(adata.var.index.get_indexer(adata.var.index))
    cl_index = adata.obs.index.get_indexer(adata.obs.loc[adata.obs[clust_str] == clust1].index)
    arr_final = np.ones((adata.shape[1]), dtype=float)
    if clust2 is None:
        comp_index = adata.obs.index.get_indexer(adata.obs.loc[adata.obs[clust_str] != clust1].index)

    else:
        comp_index = adata.obs.index.get_indexer(adata.obs.loc[adata.obs[clust_str] == clust2].index)
    X_cl = adata.X.todense()[cl_index, :]
    X_comp = adata.X.todense()[comp_index, :]

    if skip is not None:
        it_values = np.delete(np.array(range(adata.shape[1])), skip)
        X_cl = adata.X.todense()[cl_index[:, None], it_values]
        X_comp = adata.X.todense()[comp_index[:, None], it_values]
    else:
        it_values = range(adata.shape[1])
    if test == "mw":
        arr = mannwhitneyu(X_cl, X_comp, alternative=alternative)[1]
    else:
        pass

    if skip is not None:
        arr_final[it_values] = arr

    else:
        arr_final = arr

    return arr_final


def get_p_clusters(adata, clust, skip_precalc=False, pval_type="all", cluster: str = "cluster"):
    # rows = adata.X.shape[0]
    pval_arrs = []
    # nr_clust = adata.obs["cluster"].cat.codes
    colnames = []
    if skip_precalc:
        first = False
        skip_index = None
    else:
        first = True
        skip_index = None
    for i in np.array(adata.obs[cluster].cat.categories):
        if str(clust) == str(i):
            continue
        arr = get_statistics(adata, str(clust), str(i), test="mw", skip=skip_index, clust_str=cluster)
        if first:
            skip_index = np.where(arr == 1)[0]
            first = False
        pval_arrs.append(arr)
        colnames.append("clust" + str(clust) + "vs" + str(i))

    pval_np = np.array(pval_arrs)

    pval_np = np.transpose(pval_np.reshape((pval_np.shape[0], pval_np.shape[1])))

    df = pd.DataFrame(pval_np, index=adata.var.index, columns=colnames)

    if pval_type == "all":
        arr = df.max(axis=1)
    else:
        arr = df

    return arr


def get_marker_genes(adata, hvg=False, hvg_string="highly_variable", cluster: str = "cluster"):
    if cluster not in adata.obs:
        Exception("There need to be clusters defined for this anndata object")
    nr_clust = adata.obs[cluster].cat.codes
    markers_desc = []
    adata_obj = adata if not hvg else adata[:, adata.var[hvg_string]].copy()
    for i in np.array(adata.obs[cluster].cat.categories):
        curr_cl_data = get_p_clusters(adata_obj, i, cluster=cluster)
        # if hvg is None:
        #     curr_cl_data = get_p_clusters(adata, i, cluster=cluster)
        # else:
        #     curr_cl_data = get_p_clusters(adata[:, adata.var[hvg_string]].copy(), i, cluster=cluster)
        markers_desc.append(np.array(curr_cl_data[np.argsort(curr_cl_data)].index))
    markers = pd.DataFrame(np.transpose(markers_desc), columns=[f"{cluster}_{i}" for i in range(np.max(nr_clust) + 1)])

    return markers
