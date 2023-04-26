#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from anndata import AnnData
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from scipy.stats import mannwhitneyu
from scipy import sparse
from typing import Union


def get_p_vals(
    adata: AnnData, clust1: str, clust2=None, test: str = "mw", alternative: str = "greater", skip=None, clust_str: str = "cluster"
):
    """Run a given statistical test for genes in a given anndata object,between cells in clust1 and clust2.

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
    # data_is_AnnData = isinstance(data, AnnData)
    # if data_is_AnnData:

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
        mw = mannwhitneyu(X_cl, X_comp, alternative=alternative)
        arr = mw[1]
    else:
        raise NotImplementedError(f"Test '{test}' is not implemented")

    if skip is not None:
        arr_final[it_values] = arr

    else:
        arr_final = arr

    return arr_final


def get_statistics(adata: AnnData, clust1: str, clust2=None, test: str = "mw", alternative: str = "greater", clust_str: str = "cluster"):
    """\
    Get summary statistics for genes.

    Similar to get_p_vals function,
    except this function returns additional info on effect size and the test statistic.


    Parameters
    ----------
    adata : Anndata object
        Anndata object with clustering having been defined in adata.obs.
    clust1 : String
        String of a single given cluster to identify rows(cells) in the cluster in adata.obs.
    clust2 : String, optional
        Group to test per gene vs clust1, if none then a clust1 vs rest test is performed. The default is None.
    test : String, optional
        The given statistical test to perform, per gene. The default is "mw".
    alternative : String, optional
        Direction/type of test. The default is "greater".
    clust_str : str, optional
        The string to identify the cluster column in adata.obs. The default is "cluster".

    Raises
    ------
    NotImplementedError
        In future more tests will be implemented,
        for now raise and error if a test is called that does not exist.

    Returns
    -------
    p_vals : np.ndarray
        p values for the test, each row is a different gene.
    test_stat : np.ndarray
        test statistic for each test.
    effect_size : np.ndarray
        The effect size for each test. In the case of the Mann-Whitney test it is defined as the AUC.
    """
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    cl_index = adata.obs.index.get_indexer(adata.obs.loc[adata.obs[clust_str] == clust1].index)
    if clust2 is None:
        comp_index = adata.obs.index.get_indexer(adata.obs.loc[adata.obs[clust_str] != clust1].index)

    else:
        comp_index = adata.obs.index.get_indexer(adata.obs.loc[adata.obs[clust_str] == clust2].index)
    X_cl = adata.X.todense()[cl_index, :]
    X_comp = adata.X.todense()[comp_index, :]

    if test == "mw":
        mw = mannwhitneyu(X_cl, X_comp, alternative=alternative)
        test_stat = mw[0]
        p_vals = mw[1]
        effect_size = test_stat / (X_cl.shape[0] * X_comp.shape[0])

    else:
        raise NotImplementedError(f"Test '{test}' is not implemented")

    return p_vals, test_stat, effect_size


def get_p_clusters(
    adata: AnnData, clust: Union[str, int], skip_precalc: bool = False, pval_type: str = "all", cluster: str = "cluster"
) -> Union[DataFrame, ndarray]:
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
        arr = get_p_vals(adata, str(clust), str(i), test="mw", skip=skip_index, clust_str=cluster)
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


def get_stat_clusters(adata: AnnData, clust: Union[str, int], cluster: str = "cluster") -> dict:
    """\
    Get differential expression statistics on genes for a given cluster, against all other clusters.

    Parameters
    ----------
    adata : AnnData
        AnnData object with defined clusters in adata.obs.
    clust : Union[str, int]
        Perform tests on genes, expressed in cells in this cluster.
    cluster : str, optional
        String to identify cluster column in adata.obs. The default is "cluster".

    Returns
    -------
    dict
        Dictionary with 3 entries, p values, test statistics and effect sizes,
        defined for each gene and each cluster paired with the one in the clust parameter.

    """
    pvals = []
    test_stat = []
    es = []
    colnames = []
    for i in np.array(adata.obs[cluster].cat.categories):
        if str(clust) == str(i):
            continue
        p_arr, mw_arr, es_arr = get_statistics(adata, str(clust), str(i), test="mw", clust_str=cluster)
        pvals.append(p_arr)
        test_stat.append(mw_arr)
        es.append(es_arr)
        colnames.append("clust" + str(clust) + "vs" + str(i))
    val_dict = {"p_values": pvals, "test_statistic": test_stat, "effect_size": es}
    for k, v in val_dict.items():
        temp_np = np.array(v)
        temp_np = np.transpose(temp_np.reshape((temp_np.shape[0], temp_np.shape[1])))
        temp_df = pd.DataFrame(temp_np, index=adata.var.index, columns=colnames)
        val_dict[k] = temp_df
    return val_dict


def get_marker_genes(adata: AnnData, hvg: bool = False, hvg_string: str = "highly_variable", cluster: str = "cluster") -> DataFrame:
    """\
    Find marker genes for each cluster.

    Similar to find_markers but returns only,
    top marker genes for each cluster, with no additional information.

    Parameters
    ----------
    adata : AnnData
        AnnData object with defined clustering in adata.obs.
    hvg : bool, optional
        Whether to only test highly variable genes. The default is False.
    hvg_string : str, optional
        String to identify highly variable genes in adata.var. The default is "highly_variable".
    cluster : str, optional
        String to identify cluster column in adata.obs. The default is "cluster".

    Returns
    -------
    DataFrame
        DataFrame with each column corresponding to differentially
        expressed genes for each cluster in descending order.

    """
    if cluster not in adata.obs:
        Exception("There need to be clusters defined for this anndata object")
    nr_clust = adata.obs[cluster].cat.codes
    markers_desc = []
    adata_obj = adata if not hvg else adata[:, adata.var[hvg_string]].copy()
    for i in np.array(adata.obs[cluster].cat.categories):
        curr_cl_data = get_p_clusters(adata_obj, i, cluster=cluster)
        markers_desc.append(np.array(curr_cl_data[np.argsort(curr_cl_data)].index))
    markers = pd.DataFrame(np.transpose(markers_desc), columns=[f"{cluster}_{i}" for i in range(np.max(nr_clust) + 1)])

    return markers


def find_markers(adata: AnnData, hvg: bool = False, hvg_string: str = "highly_variable", cluster: str = "cluster") -> dict:
    """\
    Return summary statistics on differentially expressed genes by groups defined in cluster.

    This function is based on findMarkers in scran.

    Parameters
    ----------
    adata : AnnData
        AnnData object with defined clustering in adata.obs.
    hvg : bool, optional
        Whether to only test highly variable genes. The default is False.
    hvg_string : str, optional
        String to identify highly variable genes in adata.var. The default is "highly_variable".
    cluster : str, optional
        String to identify cluster column in adata.obs. The default is "cluster".

    Returns
    -------
    dict
        Returns dictionary with summary statistics.
        P values, Benjamini-Hochberg False discovery rate,
        statistical measure of effect size, default for Mann-Whitney is AUC.

    """
    if cluster not in adata.obs:
        Exception("There need to be clusters defined for this anndata object")
    adata_obj = adata if not hvg else adata[:, adata.var[hvg_string]].copy()
    res = {}
    for i in np.array(adata.obs[cluster].cat.categories):
        stat_dict = get_stat_clusters(adata_obj, i, cluster=cluster)
        pval_max = stat_dict["p_values"].max(axis=1)
        index_of_maxs = stat_dict["p_values"].columns.get_indexer(stat_dict["p_values"].idxmax(axis=1))
        stat_dict["effect_size"]
        sort_order = np.argsort(pval_max)
        es = np.array(stat_dict["effect_size"])[np.arange(len(stat_dict["effect_size"])), index_of_maxs]
        res[f"{cluster}_{i}"] = pd.DataFrame(
            {"p_vals": pval_max[sort_order], "FDR": p_adjust_bh(pval_max)[sort_order], "summary_es": es[sort_order]},
            index=np.array(pval_max[sort_order].index, dtype=str),
        )

    return res


# taken from https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python/33532498#33532498
def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]
