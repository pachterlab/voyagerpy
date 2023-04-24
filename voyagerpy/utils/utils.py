#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

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

    scale_key = None
    if is_lowres(adata) and res in [None, "lowres", "lo"]:
        scale_key = "tissue_lowres_scalef"
    if is_highres(adata) and res in [None, "hires", "hi"]:
        scale_key = "tissue_hires_scalef"

    if scale_key is None:
        raise ValueError("Invalid resolution. Make sure the correct image is loaded.")

    return adata.uns["spatial"]["scale"][scale_key]


def add_per_gene_qcmetrics(adata: AnnData, subsets: Dict[str, np.ndarray], force: bool = False) -> None:
    if "sum" not in adata.var.keys() or force:
        adata.var["sum"] = adata.X.sum(axis=0).T  # type: ignore

    if "detected" not in adata.var.keys() or force:
        adata.var["detected"] = np.diff(adata.X.tocsc().indptr)  # type: ignore

    for key, subset in subsets.items():
        sum_key = f"subsets_{key}_sum"
        detected_key = f"subsets_{key}_detected"
        percent_key = f"subsets_{key}_percent"

        ss = adata[subset, :].X

        if sum_key not in adata.var.keys() or force:
            adata.var[sum_key] = ss.sum(axis=0).T  # type: ignore

        if detected_key not in adata.var.keys() or force:
            adata.var[detected_key] = np.diff(ss.tocsc().indptr)  # type: ignore

        if percent_key not in adata.var.keys() or force:
            adata.var[percent_key] = adata.var[sum_key] / adata.var["sum"] * 100


def add_per_cell_qcmetrics(adata: AnnData, subsets: Dict[str, np.ndarray], force: bool = False) -> None:
    if "sum" not in adata.obs.keys() or force:
        adata.obs["sum"] = adata.X.sum(axis=1)  # type: ignore

    if "detected" not in adata.obs.keys() or force:
        adata.obs["detected"] = np.diff(adata.X.tocsr().indptr)  # type: ignore

    for key, subset in subsets.items():
        sum_key = f"subsets_{key}_sum"
        detected_key = f"subsets_{key}_detected"
        percent_key = f"subsets_{key}_percent"

        subset_X = adata[:, subset].X
        if sum_key not in adata.obs.keys() or force:
            adata.obs[sum_key] = subset_X.sum(axis=1)  # type: ignore

        if detected_key not in adata.obs.keys() or force:
            adata.obs[detected_key] = np.diff(subset_X.tocsr().indptr)  # type: ignore

        if percent_key not in adata.obs.keys() or force:
            adata.obs[percent_key] = adata.obs[sum_key] / adata.obs["sum"] * 100


def log_norm_counts(
    adata: AnnData,
    layer: Optional[str] = None,
    inplace: bool = True,
    base: Optional[int] = 2,
    pseudocount: int = 1,
):
    # Roughly equivalent to:
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

    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X

    return adata


def scale(
    X,
    center=True,
    unit_variance: bool = True,
    center_before_scale: bool = True,
    ddof: int = 1,
):
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


def kurtosis(x, method: str = "moments"):
    if method != "moments":
        raise NotImplementedError('Only method="moments" is currently implemented')

    n = x.size
    x_bar = x.mean()
    # From asbio::kurt in R:
    # methods of moments kurtosis is
    #   m_4 / m_2^2  with m_j = sum((x-x_mean)**j)/n

    m_2 = np.square(x - x_bar).mean()
    m_4 = np.power(x - x_bar, 4).mean()

    return m_4 / m_2**2


def listify(
    x: Union[None, int, str, Iterable[str], Iterable[int]],
    size: Optional[int] = None,
) -> List[Any]:
    """Converts a string or an iterable of strings to a list of strings.

    Parameters
    ----------
    x : Union[str, Iterable[str]]
        The string or iterable to convert.

    Returns
    -------
    List[str]
        The list of strings.
    """
    nontype = type(None)
    size = size if size is not None else 1
    return [x] * size if isinstance(x, (int, float, str, nontype)) else list(x)
