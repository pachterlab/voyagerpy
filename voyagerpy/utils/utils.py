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
    """Determine whether the image data is high resolution.

    This function returns `True` iff the image data contains high resolution.

    Parameters
    ----------
    adata : AnnData
        Annoted data matrix.

    Returns
    -------
    bool
        `True` if the image data contains a high resolution image, `False` if image data is low resolution and not high resolution.

    Raises
    ------
    ValueError
        If the image data is neither high resolution nor low resolution.
    """
    if "hires" in adata.uns["spatial"]["img"]:
        return True
    if "lowres" in adata.uns["spatial"]["img"]:
        return False
    raise ValueError("Cannot find image data in .uns['spatial']")


def is_lowres(adata: AnnData) -> bool:
    """Determines whether the image data is low resolution.

    Parameters
    ----------
    adata : AnnData
        Annontated data matrix.

    Returns
    -------
    bool
        `True` if the image data contains a low resolution image, `False` if data is high resolution and not low resolution.

    Raises
    ------
    ValueError
        If the image data is neither high resolution nor low resolution.

    See also
    --------
    :py:func:`is_highres`
    """
    if "lowres" in adata.uns["spatial"]["img"]:
        return True
    if "hires" in adata.uns["spatial"]["img"]:
        return False
    raise ValueError("Cannot find image data in .uns['spatial']")


def make_unique(items: List) -> List:
    """Stably remove duplicates from a list.

    Parameters
    ----------
    items : List
        The list to remove duplicates from.

    Returns
    -------
    List
        The list with duplicates removed in the same order as the input.
    """
    items = items[:]
    for i in range(len(items) - 1, -1, -1):
        if items.count(items[i]) > 1:
            items.pop(i)
    return items


def get_scale(adata: AnnData, res: Optional[str] = None) -> float:
    """Get the scale of the image data.

    This function returns the scale of the requested image data if it exists.
    If `res` is `None`, the scale of the low resolution is return if it exists, otherwise it returns the scale of the high resolution image if it exists.
    If `res` is `"hi"` or `"hires"` and the object contains a high resolution image, the scale of the high resolution image data is returned.
    If `res` is `"lo"` or `"lowres"` and the object contains a low resolution image, the scale of the low resolution image data is returned.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    res : {None, "hi", "hires", "lo", "lowres"}, optional
        The resolution to get the scale for, by default None

    Returns
    -------
    float
        The full-resolution to requested-resolution scale factor.

    Raises
    ------
    ValueError
        If the requested resolution is not one of `None`, `"hi"`, `"hires"`, `"lo"`, or `"lowres"`.
    ValueError
        If the requested resolution is not found within the image data
    KeyError
        If the requested resolution is found within the image data, but the scale factor is not found.
    """
    if res not in [None, "hi", "hires", "lo", "lowres"]:
        raise ValueError(f"Unrecognized value {res} for res.")

    scale_dict = adata.uns["spatial"].get("scale", {})
    scale_key = None

    if is_lowres(adata) and res in [None, "lowres", "lo"]:
        scale_key = "tissue_lowres_scalef"
    elif is_highres(adata) and res in [None, "hires", "hi"]:
        scale_key = "tissue_hires_scalef"

    if scale_key is None:
        raise ValueError("Invalid resolution. Make sure the correct image is loaded.")
    elif scale_key not in scale_dict:
        raise KeyError(f"Could not find scale factor {scale_key} for {res}")

    return scale_dict[scale_key]


def add_per_gene_qcmetrics(adata: AnnData, subsets: Dict[str, np.ndarray], force: bool = False) -> None:
    """Add per-gene QC metrics to the AnnData object.

    This function computes the metrics sum and detected for each gene in the AnnData object.
    The metrics are computed for the entire dataset and for each subset in `subsets`, in addition to
    the percentage (:math:`\in [0, 100]`) of counts where the subset evaluates to `True`.

    The metric `*sum` is the number of total counts for each gene.

    The metric `*detected` is the number of cells in which each gene is detected.


    Parameters
    ----------
    adata : AnnData
        _description_
    subsets : Dict[str, np.ndarray]
        A dictionary of subsets to compute metrics for.
        The keys are the names of the subsets, and the values are boolean arrays of shape `(n_cells,)` where `True` indicates that the cell is in the subset.
    force : bool, optional
        If `True`, compute all metrics. If `False`, computed metrics will not be recomputed, by default False

    Returns
    -------
    None
        The keys `"sum"` and `"detected"` are added to `adata.var` if they do not already exist.
        For each subset `subset`, the keys
        `f"subsets_{subset}_sum"`, `f"subsets_{subset}_detected"`, and `f"subsets_{subset}_percent"` are added to `adata.var` if they do not already exist.
    """
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
    """Add per-cell QC metrics to the AnnData object.

    This function computes the metrics sum and detected for each cell in the AnnData object.
    The metrics are computed for the entire dataset and for each subset in `subsets`, in addition to
    the percentage (:math:`\in [0, 100]`) of counts where the subset evaluates to `True`.

    The metric `*sum` is the number of total counts for each cell.
    The metric `*detected` is the number of genes detected in each cell.

    Parameters
    ----------
    adata : AnnData
        Annotation data matrix.
    subsets : Dict[str, np.ndarray]
        A dictionary of subsets to compute metrics for.
        The keys are the names of the subsets, and the values are boolean arrays of shape `(n_cells,)` where `True` indicates that the cell is in the subset.
    force : bool, optional
        If `True`, compute all metrics. If `False`, computed metrics will not be recomputed, by default False

    Returns
    -------
    None
        The keys `"sum"` and `"detected"` are added to `adata.obs` if they do not already exist.
        For each subset `subset`, the keys
        `f"subsets_{subset}_sum"`, `f"subsets_{subset}_detected"`, and `f"subsets_{subset}_percent"` are added to `adata.obs` if they do not already exist.
    """
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
    adata: Union[np.ndarray, sp.csr_matrix, sp.csr_matrix, AnnData],
    layer: Optional[str] = None,
    inplace: bool = False,
    base: Union[None, int, bool] = 2,
    pseudocount: int = 1,
    zero_to_zero: bool = True,
) -> Union[np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    """Compute log-normalized counts.

    If ``adata`` is of type AnnData and ``layer`` is not ``None``, the layer is used instead of ``adata.X``.
    Otherwise, ``adata`` is assumed to be a sparse matrix or a dense matrix. All rows are normalized to sum to :math:`\\bar{N}`, then log-transformed,
    where :math:`\\bar{N}` is the mean of the total counts across all cells. If `zero_to_zero` is `True`, then zeros in the input matrix will map to zeros in the output matrix.
    If pseudocount is not 1 and zero_to_zero is False, will add pseudocount to all values before log-transforming. This makes the matrix dense in an intermediate step
    and may take a long time with large memory footprint.

    Parameters
    ----------
    adata : Union[np.ndarray, sp.csr_matrix, sp.csr_matrix, AnnData]
        Annotation data matrix, array, or sparse matrix to normalize.
    layer : Optional[str], optional
        If adata is an AnnData object, use this layer to normalize. If None, ``adata.X`` is used, by default None.
    inplace : bool, optional
        Normalize the object in-place, by default False
    base : Union[None, int, bool], optional
        The base of the logarithm to use, by default 2. If None, use natural logarithm. If False, do not log-transform.
    pseudocount : int, optional
        Pseudocounts to use. If 1, computes log1p, by default 1.
    zero_to_zero : bool, optional
        If True, zeros in the input matrix will map to zeros in the output, regardless of the pseudocount, by default True.

    Returns
    -------
    Union[np.ndarray, sp.csr_matrix, sp.csr_matrix]
        The log-normalized counts matrix or AnnData object. If an AnnData object is passed, the selected layer or adata.X is normalized.

    Raises
    ------
    TypeError
        If adata is not an AnnData object, array, or sparse matrix.
    """
    # Roughly equivalent to:
    # target_sum = adata.X.sum(axis=1).mean()
    # sc.pp.normalize_total(adata, target_sum=target_sum)
    # sc.pp.log1p(adata, base=base)

    if isinstance(adata, AnnData):
        X = adata.X if layer is None else adata.layers[layer]
    elif isinstance(adata, (np.ndarray, sp.csr_matrix, sp.csc_matrix)):
        X = adata
    else:
        raise TypeError("adata must be AnnData, np.ndarray, sp.csr_matrix, or scipy.sparse.csc_matrix")

    if not inplace:
        X = X.copy()

    cell_sums = np.ravel(X.sum(axis=1))
    cell_sums /= cell_sums.mean()

    # Normalize matrix in-place
    if sp.issparse(X):
        if sp.isspmatrix_csr(X):
            X.data /= np.repeat(cell_sums, np.diff(X.indptr))
        elif sp.isspmatrix_csc(X):
            X.data /= cell_sums[X.indices]

        # Add pseudocount - 1 since we actually use log1p
        if pseudocount != 1:
            if zero_to_zero:
                X.data += pseudocount - 1
            else:
                # Let's try to avoid this state
                cls_ = type(X)
                X = cls_(X.A + pseudocount - 1)
    else:
        X /= np.expand_dims(cell_sums, axis=1)

        if pseudocount != 1:
            if zero_to_zero:
                nonzero = np.where(X != 0)
                X[nonzero] += pseudocount - 1
            else:
                X += pseudocount - 1

    if base is False:
        return X

    # in-place log1p
    data, where = (X.data, True) if sp.issparse(X) else (X, X > 0)
    np.log1p(data, out=data, where=where)

    if base is not None and base is not True:
        data /= np.log(base)

    return X


def scale(
    X: Union[sp.spmatrix, np.ndarray, np.matrix],
    center: bool = True,
    unit_variance: bool = True,
    center_before_scale: bool = True,
    ddof: int = 1,
) -> np.ndarray:
    if sp.issparse(X):  # or isinstance(X, np.matrix):
        A = X.todense()  # type: ignore
    elif isinstance(X, np.ndarray):
        A = X.copy()
    else:
        raise TypeError("X must be of type np.ndarray or sp.spmatrix.")

    del X

    # if not isinstance(A, np.ndarray) or isinstance(A, np.matrix):
    #     raise RuntimeError("A must be a numpy array. This should not happen.")

    kwargs = dict(axis=0, keepdims=True)
    if isinstance(A, np.matrix):
        kwargs.pop("keepdims")

    if center and center_before_scale:
        A -= A.mean(**kwargs)

    if unit_variance:
        std = A.std(ddof=ddof, **kwargs)
        w = np.where(std < 1e-8)
        std[w] = 1
        A = np.divide(A, std)

    if center and not center_before_scale:
        A -= A.mean(axis=0)

    return A


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
