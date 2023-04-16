#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:41:26 2023

@author: sinant
"""
import numpy as np
from scipy import sparse
from scipy.stats import iqr
import numba
import pandas as pd
from scipy.stats import gaussian_kde

import statsmodels.api as sm


# %%
@numba.njit(cache=True)
def sparse_mean_var_minor_axis(data, indices, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse matrix for the minor axis.
    Given arrays for a csr matrix, returns the means and variances for each
    column back.
    """
    non_zero = indices.shape[0]

    means = np.zeros(minor_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    counts = np.zeros(minor_len, dtype=np.int64)

    for i in range(non_zero):
        col_ind = indices[i]
        means[col_ind] += data[i]

    for i in range(minor_len):
        means[i] /= major_len

    for i in range(non_zero):
        col_ind = indices[i]
        diff = data[i] - means[col_ind]
        variances[col_ind] += diff * diff
        counts[col_ind] += 1

    for i in range(minor_len):
        variances[i] += (major_len - counts[i]) * means[i] ** 2
        variances[i] /= major_len

    return means, variances


@numba.njit(cache=True)
def sparse_mean_var_major_axis(data, indices, indptr, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse array for the major axis.
    Given arrays for a csr matrix, returns the means and variances for each
    row back.
    """
    means = np.zeros(major_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    for i in range(major_len):
        startptr = indptr[i]
        endptr = indptr[i + 1]
        counts = endptr - startptr

        for j in range(startptr, endptr):
            means[i] += data[j]
        means[i] /= minor_len

        for j in range(startptr, endptr):
            diff = data[j] - means[i]
            variances[i] += diff * diff

        variances[i] += (minor_len - counts) * means[i] ** 2
        variances[i] /= minor_len

    return means, variances


def sparse_mean_variance_axis(mtx: sparse.spmatrix, axis: int):
    """
    This code and internal functions are based on sklearns
    `sparsefuncs.mean_variance_axis`.
    Modifications:
    * allow deciding on the output type, which can increase accuracy when calculating the mean and variance of 32bit floats.
    * This doesn't currently implement support for null values, but could.
    * Uses numba not cython
    """
    assert axis in (0, 1)
    if isinstance(mtx, sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        raise ValueError("This function only works on sparse csr and csc matrices")
    if axis == ax_minor:
        return sparse_mean_var_major_axis(mtx.data, mtx.indices, mtx.indptr, *shape, np.float64)
    else:
        return sparse_mean_var_minor_axis(mtx.data, mtx.indices, *shape, np.float64)


# %%


def get_mean_var(X, *, axis=0):
    if sparse.issparse(X):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
        mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


# %%


def inverse_density_weights(x, adjust=1):
    kde = sm.nonparametric.KDEUnivariate(x)
    bw = 0.9 * min(np.std(x), iqr(x) / 1.34) * (len(x) ** (-1 / 5))
    # out = kde.fit(bw=bw,gridsize=512,clip=(min(m),max(m)),cut=0)
    out = kde.fit(bw=bw, gridsize=512, cut=0)
    kde_supp = out.support
    # kde_dens = kde.density
    kde_out = out.evaluate(kde_supp)

    w = 1 / np.interp(x, kde_supp, kde_out)
    w = w / np.mean(w)
    return w


# %%


def fit_trend_var(gene_var, _gene_mean, min_min=0.1, parametric=True, lowess=False, density_weights=True):
    is_okay = np.intersect1d(np.where(gene_var > 1.0e-8), np.where(gene_mean >= min_mean))
    v = gene_var[is_okay]
    m = gene_mean[is_okay]

    if density_weights:
        w = inverse_density_weights(m)
    else:
        w = np.ones(len(m))

    to_fit = np.log(v)
    left_edge = min(m)
    PARAMFUN = lambda x: np.min(np.column_stack((np.ones(len(x)), x / left_edge)), axis=1)

    ...
    ...
    ...


# def get_start_params():


# %%

# TODO get each gene variance and mean of log expression values

adata = derp
# %%
adata.X = sparse.csr_matrix(np.array(pd.read_csv("data_from_r.csv", sep=" ", header=0, index_col=0)))

gene_mean, gene_var = get_mean_var(adata.X, axis=0)


# %%


# %%

w = inverse_density_weights(m)


# %%

# left_n = 100
# left_prop = 0.1
# grid_length = 10
# b_grid_range = 5
# n_grid_max = 10
