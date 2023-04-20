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
from math import exp
import statsmodels.api as sm
from scipy.optimize import least_squares
import scanpy as sc
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric._smoothers_lowess import lowess


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
def get_parametric_start(_means, _vars, left_n=100, left_prop=0.1, grid_length=10, b_grid_range=5, n_grid_max=10):
    o = np.sort(_means)
    n = _vars.shape[0]

    left_n = min(left_n, n * left_prop)
    keep = pd.Index(_means).get_indexer(o[: max(1, 100)])

    y = _vars[keep]

    x = _means[keep]
    lm = LinearRegression(fit_intercept=False)
    grad = lm.fit(x.reshape(-1, 1), y).coef_[0]

    b_grid_pts = 2 ** np.linspace(-b_grid_range, b_grid_range, n_grid_max)
    n_grid_pts = 2 ** np.linspace(0, n_grid_max, grid_length)
    hits = np.array([(x, y) for x in b_grid_pts for y in n_grid_pts], dtype=np.float64)

    possible_vals = np.apply_along_axis(lambda x: sum((_vars - (1.13 * x[0] * _means) / ((_means ** x[1]) + x[0] + 0.001)) ** 2), 1, hits)

    min_ind = np.argmin(possible_vals)
    b_start = np.log(hits[min_ind][0])
    n_start = np.log(max(1e-8, hits[min_ind][1] - 1))
    a_start = np.log(grad * hits[min_ind][0])

    return a_start, b_start, n_start


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


# define your function f with inputs x and parameters a, b, c
def f(x, a, b, n):
    return (exp(a) * x) / (x ** (1 + exp(n)) + exp(b))


# define the residual function with weights, which is the difference between the function and the data


def residual(params, _mean, _vars, _weight):
    a, b, n = params

    rss = (_vars - f(_mean, a, b, n)) * np.sqrt(_weight)

    return rss


def parametric_fit(_mean, _vars, _weight, start):
    x = _mean
    y = _vars
    w = _weight  # set weights for each point

    # set initial guess for the parameters
    # params0 = [1.28, 0.149, 1.16]
    params0 = np.array(start)
    # run the optimization
    res = least_squares(residual, params0, args=(x, y, w), ftol=1e-12, verbose=0, max_nfev=1200)  # ,method="dogbox")

    # get the optimized parameters
    a, b, n = res.x

    # print the optimized parameters
    # print("Optimized parameters: a={}, b={}, c={}".format(a, n, b))
    return a, b, n


def correct_logged_expectation(x, y, w, FUN):
    leftovers = y / FUN(x)
    med = weighted_median(leftovers, w)
    OUT = lambda x: FUN(x) * med

    std_dev = weighted_median(abs(leftovers / (med - 1)), w) * 1.4826


def weighted_median(x, w):
    if x.shape != w.shape or w is None:
        w = np.ones(x.shape[0])

    o = pd.Index(x).get_indexer(np.sort(x))

    x = x[o]
    w = w[o]
    # return np.cumsum(w)
    p = np.cumsum(w) / sum(w)

    n = np.where(p < 0.5)[0].shape[0]

    # return n
    # return np.where(p>0.5)[0]
    if p[n] < 0.5:
        return x[n]
    else:
        return (x[n] + x[n + 1]) / 2

    # return n


def fit_trend_var(gene_mean, gene_var, min_mean=0.1, parametric=True, _lowess=False, density_weights=True):
    is_okay = np.intersect1d(np.where(gene_var > 1.0e-8), np.where(gene_mean >= min_mean))
    v = gene_var[is_okay]
    m = gene_mean[is_okay]

    if density_weights:
        w = inverse_density_weights(m)
    else:
        w = np.ones(len(m))

    to_fit = np.log(v)

    left_edge = min(m)
    # PARAMFUN = lambda x: np.min(np.column_stack((np.ones(len(x)), x / left_edge)), axis=1)
    PARAMFUN = lambda x: np.minimum(x / left_edge, 1)

    if parametric:
        a_start, b_start, n_start = get_parametric_start(m, v)

        a, b, n = parametric_fit(m, v, w, [a_start, b_start, n_start])
        to_fit = to_fit - np.log(f(m, a, b, n))
        PARAMFUN = lambda x: f(x, a, b, n)

    if _lowess:
        idx = np.round(np.linspace(0, len(m) - 1, 200)).astype(int)
        ll = lowess(to_fit, exog=m, xvals=idx, resid_weights=w)
    else:
        UNSCALEDFUN = PARAMFUN

    correct_logged_expectation(m, v, w, UNSCALEDFUN)
    return ll


# def get_start_params():
# %%


# %%

# TODO get each gene variance and mean of log expression values


# %%
# adata.X = sparse.csr_matrix(np.array(pd.read_csv("data_from_r.csv", sep=" ", header=0, index_col=0)))
adata = sc.read_h5ad("adata")
gene_mean, gene_var = get_mean_var(adata.X, axis=0)
min_mean = 0.1

# %%
get_parametric_start(m, v)
is_okay = np.intersect1d(np.where(gene_var > 1.0e-8), np.where(gene_mean >= min_mean))
v = gene_var[is_okay]
m = gene_mean[is_okay]
# %%

w = inverse_density_weights(m)


# %%

# left_n = 100
# left_prop = 0.1
# grid_length = 10
# b_grid_range = 5
# n_grid_max = 10


# PARAMFUN = lambda x: Aest * x / (x ^ Nest + Best)
# to_fit = to_fit - log(PARAMFUN(m))
# %%


# initial guess x


# %%


# define your data, x, y, and weights w


# %%


def f_corr2(x, a, n, b):
    return (exp(a) * x) / (x ** (1 + exp(n)) + exp(b))


# %%
ans = f_corr2(m, 2.4099, 0.5991, 2.62208)
v0 = gene_var[is_okay]
m0 = gene_mean[is_okay]
w0 = inverse_density_weights(m0)

idx = np.round(np.linspace(0, len(m0) - 1, 200)).astype(int)
# %%
parametric_fit(m, v, w, [1.28, 1.15, 0.14])
a1, a2, a3 = get_parametric_start(m, v)
fitted = fit_trend_var(v, m)
out, resid_weights = fit_trend_var(v, m)
# %%

weighted_median(m0, w0)
