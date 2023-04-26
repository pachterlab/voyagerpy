#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from math import exp

import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from scipy import sparse
from scipy.stats import iqr, norm
from sklearn.linear_model import LinearRegression

from statsmodels.stats.multitest import fdrcorrection
from statsmodels.nonparametric.kde import KDEUnivariate
from scanpy.preprocessing._utils import _get_mean_var


def get_parametric_start(_means, _vars, left_n=100, left_prop=0.1, grid_length=10, b_grid_range=5, n_grid_max=10):
    """Get initial guess of parameters a,b,n in for for nonlinear optimization of function f.

    Parameters
    ----------
    _means : array like of length n
        Array of means
    _vars : array like of length n
        Array of variances
    left_n : int, optional
        How many starting points to use to estimate a. The default is 100.
    left_prop : float, optional
        Proporiton of points to use to estimate a. The default is 0.1.
    grid_length : float, optional
        length of grid for . The default is 10.
    b_grid_range : float, optional
        range fro b parameter grid search. The default is 5.
    n_grid_max : float, optional
        max for n parameter grid search. The default is 10.

    Returns
    -------
    a_start : float
        Initial guess for parameter a for minimization
    b_start : float
        Initial guess for parameter b for minimization
    n_start : float
        Initial guess for parameter b for minimization

    """
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
    """\
    Calculate inverse density weights for data x.

    Parameters
    ----------
    x : ndarray
        Array of points, in this case means.
    adjust : int, optional
        DESCRIPTION. The default is 1.

    Raises
    ------
    Exception
        Fail if zero length of means.

    Returns
    -------
    w : ndarray
        Array of inverse density weights.

    """
    if len(x) == 0:
        raise Exception("Cannot parse nonzero array")
    kde = KDEUnivariate(x)
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
    """\
    Variance function.

    This function is used to describe the variances as a function of means.

    Parameters
    ----------
    x : ndarray
        Values to calculate
    a : float
        Parameter a.
    b : float
        Parameter b.
    n : float
        Parameter n.

    Returns
    -------
    TYPE
        return f(x).

    """
    return (exp(a) * x) / (x ** (1 + exp(n)) + exp(b))


# define the residual function with weights, which is the difference between the function and the data


def residual(params, _mean, _vars, _weight):
    """\
    Pseudo residual function for f.

    Implemented with weights

    Parameters
    ----------
    params : List
        Params for function f.
    _mean : ndarray
        Array of means.
    _vars : ndarray
        Array of variances.
    _weight : ndarray
        Array of weights.

    Returns
    -------
    rss : ndarray
        weighted residuals.

    """
    a, b, n = params

    rss = (_vars - f(_mean, a, b, n)) * np.sqrt(_weight)

    return rss


def parametric_fit(_mean, _vars, _weight, start):
    """\
    Fit function with parameters for given data.

    Parameters
    ----------
    _mean : ndarray
        Array of means.
    _vars : TYPE
        Array of variances.
    _weight : TYPE
        Array of weights.
    start : TYPE
        Initial start values of parameters a,b,n.

    Returns
    -------
    a : TYPE
        Best guess of a.
    b : TYPE
        Best guess of b.
    n : TYPE
        Best guess of n.

    """
    x = _mean
    y = _vars
    w = _weight  # set weights for each point

    # set initial guess for the parameters
    params0 = np.array(start)
    # run the optimization
    res = least_squares(residual, params0, args=(x, y, w), ftol=1e-12, verbose=0, max_nfev=1200)  # ,method="dogbox")

    # get the optimized parameters
    a, b, n = res.x

    return a, b, n


def f_predict(x, a, b, n):
    a0 = exp(a)
    b0 = exp(b)
    n0 = exp(n) + 1
    return (a0 * x) / (x**n0 + b0)


def correct_logged_expectation(x, y, w, FUN):
    leftovers = y / FUN(x)
    med = weighted_median(leftovers, w)
    OUT = lambda x: FUN(x) * med

    std_dev = weighted_median(abs((leftovers / med) - 1), w) * 1.4826

    return OUT, std_dev


def weighted_median(x, w):
    """\
    Return weighted median of x.

    Parameters
    ----------
    x : ndarray
        data points.
    w : ndarray
        weights.

    Returns
    -------
    float
        weighted median of x.

    """
    if x.shape != w.shape or w is None:
        w = np.ones(x.shape[0])

    # o = pd.Index(x).get_indexer(np.sort(x))
    o = np.argsort(x)

    x = x[o]
    w = w[o]
    p = np.cumsum(w) / sum(w)
    n = np.where(p < 0.5)[0].shape[0]
    if p[n] < 0.5:
        return x[n]
    else:
        return (x[n] + x[n + 1]) / 2

    # return n


def decompose_log_exprs(_means, _vars, fit_means, fit_vars, names=None):
    """\
    Decompose the variance into technical and biological.

    Parameters
    ----------
    _means : ndarray
        Means.
    _vars : ndarray
        Variances.
    fit_means : ndarray
        Kept for compatibility in R.
    fit_vars : ndarray
        Kept for compatibility in R.
    names : array-like, optional
        Gene names. The default is None.

    Returns
    -------
    output : DataFrame
        Returns dataframe with relevant columns.

    """
    fit, std_dev = fit_trend_var(fit_means, fit_vars)

    output = pd.DataFrame({"mean": _means, "total": _vars, "tech": fit(_means)}, index=names)
    output["bio"] = output["total"] - output["tech"]
    output["p_value"] = 1 - norm.cdf(output["bio"] / output["tech"], scale=std_dev)
    filt_out = output[~output["p_value"].isna()]
    # filt_out["FDR"] = fdrcorrection(filt_out["p_value"])[1]
    _, pval_corr = fdrcorrection(filt_out["p_value"])
    output.loc[filt_out.index, "FDR"] = pval_corr

    return output
    # 1 - norm().cdf(0.1)


def model_gene_var(adata, block=None, design=None, subset_row=None, subset_fit=None, gene_names=None):
    """\
    Return the modelled gene variance.

    Modelled on similar method in SCRAN package.
    Parameters
    ----------
    adata : AnnData
        DESCRIPTION.
    block : None, optional
        Compatibility with R. The default is None.
    design : None, optional
        Compatibility with R. The default is None.
    subset_row : ndarray, optional
        If only a subset of the rows are used. The default is None.
    subset_fit : bool, optional
        if subsetting should occur, yet to be implmented. The default is None.
    gene_names : ndarray, optional
        If gene names are provided. The default is None.

    Returns
    -------
    collected : DataFrame
        Information on modelled variance into biological and technical.

    """
    if sparse.issparse(adata):
        x_means, x_vars = _get_mean_var(adata, axis=0)
        names = gene_names
    else:
        x_means, x_vars = _get_mean_var(adata.X, axis=0)
        names = adata.var.index
    # decompose_log_exprs(x_means, x_vars)
    if subset_fit is None:
        fit_stats_means = x_means
        fit_stats_vars = x_vars
    else:
        return None

    collected = decompose_log_exprs(x_means, x_vars, fit_stats_means, fit_stats_vars, names=names)

    return collected


def fit_trend_var(gene_mean, gene_var, min_mean=0.1, parametric=True, lowess=False, density_weights=True):
    """\
    Fit the variance function to the data.

    Parameters
    ----------
    gene_mean : ndarray
        mean of each gene.
    gene_var : ndarray
        variances for each gene.
    min_mean : float, optional
        Minimum mean for a gene to be included in fitting the trend. The default is 0.1.
    parametric : bool, optional
        Whether to run a parametric trend fit. The default is True.
    lowess : bool, optional
        Whether to run a lowess trend fit. The default is False.
    density_weights : bool, optional
        Whether to include density weights in the fitting. The default is True.

    Returns
    -------
    fit : function
        fitted and corrected variance function.
    std_dev : TYPE
        standard deviation of fit.

    """
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
        # to_fit = to_fit - np.log(f(m, a, b, n))
        PARAMFUN = lambda x: f_predict(x, a, b, n)
        # return PARAMFUN

    if lowess:
        # idx = np.round(np.linspace(0, len(m) - 1, 200)).astype(int)
        # ll = lowess(to_fit, exog=m, xvals=idx, resid_weights=w)
        pass
    else:
        UNSCALEDFUN = PARAMFUN
    fit, std_dev = correct_logged_expectation(m, v, w, UNSCALEDFUN)
    return fit, std_dev


def get_top_hvgs(stats, n=2000, stat="bio", var_threshold=0):
    return np.array(stats.nlargest(n, "bio").index, dtype="str")
