#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:08:42 2022

@author: sinant
"""


import functools
from copy import deepcopy
from math import ceil
from typing import (
    Any,
    Collection,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import geopandas as gpd
import numpy as np
from anndata import AnnData
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from pandas import options
from pandas.api.types import is_categorical_dtype
from scipy.stats import gaussian_kde

from voyagerpy import spatial as spt

options.mode.chained_assignment = None  # default='warn'

plt.style.use("ggplot")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "k"
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["axes.edgecolor"] = "#00000050"
plt.rcParams["axes.grid.which"] = "both"


def configure_violins(violins, cmap=None, edgecolor="#00000050", alpha=0.7):
    cmap = plt.get_cmap(cmap) if isinstance(cmap, (str, type(None))) else cmap
    for i, violin in enumerate(violins["bodies"]):
        violin.set_facecolor(cmap(i))
        violin.set_edgecolor(edgecolor)
        violin.set_alpha(alpha)


def simple_violin_plot(axs, adata, y, cmap=None):

    for feature, ax in zip(y, axs.flat):
        violins = ax.violinplot(adata.obs[feature], showmeans=False, showextrema=False, showmedians=False)
        configure_violins(violins, cmap)

        ax.set_ylabel(feature)
    return axs


def grouped_violin_plot(axs, adata, x, y, cmap=None):
    groups = adata.obs.groupby(x)[y].groups
    keys = sorted(groups.keys())
    grouped_data = [adata.obs.loc[groups[key]] for key in keys]

    for feature, ax in zip(y, axs.flat):
        dat = [group[feature] for group in grouped_data]
        violins = ax.violinplot(dat, showmeans=False, showextrema=False, showmedians=False)
        configure_violins(violins, cmap)

        ax.set_xticks(np.arange(len(keys)) + 1, labels=keys)
        ax.set_xlabel(x)
        ax.set_ylabel(feature)

    for i, key in enumerate(keys):
        ax.scatter([], [], label=key, color=cmap(i))
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=x, frameon=False)

    return axs


def plot_barcode_data(
    adata: AnnData,
    y: Union[str, Sequence[str]],
    x: Optional[str] = None,
    ncol: int = 3,
    cmap: Union[None, str, colors.ListedColormap, colors.LinearSegmentedColormap] = None,
    color_by: Optional[str] = None,
):
    if not isinstance(x, (str, type(None), int)):
        raise TypeError("x must be either None or str")

    if not isinstance(y, (list, tuple, np.ndarray)):
        y = [y]

    nplots = len(y)
    ncol = min(nplots, ncol)
    nrow = int(np.ceil(nplots / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(8, 4))

    if isinstance(cmap, (str, type(None))):
        cmap = plt.get_cmap(cmap)

    if nplots == 1:
        axs = np.array([axs])

    if x is None:
        simple_violin_plot(axs, adata, y, cmap)
    elif is_categorical_dtype(adata.obs[x]):
        grouped_violin_plot(axs, adata, x, y, cmap)
    else:
        for feature, ax in zip(y, axs.flat):
            x_dat = adata.obs[x]
            y_dat = adata.obs[feature]
            color = np.zeros_like(x_dat) if color_by is None else adata.obs[color_by].astype(int)
            scat = ax.scatter(x_dat, y_dat, c=color, vmax=cmap.N, alpha=0.5, s=8)
            ax.set_xlabel(x)
            ax.set_ylabel(feature)
        ax.legend(*scat.legend_elements(), bbox_to_anchor=(1.04, 0.5), loc="center left", title=color_by, frameon=False)

    fig.tight_layout()
    return axs


def plot_bin2d(
    data: Union[AnnData, "pd.DataFrame"],
    x: str,
    y: str,
    filt: Optional[str] = None,
    subset: Optional[str] = None,
    bins: int = 100,
    name_true: Optional[str] = None,
    name_false: Optional[str] = None,
    hex_plot: bool = False,
    binwidth: Optional[float] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:

    get_dataframe = lambda df: df.obs if x in df.obs and y in df.obs else df.var
    obs = get_dataframe(data) if isinstance(data, AnnData) else data

    #     I don't know how the range is computed in ggplot2
    #     r = ((-6.377067e-05,  4.846571e+04), (-1.079733e-05, 8.205973e+03))
    r = None

    plot_kwargs = dict(
        bins=bins,
        cmap="Blues",
        range=r,
    )

    figsize = kwargs.pop("figsize", (10, 7))
    plot_kwargs.update(kwargs)

    grid_kwargs = dict(visible=True, which="both", axis="both", color="k", linewidth=0.5, alpha=0.2)

    if hex_plot:
        renaming = [
            ("gridsize", "bins", bins),
            ("extent", "range", None),
            ("mincnt", "cmin", 1),
        ]
        for hex_name, hist_name, default in renaming:
            val = plot_kwargs.pop(hist_name, default)
            plot_kwargs.setdefault(hex_name, val)

        plot_kwargs.setdefault("edgecolor", "#8c8c8c")
        plot_kwargs.setdefault("linewidth", 0.2)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    plot_fun = ax.hexbin if hex_plot else ax.hist2d

    x = obs[x]
    y = obs[y]

    if subset is None:
        myfilt: Any = Ellipsis if filt is None else obs[filt].astype(bool)

        im = plot_fun(x[myfilt], y[myfilt], **plot_kwargs)  # type: ignore
        plt.colorbar(im[-1] if isinstance(im, tuple) else im)

    else:
        subset_name = subset
        name_true = name_true or subset_name
        name_false = name_false or f"!{subset_name}"

        subset_true = obs[subset].astype(bool)
        subset_false = (1 - subset_true).astype(bool)

        im1 = plot_fun(x[subset_false], y[subset_false], **plot_kwargs)

        plot_kwargs["cmap"] = "Reds"
        im2 = plot_fun(x[subset_true], y[subset_true], **plot_kwargs)

        plt.colorbar(im1[-1] if isinstance(im1, tuple) else im1, label=name_true)
        plt.colorbar(im2[-1] if isinstance(im2, tuple) else im2, label=name_false)

    ax.grid(**grid_kwargs)
    ax.set_facecolor("w")
    return ax


def plot_features_bin2d(adata: AnnData, *args, **kwargs) -> Axes:
    return plot_bin2d(adata.var, *args, **kwargs)


def plot_barcodes_bin2d(adata: AnnData, *args, **kwargs) -> Axes:
    return plot_bin2d(adata.obs, *args, **kwargs)


def rcDecorator(context):
    def decorator(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            with plt.rc_context(context):
                return func(*args, **kwargs)

        return inner_wrapper

    return decorator


def nogrid(func):

    context = {"axes.grid": False}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(context):
            return func(*args, **kwargs)

    return wrapper


def plot_spatial_feature(
    adata: AnnData,
    features: Union[str, Sequence[str]],
    ncol: Optional[int] = None,
    barcode_geom: Optional[str] = None,
    annot_geom: Optional[str] = None,
    tissue: bool = True,
    colorbar: bool = False,
    cmap: Optional[str] = "Blues",
    categ_type: Union[str, Collection[str]] = {},
    geom_style: Optional[Dict] = {},
    annot_style: Optional[Dict] = {},
    alpha: float = 0.2,
    divergent: bool = False,
    color: Optional[str] = None,
    _ax: Optional[Axes] = None,
    legend: bool = True,
    plot: bool = True,
    subplot_kwds: Optional[Dict] = {},
    legend_kwds: Optional[Dict] = {},
    **kwds,
) -> Union[np.ndarray, Any]:

    if isinstance(features, list):
        feat_ls = features
    elif isinstance(features, str):
        feat_ls = [features]
    else:
        raise TypeError("features must be a string or a list of strings")

    # check input
    if ("geometry" not in adata.obs) or "geom" not in adata.uns["spatial"]:
        adata = spt.get_geom(adata)
    for i in feat_ls:
        if i not in adata.obs and i not in adata.var.index:
            raise ValueError(f"Cannot find {i!r} in adata.obs or gene names")
    # copy observation dataframe so we can edit it without changing the inputs
    obs = adata.obs

    # check if barcode geometry exists
    if barcode_geom is not None:
        if barcode_geom not in obs:
            raise ValueError(f"Cannot find {barcode_geom!r} data in adata.obs")

        # if barcode_geom is not spot polygons, change the default
        # geometry of the observation matrix, so we can plot it
        if barcode_geom != "spot_poly":
            obs.set_geometry(barcode_geom)

    # check if features are in rowdata

    # Check if too many subplots
    if len(feat_ls) > 6:
        raise ValueError("Too many features to plot, reduce the number of features")
    if ncol is not None:
        if ncol > 3:
            raise ValueError("Too many columns for subplots")
        if ncol > len(feat_ls):
            raise ValueError("Too many columns")
    # only work with spots in tissue
    if tissue:
        obs = obs[obs["in_tissue"] == 1]
    # use a divergent colormap
    if divergent:
        cmap = "Spectral"

    # create the subplots with right cols and rows
    if _ax is None:
        plt_nr = len(feat_ls)
        nrows = 1
        # ncols = ncol if ncol is not None else 1

        # defaults
        if ncol is None:
            if plt_nr < 4:
                ncols = plt_nr
            if plt_nr >= 4:
                nrows = 2
                ncols = 3

        else:
            ncols = ncol
            nrows = ceil(plt_nr / ncols)

        # if(subplot_kwds is None):
        #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10,10))

        if "figsize" in subplot_kwds:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **subplot_kwds)
        else:

            # rat = row /col
            if nrows >= 2 and ncols == 3:
                _figsize = (10, 7)

            else:
                _figsize = (10, 10)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=_figsize, **subplot_kwds)
            # last axis not used
            if (ncols * nrows) - 1 == len(feat_ls):
                axs[-1, -1].axis("off")
        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

        # plt.subplots_adjust(wspace = 1/ncols +  0.2)
    else:
        ncols = 1
        nrows = 1
        axs = _ax
    # iterate over features to plot
    x = 0
    y = 0

    for i in range(len(feat_ls)):
        legend_kwds_ = deepcopy(legend_kwds)
        _legend = legend
        if tissue:

            # if gene value
            if feat_ls[i] in adata.var.index:
                # col = adata.var[features]
                col = adata[adata.obs["in_tissue"] == 1, feat_ls[i]].X.todense().reshape((adata[adata.obs["in_tissue"] == 1, :].shape[0])).T

                col = np.array(col.ravel()).T
                obs[feat_ls[i]] = col
            if feat_ls[i] in obs.columns:
                # feat = features
                pass
        else:

            if feat_ls[i] in adata.var.index:
                # col = adata.var[features]
                col = adata[:, feat_ls[i]].X.todense().reshape((adata.shape[0])).T
                obs[feat_ls[i]] = col
            if feat_ls[i] in obs.columns:
                pass

        if ncols > 1 and nrows > 1:
            ax = axs[x, y]
        if ncols == 1 and nrows > 1:
            ax = axs[x]
        if nrows == 1 and ncols > 1:
            ax = axs[y]
        if ncols == 1 and nrows == 1:
            ax = axs

        if feat_ls[i] in adata.var.index or adata.obs[feat_ls[i]].dtype != "category":
            legend_kwds_.setdefault("label", feat_ls[i])
            legend_kwds_.setdefault("orientation", "vertical")
            legend_kwds_.setdefault("shrink", 0.3)
        else:
            #  colorbar for discrete categories if pandas column is categorical
            _legend = False
            add_colorbar_discrete(
                ax, fig, cmap, feat_ls[i], adata.obs[feat_ls[i]].unique().shape[0], list(adata.obs[feat_ls[i]].cat.categories)
            )

        if color is not None:
            cmap = None

        obs.plot(
            feat_ls[i],
            ax=ax,
            color=color,
            legend=_legend,
            cmap=cmap,
            legend_kwds=legend_kwds_,
            **geom_style,
            **kwds,
        )
        if annot_geom is not None:
            if annot_geom in adata.uns["spatial"]["geom"]:

                # check annot_style is dict with correct values
                plg = adata.uns["spatial"]["geom"][annot_geom]
                if len(annot_style) != 0:
                    gpd.GeoSeries(plg).plot(ax=ax, **annot_style, **kwds)
                else:
                    gpd.GeoSeries(plg).plot(color="blue", ax=ax, alpha=alpha, **kwds)
            else:
                raise ValueError(f"Cannot find {annot_geom!r} data in adata.uns['spatial']['geom']")

            pass

        y = y + 1
        if y >= ncols:
            y = 0
            x = x + 1
    # colorbar title
    if _ax is not None:
        fig = ax.get_figure()
    axs = fig.get_axes()
    for i in range(len(axs)):
        if axs[i].properties()["label"] == "<colorbar>" and axs[i].properties()["ylabel"] != "":
            axs[i].set_title(axs[i].properties()["ylabel"], ha="left")
            axs[i].set_ylabel("")

    return axs  # ,fig


def spatial_reduced_dim(
    adata: AnnData,
    dimred: str,
    ncomponents: Union[int, Sequence[int]],
    barcode_geom: Optional[str] = None,
    ncol: Optional[int] = None,
    annot_geom: Optional[str] = None,
    tissue: bool = True,
    cmap: Optional[str] = "Blues",
    geom_style: Optional[Dict] = {},
    annot_style: Optional[Dict] = {},
    alpha: float = 0.2,
    divergent: bool = False,
    color: Optional[str] = None,
    _ax: Optional[Axes] = None,
    legend: bool = True,
    subplot_kwds: Optional[Dict] = {},
    legend_kwds: Optional[Dict] = {},
    **kwds,
):
    if isinstance(ncomponents, list):
        dims = ncomponents
    elif isinstance(ncomponents, int):
        dims = list(range(ncomponents))
    else:
        raise TypeError("features must be a integer or a list of integers")
    dim_nr = len(dims)
    # check input
    if ("geometry" not in adata.obs) or "geom" not in adata.uns["spatial"]:
        adata = spt.get_geom(adata)

    if dimred not in adata.obsm:
        raise ValueError(f"Cannot find {dimred!r} in adata.obsm")

    # create df for dimension reduction

    ls = []
    for i in range(adata.obsm[dimred].shape[1]):
        ls.append(dimred + str(i))
    red_arr = gpd.GeoDataFrame(adata.obsm[dimred], columns=ls, index=adata.obs.index)
    # check if barcode geometry exists
    if barcode_geom is not None:
        if barcode_geom not in adata.obs:
            raise ValueError(f"Cannot find {barcode_geom!r} data in adata.obs")

        # if barcode_geom is not spot polygons, change the default
        # geometry of the observation matrix, so we can plot it
        if barcode_geom != "spot_poly":
            red_arr.set_geometry(adata.obs[barcode_geom], inplace=True)
        else:
            red_arr.set_geometry(adata.obs["spot_poly"], inplace=True)

    # check if features are in rowdata

    # Check if too many subplots
    if dim_nr > 6:
        raise ValueError("Too many components to plot, reduce the number of components")
    if ncol is not None:
        if ncol > 3:
            raise ValueError("Too many columns for subplots")
        if ncol > dim_nr:
            raise ValueError("Too many columns")
    # only work with spots in tissue
    if tissue:
        red_arr = red_arr[adata.obs["in_tissue"] == 1]
    # use a divergent colormap
    if divergent:
        cmap = "Spectral"

    # create the subplots with right cols and rows
    if _ax is None:
        plt_nr = dim_nr
        nrows = 1
        # ncols = ncol if ncol is not None else 1

        # defaults
        if ncol is None:
            if plt_nr < 4:
                ncols = plt_nr
            if plt_nr >= 4:
                nrows = 2
                ncols = 3

        else:
            ncols = ncol
            nrows = ceil(plt_nr / ncols)

        # if(subplot_kwds is None):
        #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10,10))

        if "figsize" in subplot_kwds:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **subplot_kwds)
        else:

            # rat = row /col
            if nrows >= 2 and ncols == 3:
                _figsize = (10, 7)

            else:
                _figsize = (10, 10)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=_figsize, **subplot_kwds)
            # last axis not used
            if (ncols * nrows) - 1 == dim_nr:
                axs[-1, -1].axis("off")
        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

        # plt.subplots_adjust(wspace = 1/ncols +  0.2)
    else:
        ncols = 1
        nrows = 1
        axs = _ax
    # iterate over features to plot
    x = 0
    y = 0

    for i in dims:
        legend_kwds_ = deepcopy(legend_kwds)
        _legend = legend

        if ncols > 1 and nrows > 1:
            ax = axs[x, y]
        if ncols == 1 and nrows > 1:
            ax = axs[x]
        if nrows == 1 and ncols > 1:
            ax = axs[y]
        if ncols == 1 and nrows == 1:
            ax = axs

        legend_kwds_.setdefault("label", red_arr.columns[i])
        legend_kwds_.setdefault("orientation", "vertical")
        legend_kwds_.setdefault("shrink", 0.3)

        if color is not None:
            cmap = None

        red_arr.plot(
            red_arr.columns[i],
            ax=ax,
            color=color,
            legend=_legend,
            cmap=cmap,
            legend_kwds=legend_kwds_,
            **geom_style,
            **kwds,
        )
        if annot_geom is not None:
            if annot_geom in adata.uns["spatial"]["geom"]:

                # check annot_style is dict with correct values
                plg = adata.uns["spatial"]["geom"][annot_geom]
                if len(annot_style) != 0:
                    gpd.GeoSeries(plg).plot(ax=ax, **annot_style, **kwds)
                else:
                    gpd.GeoSeries(plg).plot(color="blue", ax=ax, alpha=alpha, **kwds)
            else:
                raise ValueError(f"Cannot find {annot_geom!r} data in adata.uns['spatial']['geom']")

            pass

        y = y + 1
        if y >= ncols:
            y = 0
            x = x + 1
    # colorbar title
    if _ax is not None:
        fig = ax.get_figure()
    axs = fig.get_axes()
    for i in range(len(axs)):
        if axs[i].properties()["label"] == "<colorbar>" and axs[i].properties()["ylabel"] != "":
            axs[i].set_title(axs[i].properties()["ylabel"], ha="left")
            axs[i].set_ylabel("")

    return axs  # ,fig


def add_colorbar_discrete(ax, fig, cmap, cbar_title: str, cat_nr: int, cat_names: list) -> Colorbar:
    # catnr = adata.obs[feat_ls[i]].unique().shape[0]
    bounds = list(range(cat_nr + 1))
    norm = colors.BoundaryNorm(bounds, cm.get_cmap(cmap).N)
    ticks = list(range(cat_nr))
    ticks = [x + 0.5 for x in ticks]
    cbar = fig.colorbar(
        cm.ScalarMappable(cmap=cm.get_cmap(cmap), norm=norm),
        ax=ax,
        # extend="both",
        extendfrac="auto",
        ticks=ticks,
        spacing="uniform",
        orientation="vertical",
        drawedges=False,
        # label=catname,
        shrink=0.3,
    )
    # dd = list(adata.obs[feat_ls[i]].cat.categories)
    cc = cbar.ax.set_yticklabels(cat_names)
    cbar.ax.set_title(cbar_title)
    cbar.ax.grid(None, which="major")

    return cbar


def subplots_single_colorbar(nrow: int = 1, ncol: int = 1, **kwargs):
    fig_kwargs = {"layout": "tight"}
    fig_kwargs.update(kwargs)
    figsize = fig_kwargs.pop("figsize", None)
    if isinstance(figsize, tuple):
        figsize = (figsize[0] * (ncol + 1) / ncol, figsize[1])

    fig = plt.figure(figsize=figsize, **fig_kwargs)

    width_ratios = [1] * ncol + [0.6]
    spec = fig.add_gridspec(nrow, ncol + 1, width_ratios=width_ratios, height_ratios=[1] * nrow)
    axs = np.array([[fig.add_subplot(spec[row, col]) for col in range(ncol)] for row in range(nrow)])

    cax = fig.add_subplot(spec[:, -1])
    cax.set_frame_on(False)
    cax.grid(False)
    cax.set_xticks([])
    cax.set_yticks([])
    return fig, axs, cax


def plot_dim_loadings(
    adata: AnnData,
    dims: Sequence[int],
    ncol: int = 2,
    figsize: Union[Tuple[float, float], float, int] = 6,
    n_extremes: int = 5,
    show_symbol: bool = True,
):
    dat = adata.varm["PCs"][:, dims]

    ncol = min(ncol, len(dims))
    nrow = int(np.ceil(len(dims) / ncol))

    if isinstance(figsize, (int, float)):
        figsize = (nrow * figsize / ncol, figsize)

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i in range(len(dims), nrow * ncol):
        axs.flat[i].remove()

    for i, (ax, dim) in enumerate(zip(axs.flat, dims)):
        ax.set_title(f"PC{dim}")

        idx = np.argsort(dat[:, i])
        idx = np.hstack([idx[:n_extremes], idx[-n_extremes:]])
        genes = adata.var.index[idx]
        if not show_symbol:
            genes = adata.var.loc[genes, "gene_ids"]  # .values

        loadings = dat[idx, i]
        lines = [((0, row), (loading, row)) for row, loading in enumerate(loadings)]

        line_segments = LineCollection(lines, colors="k")
        ax.add_collection(line_segments)

        ax.scatter(loadings, genes, c="b")
        ax.axvline(0, linestyle="--", c="k")

        # Show ticks at the bottom of each column
        ax.tick_params("x", labelbottom=i + ncol >= len(dims))

    fig.supxlabel("Loading")
    fig.supylabel("Gene")
    fig.tight_layout()

    return axs


def elbow_plot(adata: AnnData, ndims: int = 20, reduction: str = "pca", ax: Optional[Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()
    var_ratio = adata.uns[reduction]["variance_ratio"][:ndims]
    ax.scatter(np.arange(var_ratio.size), var_ratio * 100, s=8, c="k")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xlabel("PC")
    return ax


@rcDecorator({"axes.edgecolor": "#00000050", "axes.grid.which": "both"})
def plot_pca(adata: AnnData, ndim: int = 5, cmap: str = "tab10", colorby: str = "cluster", figsize=None):

    data = adata.obsm["X_pca"]

    fig, ax, cax = subplots_single_colorbar(ndim, ndim, figsize=figsize)

    max_color = plt.get_cmap(cmap).N
    colors = adata.obs[colorby].astype(int)
    var_expl = np.round(adata.uns["pca"]["variance_ratio"] * 100).astype(int)

    scatter_kwargs = dict(
        c=colors,
        s=8,
        alpha=0.5,
        cmap=cmap,
        vmin=0,
        vmax=max_color,
    )

    for row in range(ndim):
        for col in range(ndim):
            if row != col:
                ax[row, col].scatter(*data[:, (col, row)].T, **scatter_kwargs)
            if row < ndim - 1:
                ax[row, col].set_xticklabels([])
            if col > 0:
                ax[row, col].set_yticklabels([])
            ax[row, col].set_frame_on(True)

        ax[0, row].set_xlabel(f"PC{row} ({var_expl[row]:d}%)")
        ax[0, row].xaxis.set_label_position("top")
        ax[row, -1].set_ylabel(f"PC{row} ({var_expl[row]:d}%)", rotation=270, labelpad=15)
        ax[row, -1].yaxis.set_label_position("right")

        density = gaussian_kde(data[:, row])
        xs = np.linspace(data[:, row].min(), data[:, row].max(), 200)
        ax[row, row].plot(xs, density(xs), c="k", linewidth=1)

    legend_elements = ax.flat[1].collections[0].legend_elements()
    cax.legend(*legend_elements, loc="center", title=colorby, frameon=False)

    fig.tight_layout(pad=0)
    return ax
