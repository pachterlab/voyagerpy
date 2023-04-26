#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:08:42 2022

@author: sinant
"""

import functools
from copy import deepcopy
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from anndata import AnnData
from matplotlib import cm, colors, gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.path import Path as mplPath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame, Series, options
from pandas.api.types import is_categorical_dtype
from scipy import sparse as sp
from scipy.stats import gaussian_kde, linregress

from voyagerpy import spatial, utils

from .cmap_helper import DivergentNorm

# options.mode.chained_assignment = None  # default='warn'

plt.style.use("ggplot")
plt.rcParams["axes.edgecolor"] = "#00000050"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid.which"] = "both"
plt.rcParams["axes.grid.axis"] = "both"
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["figure.labelsize"] = 11
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["grid.alpha"] = 0.1
plt.rcParams["grid.color"] = "k"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["lines.markersize"] = 4
# plt.rcParams["figure.dpi"] = 100  # 100 is the default
# figure.constrained_layout.wspace 0.02
# figure.subplot.wspace 0.2
""" #TODO: 

ax title font_size 10
fig.supylabel fontsize 10 (figure.labelsize)
"""


def imshow(
    adata: AnnData,
    res: Optional[str] = None,
    ax: Optional[Axes] = None,
    tmp: bool = False,
    title: Optional[str] = None,
    **kwargs,
) -> Axes:
    """Show an image stored in adata.uns["spatial"].

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the image.
    res : Optional[str], optional
        The resolution of the image to show. If None, the first resolution found is used, by default None.
    ax : Optional[Axes], optional
        The axis showing the image. If None, a new axis is created, by default None.
    tmp : bool, optional
        If True, a temporary image is shown, by default False. This is useful to show an image with
        unapplied transformations.
    title : Optional[str], optional
        The title for the axis, by default None

    **kwargs: Parameters passed to matplotlib.axes.Axes.imshow.
    Returns
    -------
    Axes
        The axis showing the image.
    """

    img_key = "img_tmp" if tmp else "img"
    if res is None:
        res = list(adata.uns["spatial"][img_key].keys())[0]

    img = adata.uns["spatial"][img_key][res]

    if ax is None:
        _, axs = plt.subplots()
    else:
        axs = ax

    im_kwargs = dict(origin="lower")
    extent = kwargs.get("extent", None)

    axs.set_xticks([])
    axs.set_yticks([])
    im_kwargs.update(kwargs)
    if title is not None:
        axs.set_title(title)
    if extent is not None:
        left, right, top, bottom = extent
        y_rows, x_cols = img.shape[:2]
        extent = (
            max(0, left),
            min(x_cols, right),
            max(0, top),
            min(y_rows, bottom),
        )
        im_kwargs["extent"] = extent
        axs.imshow(img[top : bottom + 1, left : right + 1], **im_kwargs)
    else:
        axs.imshow(img, **im_kwargs)
    return axs


def configure_violins(
    violins,
    cmap: Optional[str] = None,
    edgecolor: str = "#00000050",
    alpha: float = 0.7,
    facecolor: Optional[str] = None,
) -> Tuple[str, float]:
    colormap = plt.get_cmap(cmap) if isinstance(cmap, (str, type(None))) else cmap
    for i_color, violin in enumerate(violins["bodies"]):
        violin.set_facecolor(facecolor or colormap(i_color))
        violin.set_edgecolor(edgecolor)
        violin.set_alpha(alpha)
    return edgecolor, alpha


def simple_violinplot(
    ax: Axes,
    df: DataFrame,
    y: Union[str, int],
    cmap: Optional[str] = None,
    scatter_points: bool = True,
    jitter: bool = False,
    **kwargs,
):
    violin_opts = dict(showmeans=False, showextrema=False, showmedians=False)
    kwargs.pop("legend", False)

    labels = {
        "x": kwargs.pop("x_label", None),
        "y": kwargs.pop("y_label", y),
    }

    violin_opts.update(kwargs)
    violins = ax.violinplot(df[y], **violin_opts)
    configure_violins(violins, cmap)

    if scatter_points:
        cols = df[y]
        x_vals = np.ones_like(cols)
        x_colors = np.ones_like(cols)
        if cols.ndim == 2:
            x_offsets = np.arange(cols.shape[1]).ravel()
            x_vals += x_offsets
            x_colors += x_offsets
            for violin, x_offset, y_col in zip(
                violins["bodies"], x_offsets, cols.values.T
            ):
                x_vals[:, x_offset] = jitter_points(
                    violin.get_paths()[0],
                    y_col,
                    x=x_offset,
                    jitter=jitter,
                )
        else:
            x_vals = jitter_points(
                violins["bodies"][0].get_paths()[0],
                y_vals=cols.values,
                jitter=jitter,
            )
        y_vals = cols.values

        scatter(
            x_vals,
            y_vals,
            color_by=x_colors,
            cmap=cmap,
            ax=ax,
            is_categorical=True,
            legend=False,
            alpha=0.7,
            labels=labels,
        )
    else:
        ax.set_ylabel(labels["y"])
        ax.set_xlabel(labels["x"])
    ax.set_xticks([])

    return ax


def jitter_points(
    violin: mplPath, y_vals: np.ndarray, x: int = 0, jitter: bool = False
):
    x_vert, y_vert = np.array(violin.cleaned().vertices).T
    at_max_y = np.argmax(y_vert) + 1
    x_vals = x + np.ones_like(y_vals, dtype=float)
    interpolated = np.interp(y_vals, y_vert[:at_max_y], x_vert[:at_max_y])

    if jitter:
        rand = np.random.uniform(size=x_vals.size) * 2 - 1
        # rand = np.linspace(0, 1, x_vals.size) * 2 - 1
        diff = x_vals - interpolated
        x_vals += diff * rand

    return x_vals


def grouped_violinplot(
    ax: Axes,
    df: DataFrame,
    x: str,
    y: str,
    cmap: Optional[str] = None,
    legend: bool = True,
    vert: bool = True,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    scatter_points: bool = True,
    jitter: bool = False,
    **kwargs,
):
    if not vert:
        x, y = y, x

    groups = df.groupby(x)[y].groups
    keys = sorted(groups.keys())
    labels = keys

    if df[x].dtype == "category" and keys == [0, 1]:
        labels = [False, True]

    grouped_data = [df.loc[groups[key], y] for key in keys]
    violin_opts = dict(
        showmeans=False,
        showextrema=False,
        showmedians=False,
        vert=vert,
        widths=0.8,
    )
    violins = ax.violinplot(grouped_data, **violin_opts)
    facecolor = "white" if scatter_points and jitter else None
    configure_violins(violins, cmap, facecolor=facecolor)

    set_ticks = ax.set_xticks if vert else ax.set_yticks
    set_ticks(np.arange(len(keys)) + 1, labels=labels)

    if not vert:
        x, y = y, x

    colormap = plt.get_cmap(cmap)
    if scatter_points:
        for i, (label, violin) in enumerate(zip(labels, violins["bodies"])):
            x_dat, y_dat = [], []
            if scatter_points:
                y_dat = grouped_data[i]
                x_dat = jitter_points(
                    violin.get_paths()[0],
                    y_dat,
                    x=i,
                    jitter=jitter,
                )

            scatter(
                x_dat,
                y_dat,
                ax=ax,
                label=label,
                color=colormap(i),
                **kwargs,
            )

    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)

    if not legend:
        return ax

    # ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=x, frameon=False)
    return ax


def plot_single_barcode_data(
    adata: AnnData,
    y: Union[int, str],
    x: Union[int, str, None] = None,
    obsm: Optional[str] = None,
    cmap: Optional[str] = None,
    ax: Optional[Axes] = None,
    legend: bool = False,
    color_by: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    **kwargs,
):
    if obsm is not None:
        obs = adata.obsm[obsm].copy()
        if color_by is not None and color_by not in obs:
            obs[color_by] = adata.obs[color_by].copy()
    else:
        obs = adata.obs
    if ax is None:
        fig, ax = plt.subplots()
    assert ax is not None

    if is_categorical_dtype(obs[y]):
        if x is None:
            raise NotImplementedError('"Rectangule" plots not implemented')
        elif is_categorical_dtype(obs[x]):
            raise NotImplementedError('"Rectangule" plots not implemented')
        else:
            # Create a horizontal plot, so we group by y instead of x
            ax = grouped_violinplot(
                ax,
                obs,
                x,
                y,
                cmap,
                legend=legend,
                x_label=x_label,
                y_label=y_label,
                vert=False,
            )
    else:
        if x is None:
            ax = simple_violinplot(
                ax, obs, y, cmap, x_label=x_label, y_label=y_label, **kwargs
            )

        elif is_categorical_dtype(obs[x]):
            ax = grouped_violinplot(
                ax,
                obs,
                x,
                y,
                cmap,
                legend=legend,
                x_label=x_label,
                y_label=y_label,
                **kwargs,
            )

        else:
            _scatter_kwargs = dict(
                alpha=0.5,
                labels=dict(x=x_label or x, y=y_label or y),
            )
            _scatter_kwargs.update(kwargs)
            ax = scatter(
                x, y, color_by=color_by, cmap=cmap, ax=ax, data=obs, **_scatter_kwargs
            )

    return ax


def configure_subplots(
    nplots: int, ncol: Optional[int] = 2, **kwargs
) -> Tuple[Figure, npt.NDArray[plt.Axes]]:
    ncol = min(ncol or 2, nplots)
    nrow = int(np.ceil(nplots / ncol))

    # TODO: We may need to increase the space if we draw legends, yticks, etc
    hspace = kwargs.pop("hspace", 0.2)
    wspace = kwargs.pop("wspace", None)

    default_figsize = (6, 6)
    plot_kwargs = {
        "figsize": default_figsize,
        "gridspec_kw": {"wspace": wspace, "hspace": hspace},
        "squeeze": False,
    }
    plot_kwargs.update(kwargs)

    # tight layout incompatible with gridspec_kw arguments wspace and hspace
    if plot_kwargs.get("layout", None) == "tight":
        plot_kwargs.pop("gridspec_kw", None)

    fig, axs = plt.subplots(nrow, ncol, **plot_kwargs)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    assert isinstance(axs, np.ndarray)
    for ax in axs.flat[nplots:]:
        ax.axis("off")

    return fig, axs


def plot_barcode_data(
    adata: AnnData,
    y: Union[str, Sequence[str]],
    x: Optional[str] = None,
    obsm: Optional[str] = None,
    ncol: Optional[int] = None,
    cmap: Union[
        None, str, colors.ListedColormap, colors.LinearSegmentedColormap
    ] = None,
    color_by: Optional[str] = None,
    sharex: Union[None, Literal["none", "all", "row", "col"], bool] = None,
    sharey: Union[None, str, bool] = None,
    x_label: Union[None, str, Sequence[str]] = None,
    y_label: Union[None, str, Sequence[str]] = None,
    rc_context: Optional[Dict[str, Any]] = None,
    ax: Optional[Axes] = None,
    subplot_kwargs: Optional[Dict] = None,
    **kwargs,
):
    x_features = utils.listify(x)
    y_features = utils.listify(y)

    if x_label is None:
        x_labels = x_features[:]
        if obsm is not None:
            x_labels = [f"{lab}_{obsm}" for lab in x_labels]
    else:
        x_labels = utils.listify(x_label)

    if y_label is None:
        y_labels = y_features[:]
        if obsm is not None:
            y_labels = [f"{lab}_{obsm}" for lab in y_labels]
    else:
        y_labels = utils.listify(y_label)

    del x, y, x_label, y_label

    nplots = len(y_features) * len(x_features)

    if ncol is None:
        ncol = len(x_features)

    if sharex is None:
        if len(y_features) and ncol == len(x_features):
            sharex = "col"
        else:
            sharex = False
    if sharey is None:
        if len(y_features) and ncol == len(x_features):
            sharey = "row"
        else:
            sharey = False

    if ax is not None:
        axs_arr = np.array([ax])
        fig = ax.get_figure()
    else:
        _subplot_kwargs = dict(
            sharex=sharex,
            sharey=sharey,
            figsize=kwargs.pop("figsize", None),
            layout=kwargs.pop("layout", "constrained"),
        )
        default_rc_context = {
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.right": False,
            "axes.grid": False,
        }
        _subplot_kwargs.update(subplot_kwargs or {})
        default_rc_context.update(rc_context or {})
        with plt.rc_context(default_rc_context):
            fig, axs_arr = configure_subplots(nplots, ncol, **_subplot_kwargs)

    feature_iterator = (
        (
            (y_lab, y_feat),
            (x_lab, x_feat),
        )
        for (y_lab, y_feat) in zip(y_labels, y_features)
        for (x_lab, x_feat) in zip(x_labels, x_features)
    )

    for i_plot, (y_feat, x_feat) in enumerate(feature_iterator):
        i_col = i_plot % ncol
        add_legend = i_col == ncol - 1
        x_label, x_feat = x_feat
        y_label, y_feat = y_feat

        axs_arr.flat[i_plot] = plot_single_barcode_data(
            adata,
            y=y_feat,
            x=x_feat,
            obsm=obsm,
            cmap=cmap,
            ax=axs_arr.flat[i_plot],
            legend=add_legend,
            color_by=color_by,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )

    return axs_arr


def plot_bin2d(
    data: Union[AnnData, DataFrame],
    x: str,
    y: str,
    filt: Optional[str] = None,
    subset: Optional[str] = None,
    bins: int = 101,
    name_true: Optional[str] = None,
    name_false: Optional[str] = None,
    hex_plot: bool = False,
    cmap: Optional[str] = "Blues7",
    cmap_true: Optional[str] = "Reds",
    binwidth: Optional[float] = None,
    ax: Optional[Axes] = None,
    **plot_kwargs,
) -> Axes:
    get_dataframe = lambda df: df.obs if x in df.obs and y in df.obs else df.var
    obs = get_dataframe(data) if isinstance(data, AnnData) else data

    _plot_kwargs = dict(
        bins=bins,
        cmap=cmap,
        cmin=1,
    )

    _legend_kwargs = dict(
        y=_plot_kwargs.pop("y_label", y),
        x=_plot_kwargs.pop("x_label", x),
    )

    figsize = plot_kwargs.pop("figsize", None)
    plot_kwargs.update(_plot_kwargs)

    if hex_plot:
        # hexbin has similar arguments as hist2d, but some names are different
        renaming = [
            ("gridsize", "bins", bins),
            ("mincnt", "cmin", 1),
        ]
        for hex_name, hist_name, default in renaming:
            val = plot_kwargs.pop(hist_name, default)
            plot_kwargs.setdefault(hex_name, val)

        plot_kwargs.setdefault("edgecolor", "#8c8c8c")
        plot_kwargs.setdefault("linewidth", 0.2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plot_fun = ax.hexbin if hex_plot else ax.hist2d
    x_dat = obs[x]
    y_dat = obs[y]
    del x, y

    if filt is None:
        selection = ...
    elif isinstance(filt, str):
        selection = obs[filt].astype(bool)
    else:
        selection = filt

    x_dat = x_dat[selection]
    y_dat = y_dat[selection]

    if subset is None:
        if filt is None:
            selection = ...
        elif isinstance(filt, str):
            selection = obs[filt].astype(bool)
        else:
            selection = filt

        im = plot_fun(x_dat, y_dat, **plot_kwargs)  # type: ignore
        cbar = fig.colorbar(im[-1] if isinstance(im, tuple) else im, label="count")
    else:
        subset_name = ""
        if isinstance(subset, str):
            subset_name = subset
            subset_true = obs[subset].astype(bool)
            subset_false = (1 - subset_true).astype(bool)
        else:
            subset_true = subset.astype(bool)
            subset_false = (1 - subset).astype(bool)

        name_true = name_true or subset_name or "True"
        name_false = name_false or (f"!{subset_name}" if subset_name else "False")

        x_min, x_max = x_dat.min(), x_dat.max()
        y_min, y_max = y_dat.min(), y_dat.max()

        if hex_plot:
            plot_kwargs["extent"] = (x_min, x_max, y_min, y_max)
        else:
            plot_kwargs["range"] = [[x_min, x_max], [y_min, y_max]]

        im_false = plot_fun(x_dat[subset_false], y_dat[subset_false], **plot_kwargs)

        plot_kwargs["cmap"] = cmap_true
        im_true = plot_fun(x_dat[subset_true], y_dat[subset_true], **plot_kwargs)

        plt.colorbar(
            im_false[-1] if isinstance(im_false, tuple) else im_false, label=name_false
        )
        plt.colorbar(
            im_true[-1] if isinstance(im_true, tuple) else im_true, label=name_true
        )

    ax.set_ylabel(_legend_kwargs["y"])
    ax.set_xlabel(_legend_kwargs["x"])

    ax.grid()
    return ax


def plot_expression(
    adata: AnnData,
    gene: Union[str, Sequence[str]],
    y: Optional[str] = None,
    obsm: Optional[str] = None,
    ax: Union[None, Axes, np.ndarray[Axes]] = None,
    **kwargs,
):
    _rc_params = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
    with plt.rc_context(_rc_params):
        if y is None:
            return plot_expression_violin(adata, gene, **kwargs)

        else:
            return plot_expression_scatter(adata, gene, y, obsm=obsm, ax=ax, **kwargs)


def plot_expression_scatter(
    adata: AnnData,
    gene: str,
    y: str,
    obsm: Optional[str] = None,
    ax: Union[None, Axes] = None,
    color_by: Optional[str] = None,
    layer: Optional[str] = None,
    show_symbol: bool = True,
    **kwargs,
):
    obs = adata.obs if obsm is None else adata.obsm[obsm]
    X = adata.X if layer is None else adata.layers[layer]

    if sp.issparse(X):
        X = X.toarray()
    elif isinstance(X, np.matrix):
        X = np.array(X)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    color_data = obs if color_by in obs else adata.obs

    if isinstance(gene, str):
        gene_idx = adata.var_names.get_loc(gene)
        x_data = X[:, gene_idx]
    else:
        x_data = gene
    if isinstance(y, str):
        y_data = obs[y]
        y_symbol = adata.var.at[y, "symbol"]
    else:
        y_data = y
        y_symbol = None

    ax = scatter(
        x_data,
        y_data,
        label=gene,
        color_by=color_by,
        data=color_data,
        ax=ax,
        **kwargs,
    )

    ax.set_title(y_symbol if show_symbol else gene)
    ax.set_xlabel("Expression" + f" ({layer})" if layer is not None else "")
    if y_symbol is not None:
        y_lab = y_symbol if show_symbol else gene
        ax.set_ylabel(y_lab + f" {obsm}" if obsm is not None else "")

    return ax


def plot_expression_violin(
    adata: AnnData,
    gene: Union[str, Sequence[str]],
    groupby: Optional[str] = None,
    ncol: Optional[int] = 2,
    show_symbol: bool = False,
    layer: Optional[str] = None,
    cmap: Optional[str] = None,
    subplot_kwargs: Optional[Dict] = None,
    **kwargs,
):
    # genes = [genes] if isinstance(genes, str) else genes[:]
    genes = utils.listify(gene)
    gene_ids = []

    secondary_column = adata.uns["config"]["secondary_var_names"]

    for genes in genes:
        if genes in adata.var[secondary_column].values:
            new_genes = adata.var.index[adata.var[secondary_column] == genes]
            gene_ids.extend(new_genes.tolist())
        else:
            assert genes in adata.var_names
            gene_ids.append(genes)

    X = (adata.X if layer is None else adata.layers[layer]).copy()

    obs = (adata.obs[[groupby]] if groupby is not None else adata.obs).copy()
    for gene_id in gene_ids:
        gene_idx = adata.var_names.get_loc(gene_id)
        counts = X[:, gene_idx]

        if sp.issparse(counts):
            counts = counts.A

        obs[gene_id] = counts

    _subplot_kwargs = dict(
        figsize=kwargs.pop("figsize", None),
        sharey=True,
        sharex=True,
        layout="constrained",
    )
    _subplot_kwargs.update(subplot_kwargs or {})

    nplots = len(gene_ids) if groupby is not None else 1
    fig, axs = configure_subplots(nplots, ncol, **_subplot_kwargs)

    if groupby is None:
        gene_ids = [gene_ids]

    for ax, gene_id in zip(axs.flat, gene_ids):
        if groupby is not None:
            grouped_violinplot(
                ax, obs, groupby, gene_id, legend=False, cmap=cmap, **kwargs
            )
            title = gene_id
            if show_symbol and secondary_column == "symbol":
                title = adata.var.at[gene_id, secondary_column]
            elif show_symbol:
                title = gene_id

            ax.set_title(title)
        else:
            simple_violinplot(ax, obs, gene_id, legend=False, cmap=cmap, **kwargs)
            labels = gene_id
            if show_symbol and secondary_column == "symbol":
                labels = adata.var.loc[gene_id, "symbol"]
            elif show_symbol:
                labels = gene_id
            ax.set_xticks(np.arange(len(gene_id)) + 1, labels=labels, rotation=60)

        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.supylabel("Expression" + f" ({layer})" if layer is not None else "")
    if groupby is not None:
        fig.supxlabel(groupby)

    return axs


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
    ncol: int = 2,
    barcode_geom: Optional[str] = None,
    annot_geom: Optional[str] = None,
    subset_barcodes: Union[None, slice, Sequence[str]] = None,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
    geom_style: Optional[Dict] = {},
    annot_style: Optional[Dict] = {},
    alpha: float = 0.2,
    divergent: bool = False,
    color: Optional[str] = None,
    _ax: Union[None, Axes, Iterable[Axes]] = None,
    legend: bool = True,
    cmap: Optional[str] = "Blues7",
    cat_cmap: Optional[str] = None,
    div_cmap: str = "roma",
    subplot_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    dimension: str = "barcode",
    image_kwargs: Optional[Dict] = None,
    figtitle: Optional[str] = None,
    feature_labels: Union[None, str, Sequence[Optional[str]]] = None,
    rc_context: Optional[Dict[str, Any]] = None,
    show_symbol: bool = True,
    **kwargs,
) -> Union[np.ndarray, Any]:
    feat_ls = utils.listify(features)

    assert_basic_spatial_features(adata, dimension, errors="raise")
    if obsm is not None and obsm not in adata.obsm:
        raise KeyError(f'Key "{obsm}" is not found in adata.obsm')

    # adata = adata.copy()

    secondary_gene_column = adata.uns["config"]["secondary_var_names"]
    added_features = []

    features_to_pop = []

    var_features = []
    labeled_features = []

    df = adata.obs if obsm is None else adata.obsm[obsm]
    df_repr = f"adata.obs" if obsm is None else f'adata.obsm["{obsm}"]'

    feature_labels = (
        feature_labels if feature_labels is not None else [None] * len(feat_ls)
    )
    feature_labels = (
        [feature_labels]
        if not isinstance(feature_labels, (tuple, list))
        else feature_labels[:]
    )

    for feature, label in zip(feat_ls, feature_labels):
        label = label if label is not None else feature
        is_gene = feature in adata.var_names
        if is_gene and show_symbol:
            label = adata.var.at[feature, secondary_gene_column]
        if feature in df:
            labeled_features.append((feature, label))
            continue
        if is_gene:
            labeled_features.append((feature, label))
            var_features.append(feature)
            continue

        # Add the indices of the feature found in the secondary_var_names column
        if feature in adata.var[secondary_gene_column].values:
            ids = adata.var[adata.var[secondary_gene_column] == feature].index
            added_features.extend(ids)
            var_features.extend(ids)

            labeled_features.extend([(idx, feature) for idx in ids])
            features_to_pop.append(feature)
            continue

        raise ValueError(f"Cannot find {feature!r} in {df_repr} or gene names")

    # Rename the features found via the secondary column
    feat_ls.extend(added_features)
    for feature in features_to_pop:
        feat_ls.remove(feature)

    # Remove duplicates
    for i_feature in range(len(feat_ls) - 1, -1, -1):
        feature = feat_ls[i_feature]
        if feat_ls.count(feature) > 1:
            feat_ls.pop(i_feature)

    del feat_ls
    # Select the spots to work with

    barcode_selection = subset_barcodes if subset_barcodes is not None else slice(None)
    gene_selection = (
        slice(None) if not var_features else utils.make_unique(var_features)
    )

    if dimension == "barcode":
        geo = adata.obsm["geometry"].loc[barcode_selection].copy()
        gene_selection = gene_selection if var_features else 0
    elif dimension == "gene":
        geo = adata.varm["geometry"].loc[gene_selection].copy()
    else:
        geo = adata.uns["spatial"]["geometry"][dimension].copy()

    adata = adata[barcode_selection, gene_selection].copy()
    obs = adata.obs if obsm is None else adata.obsm[obsm]

    if var_features:
        if layer is None:
            columns = adata[:, var_features].X.toarray()
        else:
            columns = adata[:, var_features].layers[layer].toarray()

        obs[var_features] = columns

    # check if barcode geometry exists
    if barcode_geom is not None:
        if barcode_geom not in geo:
            raise ValueError(f"Cannot find {barcode_geom!r} data in adata.obs")

        # if barcode_geom is not spot polygons, change the default
        # geometry of the observation matrix, so we can plot it
        if barcode_geom != geo.geometry.name:
            geo.set_geometry(barcode_geom)

    n_features = len(labeled_features)
    ncol = min(ncol, n_features)

    # use a divergent colormap
    if divergent:
        cmap = div_cmap

    _subplot_kwargs = dict(
        layout="constrained",
        figsize=kwargs.pop("figsize", None),
        sharex=True,
        sharey=True,
        squeeze=False,
        hspace=0.2 if legend else 0.05,
        wspace=0.1,
    )
    _subplot_kwargs.update(subplot_kwargs or {})

    # create the subplots with right cols and rows

    _rc_context = {
        "axes.grid": False,
        "figure.frameon": False,
        "axes.spines.bottom": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "xtick.bottom": False,
        "xtick.labelbottom": False,
        "ytick.left": False,
        "ytick.labelleft": False,
    }

    geom_is_poly = geo.geometry.geom_type[0] == "Polygon"

    # This is to match the vignettes.
    # visium_10x and nonspatial don't have the same look for this function.
    if not geom_is_poly:
        for key in _rc_context:
            _rc_context[key] = True

    _rc_context.update(rc_context or {})
    if _ax is None:
        with plt.rc_context(_rc_context):
            fig, axs = configure_subplots(
                nplots=n_features, ncol=ncol, **_subplot_kwargs
            )
        # plt.subplots_adjust(wspace = 1/ncols +  0.2)
    else:
        if isinstance(_ax, Axes):
            axs = np.array([_ax])
        elif not isinstance(_ax, np.ndarray):
            axs = np.array(_ax)
        else:
            axs = _ax

        fig = axs.flat[0].get_figure()

    kwargs.setdefault("s", 4)
    if geom_is_poly:
        kwargs.setdefault("markersize", kwargs.pop("s", None))

    # iterate over features to plot

    for _ax, (feature, label) in zip(axs.flat, labeled_features):
        legend_kwargs_ = deepcopy(legend_kwargs or {})
        _image_kwargs = deepcopy(image_kwargs or {})
        _legend = legend
        curr_cmap = cmap
        values = obs[feature]

        extra_kwargs = dict()
        if values.dtype != "category":
            curr_cmap = cmap
            vmax = None

            legend_kwargs_.setdefault("title", label)
            legend_kwargs_.setdefault("orientation", "vertical")

            if divergent:
                vmin = values.min()
                vmax = values.max()
                vcenter = 0
                extra_kwargs["norm"] = DivergentNorm(vmin, vmax, vcenter)

        else:
            #  colorbar for discrete categories if pandas column is categorical
            curr_cmap = cat_cmap
            vmax = cm.get_cmap(cat_cmap).N
            _legend = False

        if color is not None:
            curr_cmap = None

        if image_kwargs is not None:
            crop_img = _image_kwargs.pop("crop", False)
            extent = None
            if crop_img:
                x = geo.centroid.x
                y = geo.centroid.y
                pad = _image_kwargs.pop("pad", 5)
                x_min, y_min = np.maximum(
                    np.floor([x.min(), y.min()]).astype(int) - pad, 0
                )
                x_max, y_max = np.ceil([x.max(), y.max()]).astype(int) + pad
                extent = (x_min, x_max, y_min, y_max)
            _ax = imshow(adata, None, _ax, extent=extent)

        if geom_is_poly:
            legend_kwargs_.pop("title", None)

            if _legend:
                divider = make_axes_locatable(_ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                extra_kwargs["cax"] = cax
                cax.set_title(label)

            _ax = geo.plot(
                column=values,
                ax=_ax,
                color=color,
                legend=_legend,
                cmap=plt.get_cmap(curr_cmap),
                vmax=vmax,
                legend_kwds=legend_kwargs_,
                **extra_kwargs,
                **geom_style,
                **kwargs,
            )

            # This would only happen for categorical data
            if legend and not _legend:
                cmap_colors = plt.get_cmap(curr_cmap).colors
                legend_dict = {
                    lab: color
                    for lab, color in zip(sorted(np.unique(values)), cmap_colors)
                }
                for key, color in legend_dict.items():
                    _ax.scatter([], [], label=key, color=color)
                _ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.04, 0.5),
                    title=label,
                    frameon=False,
                )

        else:
            _ax = scatter(
                geo.geometry.x,
                geo.geometry.y,
                ax=_ax,
                color_by=feature,
                data=obs,
                legend_kwargs=legend_kwargs_,
                cmap=curr_cmap,
                **extra_kwargs,
                **kwargs,
            )

        if annot_geom is not None:
            if annot_geom in adata.uns["spatial"]["geom"]:
                # check annot_style is dict with correct values
                annot_kwargs = dict(ax=_ax, color="blue", alpha=alpha, **kwargs)
                annot_kwargs.update(annot_style or {})

                plg = adata.uns["spatial"]["geom"][annot_geom]
                gpd.GeoSeries(plg).plot(**annot_kwargs)
            else:
                raise ValueError(
                    f"Cannot find {annot_geom!r} data in adata.uns['spatial']['geom']"
                )

    if figtitle is not None:
        fig.suptitle(figtitle, x=0, ha="left", va="bottom")

    return axs


def assert_basic_spatial_features(
    adata, dimension="barcode", errors: str = "raise"
) -> Tuple[bool, str]:
    ret = True
    errors_to_raise = []
    get_geom_prompt = "Consider run voyagerpy.spatial.get_geom(adata) first."
    set_geom_prompt = "Consider running voyagerpy.spatial.set_geometry first"
    if dimension == "barcode":
        if "geometry" not in adata.obsm:
            error_msg = "geometry dataframe does not exist. " + set_geom_prompt
            if errors == "raise":
                raise KeyError(error_msg)
            return False, error_msg
        try:
            geom_name = adata.obsm["geometry"].geometry.name
        except AttributeError as ex:
            error_msg = ex.args[0]
            if errors == "raise":
                raise ex
            return False, error_msg
    # TODO: add checks for genes and annotgeom
    # TODO: deprecate
    return True, ""

    if not isinstance(adata.obs, gpd.GeoDataFrame):
        error_msg = "adata.obs must be a geopandas.GeoDataFrame. " + get_geom_prompt
        if errors == "raise":
            raise TypeError(error_msg)
        return False, error_msg

    if adata.obs.geometry.name not in adata.obs:
        error_msg = (
            f'"{adata.obs.geometry.name}" must be a column in adata.obs. '
            + get_geom_prompt
        )
        if errors == "raise":
            raise KeyError(error_msg)
        return False, error_msg
    if "geom" not in adata.uns["spatial"]:
        error_msg = '"geom" must be a key in adata.uns["spatial"]. ' + get_geom_prompt
        if errors == "raise":
            raise KeyError(error_msg)
        return False, error_msg

    return True, ""


def plot_local_result(
    adata: AnnData, obsm: str, features: Union[str, Sequence[str]], **kwargs
):
    if obsm not in adata.obsm:
        raise KeyError(f"`{obsm}` not found in adata.obsm.")

    kwargs.setdefault("figtitle", obsm.replace("_", " ").capitalize())

    rc_context = {
        "axes.grid": True,
        "figure.frameon": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.spines.left": True,
        "axes.spines.right": True,
    }

    axs = plot_spatial_feature(
        adata, features=features, obsm=obsm, rc_context=rc_context, **kwargs
    )
    return axs


def spatial_reduced_dim(
    adata: AnnData,
    dimred: str,
    ncomponents: Union[int, Sequence[int]],
    **kwargs,
):
    adata = adata.copy()
    if isinstance(ncomponents, (list, tuple, range)):
        dims = list(ncomponents)
    elif isinstance(ncomponents, int):
        dims = list(range(ncomponents))
    else:
        raise TypeError("features must be a integer or a sequence of integers")

    # check input
    assert_basic_spatial_features(adata, errors="raise")

    if dimred not in adata.obsm:
        raise ValueError(f"Cannot find {dimred!r} in adata.obsm")

    # create df for dimension reduction
    reductions = adata.obsm[dimred][:, dims]
    feat_names = [f"{dimred}{i}" for i in dims]
    adata.obs[feat_names] = reductions

    axs = plot_spatial_feature(
        adata=adata,
        features=feat_names,
        **kwargs,
    )

    fig = axs.flat[0].get_figure()
    fig.suptitle(dimred, x=0, ha="left", fontsize="xx-large", va="bottom")
    return axs


def add_colorbar_discrete(
    ax,
    fig,
    cmap,
    cbar_title: str,
    cat_names: list,
    scale: bool = False,
    title_kwargs: Optional[Dict] = None,
) -> Colorbar:
    # catnr = adata.obs[feat_ls[i]].unique().shape[0]

    colormap = cm.get_cmap(cmap)
    del cmap

    n_values = len(cat_names)
    n_colors = colormap.N if scale else n_values
    if not scale:
        colormap = colors.ListedColormap(colormap.colors[:n_colors])

    bounds = list(range(n_values + 1))
    n_bounds = cm.get_cmap(colormap).N if scale else n_values + 1
    norm = colors.BoundaryNorm(bounds, n_bounds)

    ticks = [x + 0.5 for x in range(n_values)]

    cbar = fig.colorbar(
        cm.ScalarMappable(cmap=colormap, norm=norm),
        ax=ax,
        extendfrac="auto",
        ticks=ticks,
        spacing="uniform",
        orientation="vertical",
        drawedges=True,
        shrink=0.3,
    )

    title_kwargs = title_kwargs or {}
    cbar.ax.set_title(cbar_title, **title_kwargs)
    cbar.ax.grid(False, "both")
    cbar.ax.set_yticklabels(cat_names)

    cbar.outline.set_visible(False)
    cbar.dividers.set_color("white")
    cbar.dividers.set_linewidth(5)

    return cbar


def subplots_single_colorbar(
    nplot: int = 1,
    ncol: int = 1,
    cax_width: float = 0.2,
    cax_space: float = 0.4,
    **kwargs,
):
    ncol = min(ncol or 2, nplot)
    nrow = int(np.ceil(nplot / ncol))

    if not kwargs.pop("legend", True):
        return *configure_subplots(nplot, ncol, **kwargs), None

    figsize = kwargs.get("figsize", None)

    if isinstance(figsize, tuple):
        figsize = (figsize[0] + cax_width + cax_space, figsize[1])
        kwargs["figsize"] = figsize

    wspace = kwargs.pop("wspace", 0.05)
    hspace = kwargs.pop("hspace", 0.05)

    fig = plt.figure(**kwargs)

    # Configure the gridspec size
    (total_width, _) = fig.get_size_inches()
    plot_width = total_width - cax_width - cax_space

    right = plot_width / total_width
    left = (plot_width + cax_space) / total_width

    has_layout = kwargs.get("layout", None) is not None
    if has_layout:
        ax_width = plot_width / ncol
        width_ratios = [ax_width] * ncol + [cax_width]
        main_spec = fig.add_gridspec(
            nrow, ncol + 1, wspace=wspace, hspace=hspace, width_ratios=width_ratios
        )
        cax_spec = main_spec[:, -1]
    else:
        gridspec_kw = dict(
            wspace=wspace,
            hspace=hspace,
            right=right,
            # left=0.0,
        )
        main_spec = fig.add_gridspec(nrow, ncol, **gridspec_kw)

    axs = np.array(
        [
            [
                fig.add_subplot(
                    main_spec[row, col],
                )
                for col in range(ncol)
            ]
            for row in range(nrow)
        ]
    )

    # Make the last column of the grid the colorbar
    if has_layout:
        cax = fig.add_subplot(main_spec[:, -1])
    else:
        caxspec_kw = dict(
            right=1.0,
            left=left,
        )
        cax_spec = fig.add_gridspec(nrow, 1, **caxspec_kw)
        cax = fig.add_subplot(cax_spec[:, -1])

    cax.set_frame_on(False)
    cax.grid(False)
    cax.set_xticks([])
    cax.set_yticks([])

    for ax in axs.flat[nplot:]:
        ax.axis("off")

    return fig, axs, cax


def plot_dim_loadings(
    adata: AnnData,
    components: Union[int, Sequence[int]],
    ncol: int = 2,
    n_loadings: int = 10,
    show_symbol: bool = True,
    varm: str = "PCs",
    **kwargs,
):
    components = (
        list(range(components)) if isinstance(components, int) else list(components)
    )
    dat = adata.varm[varm][:, components]
    nplots = len(components)

    _subplot_kwargs = dict(
        nplots=nplots,
        ncol=ncol,
        figsize=kwargs.pop("figsize", None),
        sharex=kwargs.pop("sharex", True),
        layout=kwargs.pop("layout", "constrained"),
        hspace=0.15,
    )

    fig, axs = configure_subplots(**_subplot_kwargs)

    show_symbol = show_symbol and adata.uns["config"]["var_names"] != "symbol"

    n_min = n_loadings // 2
    n_max = n_loadings - n_min
    for i, (ax, dim) in enumerate(zip(axs.flat, components)):
        ax.set_title(f"{varm} {dim}")

        # get indices of sorted dim loadings (ascending)
        idx = np.argsort(dat[:, i])
        idx = np.hstack([idx[:n_min], idx[-n_max:]])

        genes = adata.var.index[idx]
        if show_symbol:
            genes = adata.var.loc[genes, "symbol"]

        loadings = dat[idx, i]
        hlines = [((0, row), (loading, row)) for row, loading in enumerate(loadings)]

        line_segments = LineCollection(hlines, colors="k")
        ax.add_collection(line_segments)

        ax.scatter(loadings, genes, c="b")
        ax.axvline(0, linestyle="--", c="k")

        # Show ticks at the bottom of each column
        ax.tick_params("x", labelbottom=i + ncol >= len(components))

    fig.supxlabel("Loading")
    fig.supylabel("Gene")

    return axs


def elbow_plot(
    adata: AnnData, ndims: int = 20, reduction: str = "pca", ax: Optional[Axes] = None
):
    if ax is None:
        fig, ax = plt.subplots()
    var_ratio = adata.uns[reduction]["variance_ratio"][:ndims]
    ndims = var_ratio.size
    ax.plot(
        np.arange(ndims, dtype=int),
        var_ratio * 100,
        marker=".",
        markersize=6,
        linewidth=1,
        c="k",
    )
    ax.set_xticks(np.arange(0, ndims + 1, 3, dtype=int))
    ax.set_ylabel("Variance explained (%)")
    ax.set_xlabel("PC")
    return ax


def plot_pca(
    adata: AnnData,
    ndim: int = 5,
    cmap: Optional[str] = None,
    color_by: str = "cluster",
    obsm="X_pca",
    legend_kwargs: Optional[Dict[str, Any]] = None,
    subplot_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    data = adata.obsm[obsm]
    rc_context = {"axes.edgecolor": "#00000050", "axes.grid.which": "both"}
    _subplot_kwargs = dict(
        figsize=kwargs.pop("figsize", None),
    )
    _subplot_kwargs.update(subplot_kwargs or {})

    with plt.rc_context(rc_context):
        fig, axs, cax = subplots_single_colorbar(ndim * ndim, ndim, **_subplot_kwargs)

    var_expl = np.round(adata.uns["pca"]["variance_ratio"] * 100).astype(int)

    _legend_kwargs = dict(
        bbox_to_anchor=(1.04, 0.5),
        frameon=False,
        loc="center left",
        num=None,
        title=color_by,
    )
    _legend_kwargs.update(legend_kwargs or {})
    scatter_kwargs = dict(
        color_by=color_by,
        data=adata.obs,
        s=8,
        alpha=0.5,
        cmap=cmap,
        vmin=0,
        vmax=plt.get_cmap(cmap).N,
        legend=False,
    )

    scatter_kwargs.update(kwargs)

    for row in range(ndim):
        if ndim > 1:
            scatters_in_row = [ax for i, ax in enumerate(axs[row, :]) if i != row]
            scatters_in_row[0].get_shared_y_axes().join(
                scatters_in_row[0], *scatters_in_row
            )
            axs[1, row].get_shared_x_axes().join(axs[1, row], *axs[1:, row])

        for col in range(ndim):
            ax = axs[row, col]

            if row != col:
                ax = scatter(*data[:, (col, row)].T, ax=ax, **scatter_kwargs)
            if row < ndim - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])
            ax.set_frame_on(True)

        axs[0, row].set_xlabel(f"PC{row} ({var_expl[row]:d}%)")
        axs[0, row].xaxis.set_label_position("top")
        axs[row, -1].set_ylabel(
            f"PC{row} ({var_expl[row]:d}%)", rotation=270, labelpad=15
        )
        axs[row, -1].yaxis.set_label_position("right")

        density = gaussian_kde(data[:, row])
        xs = np.linspace(data[:, row].min(), data[:, row].max(), 200)
        axs[row, row].plot(xs, density(xs), c="k", linewidth=1)

    if ndim > 1:
        legend_elements = (
            axs.flat[1]
            .collections[0]
            .legend_elements(num=_legend_kwargs.pop("num", None))
        )
        # cax.legend(*legend_elements, loc="center", title=colorby, frameon=False, num=None)
        cax.legend(*legend_elements, **_legend_kwargs)
    else:
        cax.remove()

    return axs


def contour_plot(
    ax: Axes,
    x: Union[str, np.ndarray, Series],
    y: Union[str, np.ndarray, Series],
    data: Any = None,
    shape: Tuple[int, int] = (100, 100),
    levels: int = 7,
    colors: Union[str, Sequence[str]] = "cyan",
    linewidths: Optional[float] = 1,
    origin: Optional[str] = None,
) -> Axes:
    if data is not None:
        xdat = data[x] if isinstance(x, str) else x[:]
        ydat = data[y] if isinstance(y, str) else y[:]
    elif not (isinstance(x, str) or isinstance(y, str)):
        xdat = x
        ydat = y
    else:
        raise ValueError("x and y must both be arrays if data is None")

    points = np.vstack([xdat, ydat])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xdim, ydim = shape
    xdim = complex(0, xdim)
    ydim = complex(0, ydim)

    X, Y = np.mgrid[xmin:xmax:xdim, ymin:ymax:ydim]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(points)
    Z = np.reshape(kernel(positions), X.shape)

    ax.contour(
        X, Y, Z, levels=levels, colors=colors, linewidths=linewidths, origin=origin
    )

    return ax


def moran_plot(
    adata: AnnData,
    feature: Union[str, Sequence],
    graph_name: Union[None, str, np.ndarray] = None,
    color_by: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    contour_kwargs: Optional[Dict[str, Any]] = None,
    show_symbol: bool = True,
    ncol: int = 2,
    legend: bool = True,
    subplot_kwargs: Optional[Dict[str, Any]] = None,
    layer: Optional[str] = None,
    **scatter_kwargs,
) -> Axes:
    adata = adata.copy()

    features = [feature] if isinstance(feature, str) else feature
    del feature

    nplot = len(features)
    ncol = min(ncol, nplot)
    nrow = int(np.ceil(nplot / ncol))

    _subplot_kwargs = dict(
        layout=None if nplot == 1 else "constrained",
        wspace=0.2,
        hspace=0.2,
        cax_space=0.2,
        cax_width=0.01,
        figsize=scatter_kwargs.pop("figsize", None),
        legend=legend,
    )
    _subplot_kwargs.update(subplot_kwargs or {})

    if ax is None:
        fig, axs, cax = subplots_single_colorbar(nplot, ncol, **_subplot_kwargs)
    else:
        if isinstance(ax, Axes):
            axs = np.array([[ax]])
        if axs.size < nplot:
            raise ValueError("Fewer axes than plots.")

    X = adata.X if layer is None else adata.layers[layer].A

    for i_plot, (feature, ax) in enumerate(zip(features, axs.flat)):
        lagged_feature = f"lagged_{feature}"
        if lagged_feature not in adata.obs:
            spatial.compute_spatial_lag(
                adata, feature, graph_name=graph_name, inplace=True
            )

        rc_context = {
            "axes.grid": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            "axes.spines.left": True,
            "axes.spines.right": True,
        }

        y_label = f"Spatially lagged {feature}"

        _contour_kwargs = dict(
            shape=(150, 150),
            levels=7,
            colors="cyan",
            linewidths=1,
        )
        _contour_kwargs.update(contour_kwargs or {})

        labels = dict(x=feature, y=y_label)

        if feature in adata.var_names:
            i_feature = adata.var_names.get_loc(feature)
            adata.obs[feature] = X[:, i_feature]
            if show_symbol:
                symbol = adata.var.at[
                    feature, adata.uns["config"]["secondary_var_names"]
                ]
                labels["x"] = symbol
                labels["y"] = f"Spatially lagged {symbol}"

        ax = scatter(
            x=feature,
            y=lagged_feature,
            color_by=color_by,
            labels=labels,
            rc_context=rc_context,
            fitline_kwargs=dict(color="b"),
            contour_kwargs={"colors": "cyan"},
            data=adata.obs,
            ax=ax,
            legend=legend and (i_plot == nplot - 1),
            legend_kwargs=dict(cax=(cax if i_plot == nplot - 1 else None)),
            **scatter_kwargs,
        )

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.axvline(adata.obs[feature].mean(), linestyle="--", c="k", alpha=0.5)
        ax.axhline(adata.obs[lagged_feature].mean(), linestyle="--", c="k", alpha=0.5)

        ax.set_aspect("equal")

    return axs


def plot_moran_mc(
    adata: AnnData,
    feature: Union[str, Sequence[str]],
    graph_name: Optional[str] = None,
    cmap: Optional[str] = None,
    ax: Optional[Axes] = None,
    legend_title: str = "feature",
    **kwargs,
):
    linewidth = kwargs.pop("linewidth", None)
    _kwargs = dict(figsize=None)
    _kwargs.update(kwargs)
    if ax is None:
        fig, (ax,) = configure_subplots(1, 1, squeeze=True, **_kwargs)

    ax.set_prop_cycle("color", plt.get_cmap(cmap).colors)

    if graph_name is None:
        graph_name = spatial.get_default_graph(adata)
    moran_df = adata.uns["spatial"].get("moran", {}).get(graph_name, None)
    moran_sim_dict = adata.uns["spatial"].get("moran_mc", {}).get(graph_name, None)

    if moran_df is None or moran_sim_dict is None:
        raise KeyError(
            "The AnnData object does not have the results needed for this plot.\
 Please run `voyagerpy.spatial.moran` with permutations > 0 for the selected features."
        )

    features = [feature] if isinstance(feature, str) else feature[:]

    Is = moran_df.loc[feature, "I"]
    minI, maxI = Is.min(), Is.max()

    for feat, I in zip(features, Is):
        sim = moran_sim_dict[feat]["sim"]
        label = adata.var.loc[feat, "symbol"]

        x1 = min(minI, sim.min())
        x2 = max(maxI, sim.max())
        xs = np.linspace(x1, x2, 200)
        kernel = gaussian_kde(sim)
        (p,) = ax.plot(xs, kernel(xs), label=label, linewidth=linewidth)
        ax.axvline(I, color=p.get_c(), linewidth=linewidth)

    ax.legend(
        loc="center left", bbox_to_anchor=(1.04, 0.5), title=legend_title, frameon=False
    )

    return ax


def plot_barcode_histogram(
    adata: AnnData,
    feature: Union[str, Sequence[str]],
    color_by: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ncol: int = 1,
    cmap: Optional[str] = None,
    bins: int = 100,
    log: bool = True,
    stacked: bool = False,
    histtype: str = "step",
    obsm: Optional[str] = None,
    label: Optional[Sequence[str]] = None,
    subplot_kwargs: Optional[Dict[str, Any]] = None,
    **hist_kwargs,
) -> np.ndarray[Axes]:
    features: List[str] = utils.listify(feature)  # type: ignore
    nplot = len(features)
    ncol = min(ncol, nplot)

    if obsm is not None:
        if obsm not in adata.obsm:
            raise KeyError(f'"{obsm}" not found in adata.obsm')
        df: DataFrame = adata.obsm[obsm].copy()  # type: ignore
        if color_by is not None and color_by not in df:
            df[color_by] = adata.obs[color_by].copy()
    else:
        df = adata.obs.copy()

    _subplots_kwargs = dict(figsize=figsize, layout="constrained", hspace=None)
    _subplots_kwargs.update(subplot_kwargs or {})

    labels = [label] if isinstance(label, str) else label
    del label
    if labels is not None and len(labels) != len(features):
        raise ValueError("labels must be the same length as features")

    if color_by is not None:
        fig, axs, cax = subplots_single_colorbar(nplot, ncol, **_subplots_kwargs)
        all_feats = features + [color_by]
        keys, groups = zip(*df[all_feats].groupby(color_by).groups.items())

        colormap = plt.get_cmap(cmap)
        colors = [colormap(i) for i in range(len(keys))]

        for i_ax, (feat, ax) in enumerate(zip(features, axs.flat)):
            hist_range = np.array([df[feat].min(), df[feat].max()])

            hist_data = [
                np.histogram(
                    df.loc[group, feat].values,
                    bins=bins,
                    range=hist_range,
                )
                for group in groups
            ]

            counts, bin_edges = zip(*hist_data)
            bin_edges = np.array(bin_edges)
            counts = np.array(counts)
            if histtype != "line":
                n, bins_, rects = ax.hist(
                    bin_edges[:, :-1].T,
                    bins=bins,
                    log=log,
                    label=keys,
                    stacked=stacked,
                    weights=counts.T,
                    color=colors[: len(keys)],
                    histtype=histtype,
                    **hist_kwargs,
                )
            else:
                ax.set_prop_cycle("color", plt.get_cmap(cmap).colors)

                centers = np.diff(bin_edges) / 2 + bin_edges[:, :-1]
                bin_width = np.diff(bin_edges).mean(axis=1, keepdims=True)

                # CLEANUP: We're adding bins to both ends to make the lines go to zero
                if True:
                    zeros = np.zeros((centers.shape[0], 1))
                    counts = np.hstack([zeros, counts, zeros])
                    centers = np.hstack(
                        [
                            centers[:, :1] - bin_width,
                            centers,
                            centers[:, -1:] + bin_width,
                        ]
                    )

                # we plot counts + log to map zero to zero without losing plotted data
                counts += log
                rects = ax.plot(centers.T, counts.T, **hist_kwargs)

                if log:
                    ax.set_yscale("log")

            ax.grid(False, "minor")
            ax.set_xlabel(feat if labels is None else labels[i_ax])
            ax.xaxis.set_label_position("top")

        if histtype.startswith("step"):
            # Hack to get the actual handles
            handles = [patch[0] for patch in rects]
        else:
            handles = rects
        cax.legend(
            handles=handles,
            labels=keys,
            loc="center left",
            title=color_by,
            frameon=False,
        )
    else:
        fig, axs = configure_subplots(nplot, ncol)
        # TODO

    fig.supylabel("count")
    fig.supxlabel("values")
    if obsm is not None:
        fig.suptitle(f"Barcode histogram for {obsm}", ha="left", x=0.0)
    return axs


def plot_correlogram(
    adata,
    graph_name=None,
    metric="moran",
    order=None,
    show_symbol: bool = True,
    ax=None,
    features=None,
    cmap=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.get_cmap(cmap).colors)
    if graph_name is None:
        graph_name = spatial.get_default_graph(adata)

    corr_dict = (
        adata.uns.setdefault("spatial", {})
        .setdefault(metric, {})
        .setdefault("correlogram", {})
    )
    if graph_name not in corr_dict:
        raise KeyError(
            f"Graph {graph_name} not found in adata.uns['spatial']['{metric}']['correlogram']"
        )

    df = corr_dict[graph_name]
    # df = adata.uns["spatial"][metric]["correlogram"][graph_name]
    if order is None:
        order = slice(None)
    if isinstance(order, tuple):
        order = slice(*order)
    if features is None:
        features = df.index
    cols = df.columns[order]

    for feature in features:
        ss = df.loc[feature, cols]
        label = adata.var.at[feature, "symbol"] if show_symbol else feature
        ax.plot(ss.T, "-o", label=label)
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title="feature")
    ax.axhline(0, linestyle="--")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(metric.title())
    ax.set_xlabel("Lag")
    return ax


def plot_features_histogram(
    adata: AnnData,
    features: Union[str, Sequence[str]],
    fill_by: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ncol: int = 1,
    cmap: Optional[str] = None,
    bins: int = 100,
    log: bool = True,
    stacked: bool = False,
    histtype: str = "step",
    markers: Union[None, str, Sequence[str]] = None,
    show_symbol: bool = True,
    **hist_kwargs,
) -> np.ndarray[Axes]:
    features = [features] if isinstance(features, str) else features
    nplot = len(features)
    ncol = min(ncol, nplot)
    nrow = int(np.ceil(nplot / ncol))

    _hist_kwargs = dict(edgecolor="w")

    _hist_kwargs.update(hist_kwargs)

    if fill_by is not None:
        fig, axs, cax = subplots_single_colorbar(nrow, ncol, figsize=figsize)
        all_feats = features + [fill_by]
        keys, groups = zip(*adata.obs[all_feats].groupby(fill_by).groups.items())

        colormap = plt.get_cmap(cmap)
        colors = [colormap(i) for i in range(len(keys))]

        for feat, ax in zip(features, axs.flat):
            # these should probably be .var
            hist_range = np.array([adata.obs[feat].min(), adata.obs[feat].max()])
            hist_mid = hist_range.mean()
            hist_range = tuple((hist_range - hist_mid) * 1.05 + hist_mid)

            hist_data = [
                np.histogram(
                    adata.obs.loc[group, feat].values,
                    bins=bins,
                    range=hist_range,
                    **_hist_kwargs,
                )
                for group in groups
            ]

            counts, bin_edges = zip(*hist_data)
            bin_edges = np.array(bin_edges)
            counts = np.array(counts)
            if histtype != "line":
                n, bins_, rects = ax.hist(
                    bin_edges[:, :-1].T,
                    bins=bins,
                    range=hist_range,
                    log=log,
                    label=keys,
                    stacked=stacked,
                    weights=counts.T,
                    color=colors[: len(keys)],
                    histtype=histtype,
                    **_hist_kwargs,
                )
            else:
                # TODO: make this match the barcode histogram
                ax.set_prop_cycle("color", plt.get_cmap(cmap).colors)

                centers = np.diff(bin_edges) / 2 + bin_edges[:, :-1]
                rects = ax.plot(centers.T, np.maximum(0.2, counts.T))

                if log:
                    ax.set_yscale("log")

                ax.set_ylim(0.2, None)

            ax.grid(False, "minor")
            ax.set_xlabel(feat, size=10)

            add_markers(
                adata.var,
                feat,
                markers,
                ax=ax,
                cmap=cmap,
                label_col="symbol",
                legend=True,
            )

        if histtype.startswith("step"):
            # Hack to get the actual handles
            handles = [patch[0] for patch in rects]
        else:
            handles = rects
        cax.legend(
            handles=handles,
            labels=keys,
            loc="center left",
            title=fill_by,
            frameon=False,
        )
    else:
        fig, axs = configure_subplots(nplot, ncol, figsize=figsize)
        _hist_kwargs.setdefault("color", "#666666ff")

        for ax, feat in zip(axs.flat, features):
            if histtype != "line":
                n, _, rects = ax.hist(
                    adata.var[feat],
                    bins=bins,
                    log=log,
                    histtype=histtype,
                    **_hist_kwargs,
                )
            else:
                _hist_kwargs.pop("ec", None)
                _hist_kwargs.pop("edgecolor", None)
                counts, bin_edges = np.histogram(
                    adata.var[feat].dropna(inplace=False).values, bins=bins
                )
                centers = np.diff(bin_edges) / 2 + bin_edges[:-1]
                rects = ax.plot(centers, counts, **_hist_kwargs)

            add_markers(
                adata.var,
                feat,
                markers,
                ax=ax,
                cmap=cmap,
                label_col="symbol" if show_symbol else None,
                legend=True,
            )
            ax.set_xlabel(feat)

    fig.supylabel("count")

    return axs


def add_markers(
    df,
    feat,
    markers: Union[None, str, Sequence[str]],
    ax: Axes,
    cmap: Optional[str] = None,
    label_col: Optional[str] = None,
    legend: bool = True,
):
    if markers is not None:
        marker_list = [markers] if isinstance(markers, str) else markers[:]
        colors = plt.get_cmap(cmap)

        for i_marker, marker in enumerate(marker_list):
            value = df.loc[marker, feat]
            label_suffix = marker if label_col is None else df.loc[marker, label_col]
            label = f"{i_marker} - {label_suffix}"
            ax.axvline(value, color=colors(i_marker), label=label)

        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title="Marker Gene")
    return ax


def plot_fitline(
    x: Union[str, Series, np.ndarray],
    y: Union[str, Series, np.ndarray],
    alternative: str = "two-sided",
    ax: Optional[Axes] = None,
    color: Optional[str] = None,
    data: Any = None,
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
):
    if ax is None:
        fig, ax = plt.subplots()

    if data is not None:
        xdat = data[x] if isinstance(x, str) else x[:]
        ydat = data[y] if isinstance(y, str) else y[:]
    else:
        xdat = x[:]
        ydat = y[:]

    xdat = np.array(xdat)
    ydat = np.array(ydat)

    non_nans = ~(np.isnan(xdat) | np.isnan(ydat))
    reg = linregress(xdat[non_nans], ydat[non_nans], alternative=alternative)

    xmax = xdat.max()  # max(0, xdat.max())
    xmin = xdat.min()  # min(xdat.min(), 0)
    ymin = xmin * reg.slope + reg.intercept
    ymax = xmax * reg.slope + reg.intercept

    ax.axline((xmin, ymin), (xmax, ymax), color=color)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return ax


def scatter(
    x: str,
    y: str,
    color_by: Optional[str] = None,
    ax: Optional[Axes] = None,
    cmap: Optional[str] = None,
    legend: bool = True,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    data: Optional[DataFrame] = None,
    labels: Optional[Dict[str, Any]] = None,
    rc_context: Optional[Dict[str, Any]] = None,
    fitline_kwargs: Optional[Dict[str, Any]] = None,
    contour_kwargs: Optional[Dict[str, Any]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    is_categorical: Optional[bool] = None,
    aspect: Union[None, float, str] = None,
    **scatter_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # cmap cannot be passed to plt.scatter if color_by is None
    if color_by is None:
        cmap = None
        scatter_kwargs.pop("vmin", None)
        scatter_kwargs.pop("vmax", None)

    _scatter_kwargs = dict(cmap=cmap, s=8)
    cmap_kwargs = dict()

    # is_categorical = False
    if data is None:
        if isinstance(color_by, str):
            raise ValueError("data must not be None if color_by is str")
        if isinstance(x, str):
            raise ValueError("data must not be None if x is str")
        if isinstance(y, str):
            raise ValueError("data must not be None if y is str")

    xdat, xstr = (data[x], x) if isinstance(x, str) else (x, None)
    ydat, ystr = (data[y], y) if isinstance(y, str) else (y, None)
    color_dat, color_str = (
        (data[color_by], color_by) if isinstance(color_by, str) else (color_by, None)
    )

    del color_by, x, y

    # if color_dat is not None and data is not None and data[color_dat].dtype == "category":
    if is_categorical or color_dat is not None and color_dat.dtype == "category":
        is_categorical = True
        _scatter_kwargs.update(
            dict(
                vmax=plt.get_cmap(cmap).N,
                c=color_dat.astype(int),
            )
        )

    elif color_dat is not None:
        is_categorical = False
        _scatter_kwargs.update(dict(c=color_dat))

    _rc_context = {}
    _rc_context.update(rc_context or {})

    with plt.rc_context(_rc_context):
        _scatter_kwargs.update(scatter_kwargs)
        scat = ax.scatter(xdat, ydat, **_scatter_kwargs)

    if contour_kwargs is not None:
        _contour_kwargs = dict(x=xdat, y=ydat, data=data)
        _contour_kwargs.update(contour_kwargs)
        contour_plot(ax, **_contour_kwargs)

    if fitline_kwargs is not None:
        _fitline_kwargs = dict(x=xdat, y=ydat, ax=ax)
        _fitline_kwargs.update(fitline_kwargs)
        ax = plot_fitline(**_fitline_kwargs)

    labels = labels or {}

    ax.set_xlabel(labels.get("x", xstr))
    ax.set_ylabel(labels.get("y", ystr))
    ax.set_title(labels.get("title", None))

    legend_kwargs = legend_kwargs or {}

    label_on_top = False
    if legend and is_categorical:
        _legend_kwargs = dict(
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            frameon=False,
            title=color_str,
        )

        legend_kwargs.pop("label", None)
        _legend_kwargs.update(legend_kwargs)
        cax = _legend_kwargs.pop("cax", None)
        if cax is None:
            cax = ax
        else:
            _legend_kwargs.pop("bbox_to_anchor", None)
        cax.legend(*scat.legend_elements(), **_legend_kwargs)

    elif legend and color_dat is not None:
        # I'm not 100% certain which one to use
        use_divider = not True

        if use_divider:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            cmap_kwargs["cax"] = cax
        else:
            cax = None
            cmap_kwargs["ax"] = ax
            cmap_kwargs["use_gridspec"] = True

        cmap_kwargs.update(legend_kwargs or {})
        cbar_title = cmap_kwargs.pop("title", None)

        cbar = fig.colorbar(scat, **cmap_kwargs)

        if cbar_title is not None and label_on_top:
            cbar.ax.set_title(cbar_title, fontsize=8)
        elif cbar_title is not None:
            cbar.set_label(cbar_title)

    if aspect is not None:
        ax.set_aspect(aspect)
    return ax
