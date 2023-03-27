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
    Iterable,
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
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
import matplotlib.ticker
from pandas import DataFrame, Series, options
from pandas.api.types import is_categorical_dtype
from scipy.stats import gaussian_kde, linregress
from scipy import sparse as sp

from voyagerpy import spatial, utils
from .cmap_helper import DivergentNorm


options.mode.chained_assignment = None  # default='warn'

plt.style.use("ggplot")
plt.rcParams["axes.edgecolor"] = "#00000050"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid.which"] = "both"
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["grid.alpha"] = 0.1
plt.rcParams["grid.color"] = "k"
plt.rcParams["image.origin"] = "lower"


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

    axs.set_xticks([])
    axs.set_yticks([])
    im_kwargs.update(kwargs)
    if title is not None:
        axs.set_title(title)

    axs.imshow(img, **im_kwargs)
    return axs


def configure_violins(violins, cmap=None, edgecolor="#00000050", alpha=0.7):
    cmap = plt.get_cmap(cmap) if isinstance(cmap, (str, type(None))) else cmap
    for i, violin in enumerate(violins["bodies"]):
        violin.set_facecolor(cmap(i))
        violin.set_edgecolor(edgecolor)
        violin.set_alpha(alpha)
    return edgecolor, alpha


def simple_violinplot(ax: Axes, df: DataFrame, y: Union[str, int], cmap=None, **kwargs):
    violin_opts = dict(showmeans=False, showextrema=False, showmedians=False)
    kwargs.pop("legend", False)
    violin_opts.update(kwargs)
    violins = ax.violinplot(df[y], **violin_opts)
    configure_violins(violins, cmap)
    ax.set_ylabel(str(y))
    return ax


def grouped_violinplot(
    ax: Axes,
    df: DataFrame,
    x: str,
    y: str,
    cmap: Optional[str] = None,
    legend: bool = True,
    vert: bool = True,
):
    if not vert:
        x, y = y, x
    groups = df.groupby(x)[y].groups
    keys = sorted(groups.keys())
    labels = keys
    if df[x].dtype == "category" and keys == [0, 1]:
        labels = [False, True]

    grouped_data = [df.loc[groups[key], y] for key in keys]
    violin_opts = dict(showmeans=False, showextrema=False, showmedians=False, vert=vert)
    violins = ax.violinplot(grouped_data, **violin_opts)
    ec, alpha = configure_violins(violins, cmap)

    set_ticks = ax.set_xticks if vert else ax.set_yticks
    set_ticks(np.arange(len(keys)) + 1, labels=labels)

    if not vert:
        x, y = y, x

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if not legend:
        return ax

    colormap = plt.get_cmap(cmap)
    for i, label in enumerate(labels):
        ax.scatter([], [], label=label, color=colormap(i), alpha=alpha)
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", title=x, frameon=False)
    return ax


def plot_single_barcode_data(
    adata: AnnData,
    y: Union[int, str],
    x: Union[int, str, None] = None,
    cmap: Optional[str] = None,
    ax: Optional[Axes] = None,
    legend: bool = False,
    color_by: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    contour_kwargs: Optional[Dict[str, Any]] = None,
):
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
            ax = grouped_violinplot(ax, adata.obs, x, y, cmap, legend=legend, vert=False)
    else:
        if x is None:
            ax = simple_violinplot(ax, adata.obs, y, cmap)
        elif is_categorical_dtype(obs[x]):
            ax = grouped_violinplot(ax, adata.obs, x, y, cmap, legend=legend)
        else:
            colors = np.zeros_like(adata.obs[x], "int") if color_by is None else adata.obs[color_by].astype(int)
            colormap = plt.get_cmap(cmap)
            scat = ax.scatter(x, y, data=adata.obs, c=colors, cmap=colormap, vmin=0, vmax=colormap.N, alpha=0.5, s=8)
            if contour_kwargs is not None:
                points = adata.obs[[x, y]].values.T
                # contour_plot(ax, points, **contour_kwargs)
                _contour_kwargs = dict(x=x, y=y, data=adata.obs)
                _contour_kwargs.update(contour_kwargs)
                contour_plot(ax, **_contour_kwargs)

            ax.set_xlabel(x_label or x)
            ax.set_ylabel(y_label or y)
            if legend and color_by is not None:
                ax.legend(*scat.legend_elements(), bbox_to_anchor=(1.04, 0.5), loc="center left", title=color_by, frameon=False)

    return ax


def configure_subplots(nplots: int, ncol: Optional[int] = 2, **kwargs) -> Tuple[Figure, npt.NDArray[plt.Axes]]:
    ncol = min(ncol or 2, nplots)
    nrow = int(np.ceil(nplots / ncol))

    default_figsize = (10, 7) if nrow >= 2 and ncol == 3 else (10, 10)
    plot_kwargs = {"figsize": default_figsize}
    plot_kwargs.update(kwargs)

    fig, axs = plt.subplots(nrow, ncol, **plot_kwargs)
    if nplots == 1:
        axs = np.array([axs])

    assert isinstance(axs, np.ndarray)
    for ax in axs.flat[nplots:]:
        ax.axis("off")

    return fig, axs


def plot_barcode_data(
    adata: AnnData,
    y: Union[str, Sequence[str]],
    x: Optional[str] = None,
    ncol: Optional[int] = None,
    cmap: Union[None, str, colors.ListedColormap, colors.LinearSegmentedColormap] = None,
    color_by: Optional[str] = None,
    sharex: Union[None, Literal["none", "all", "row", "col"], bool] = None,
    sharey: Union[None, str, bool] = None,
    figsize: Optional[Tuple[float, float]] = None,
    x_labels: Union[None, str, Sequence[str]] = None,
    y_labels: Union[None, str, Sequence[str]] = None,
    contour_kwargs: Optional[Dict[str, Any]] = None,
    rc_context: Optional[Dict[str, Any]] = None,
):
    #  TODO: Allow ax argument

    x_features = x if isinstance(x, (list, tuple)) else [x]
    y_features = y if isinstance(y, (list, tuple)) else [y]

    if x_labels is None:
        x_labels = x_features[:]

    x_labels = x_labels if isinstance(x_labels, (list, tuple)) else [x_labels]
    if isinstance(x_labels, (list, tuple)) and len(x_labels) != len(x_features):
        raise ValueError("x_labels must have the same length as x")

    # if isinstance(x_label, str)

    if y_labels is None:
        y_labels = y_features[:]
    y_labels = y_labels if isinstance(y_labels, (list, tuple)) else [y_labels]
    if isinstance(y_labels, (list, tuple)) and len(y_labels) != len(y_features):
        raise ValueError("y_labels must have the same length as y")

    del x, y

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

    default_rc_context = {
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.left": True,
        "axes.spines.right": False,
        "axes.grid": False,
    }

    default_rc_context.update(rc_context or {})

    with plt.rc_context(default_rc_context):
        fig, axs_arr = configure_subplots(nplots, ncol, figsize=figsize, sharex=sharex, sharey=sharey)

    feature_iterator = ((y_feat, x_feat) for y_feat in zip(y_labels, y_features) for x_feat in zip(x_labels, x_features))

    for i_plot, (y_feat, x_feat) in enumerate(feature_iterator):
        i_col = i_plot % ncol
        add_legend = i_col == ncol - 1
        x_label, x_feat = x_feat
        y_label, y_feat = y_feat

        axs_arr.flat[i_plot] = plot_single_barcode_data(
            adata,
            y=y_feat,
            x=x_feat,
            cmap=cmap,
            ax=axs_arr.flat[i_plot],
            legend=add_legend,
            color_by=color_by,
            x_label=x_label,
            y_label=y_label,
            contour_kwargs=contour_kwargs,
        )

    fig.tight_layout()
    return axs_arr


def plot_bin2d(
    data: Union[AnnData, "pd.DataFrame"],
    x: str,
    y: str,
    filt: Optional[str] = None,
    subset: Optional[str] = None,
    bins: int = 101,
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
    x_label = x
    y_label = y

    x = obs[x]
    y = obs[y]

    if subset is None:
        myfilt: Any = Ellipsis if filt is None else obs[filt].astype(bool)

        im = plot_fun(x[myfilt], y[myfilt], **plot_kwargs)  # type: ignore
        cbar = plt.colorbar(im[-1] if isinstance(im, tuple) else im)
        cbar.ax.set_title("count")

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

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    ax.grid(**grid_kwargs)
    ax.set_facecolor("w")
    return ax


def plot_expression(
    adata: AnnData,
    genes: Union[str, Sequence[str]],
    groupby: Optional[str] = None,
    ncol: Optional[int] = 2,
    show_symbol: bool = False,
    layer: Optional[str] = "logcounts",
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
):
    genes = [genes] if isinstance(genes, str) else genes[:]
    gene_ids = []

    secondary_column = adata.uns["config"]["secondary_var_names"]

    for gene in genes:
        if gene in adata.var[secondary_column].values:
            new_genes = adata.var.index[adata.var[secondary_column] == gene]
            gene_ids.extend(new_genes.tolist())
        else:
            assert gene in adata.var_names
            gene_ids.append(gene)

    obs = adata.obs[[groupby]] if groupby is not None else adata.obs.copy()
    for gene_id in gene_ids:
        subset_gene = adata.var_names == gene_id
        if layer is not None:
            counts = adata.layers[layer][:, subset_gene]
        else:
            counts = adata.X[:, subset_gene]
        if isinstance(counts, sp.csr_matrix):
            counts = counts.todense()
        obs[gene_id] = counts.copy()

    nplots = len(gene_ids) if groupby is not None else 1
    fig, axs = configure_subplots(nplots, ncol, sharey=True, figsize=figsize)

    if groupby is None:
        gene_ids = [gene_ids]

    # if groupby is not None:
    #     for ax, gene_id in zip(axs.flat, gene_ids):
    #         grouped_violinplot(ax, obs, groupby, gene_id, legend=False)
    #         title = gene_id
    #         if show_symbol and secondary_column == "symbol":
    #             title = adata.var.at[gene_id, secondary_column]

    #         ax.set_title(title, fontsize=10)
    #         ax.set_ylabel("")
    #         ax.set_xlabel("")
    # else:
    #     for ax, genes in zip(axs.flat, gene_ids):
    #         simple_violinplot(ax, obs, genes, legend=False)

    for ax, gene_id in zip(axs.flat, gene_ids):
        if groupby is not None:
            grouped_violinplot(ax, obs, groupby, gene_id, legend=False, cmap=cmap)
            title = gene_id
            if show_symbol and secondary_column == "symbol":
                title = adata.var.at[gene_id, secondary_column]
            elif show_symbol:
                title = gene_id

            ax.set_title(title, fontsize=10)
        else:
            simple_violinplot(ax, obs, gene_id, legend=False, cmap=cmap)
            labels = gene_id
            if show_symbol and secondary_column == "symbol":
                labels = adata.var.loc[gene_id, "symbol"]
            elif show_symbol:
                labels = gene_id
            ax.set_xticks(np.arange(len(gene_id)) + 1, labels=labels, rotation=60)

        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.supylabel("Expression" + f" ({layer})" if layer is not None else "", fontsize=10)
    if groupby is not None:
        fig.supxlabel(groupby, fontsize=10)
    fig.tight_layout()
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
    ncol: int = 3,
    barcode_geom: Optional[str] = None,
    annot_geom: Optional[str] = None,
    tissue: bool = True,
    layer: Optional[str] = None,
    colorbar: bool = False,
    cmap: Optional[str] = "Blues",
    cat_cmap: Optional[str] = "dittoseq",
    div_cmap: str = "roma",
    categ_type: Union[str, Collection[str]] = {},
    geom_style: Optional[Dict] = {},
    annot_style: Optional[Dict] = {},
    alpha: float = 0.2,
    divergent: bool = False,
    color: Optional[str] = None,
    _ax: Union[None, Axes, Iterable[Axes]] = None,
    legend: bool = True,
    plot: bool = True,
    subplot_kwds: Optional[Dict] = {},
    legend_kwds: Optional[Dict] = {},
    dimension: str = "barcode",
    local: Optional[str] = None,
    obsm: Optional[str] = None,
    image: bool = False,
    **kwds,
) -> Union[np.ndarray, Any]:

    if isinstance(features, (list, tuple)):
        feat_ls = list(features)
    elif isinstance(features, str):
        feat_ls = [features]
    else:
        raise TypeError("features must be a string or a list of strings")

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
            labeled_features.append((feature, feature))
            continue
        if feature in adata.var.index:
            labeled_features.append((feature, feature))
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

    # TODO: This is a bit messy fix. Maybe get rid of the tissue param.
    tissue = tissue and "in_tissue" in df

    barcode_selection = slice(None) if not tissue else df["in_tissue"] == 1
    gene_selection = slice(None) if not var_features else utils.make_unique(var_features)

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

    subplot_kwds = subplot_kwds or {}
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

    if _ax is None:
        with plt.rc_context(_rc_context):
            fig, axs = configure_subplots(nplots=n_features, ncol=ncol, **subplot_kwds)
        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        # plt.subplots_adjust(wspace = 1/ncols +  0.2)
    else:
        if isinstance(_ax, Axes):
            axs = np.array([_ax])
        elif not isinstance(_ax, np.ndarray):
            axs = np.array(_ax)
        else:
            axs = _ax

        fig = axs.flat[0].get_figure()

    # iterate over features to plot

    cax_title_kwargs = dict(loc="left", fontsize=10)

    for ax, (feature, label) in zip(axs.flat, labeled_features):
        legend_kwds_ = deepcopy(legend_kwds)
        _legend = legend
        curr_cmap = cmap
        values = obs[feature]

        if values.dtype != "category":
            curr_cmap = cmap
            vmax = None

            legend_kwds_.setdefault("label", label)
            legend_kwds_.setdefault("orientation", "vertical")
            legend_kwds_.setdefault("shrink", 0.6)

        else:
            #  colorbar for discrete categories if pandas column is categorical
            curr_cmap = cat_cmap
            vmax = cm.get_cmap(cat_cmap).N
            _legend = False

            add_colorbar_discrete(
                ax=ax,
                fig=fig,
                cmap=curr_cmap,
                cbar_title=feature,
                cat_names=list(values.cat.categories),
                scale=False,
                title_kwargs=cax_title_kwargs,
            )

        norm = None
        if divergent:
            vmin = values.min()
            vmax = values.max()
            vcenter = 0
            norm = DivergentNorm(vmin, vmax, vcenter)
        if color is not None:
            curr_cmap = None

        if image:
            ax = imshow(adata, None, ax)

        geo.plot(
            column=values,
            ax=ax,
            color=color,
            legend=_legend,
            cmap=curr_cmap,
            norm=norm,
            vmax=vmax,
            legend_kwds=legend_kwds_,
            **geom_style,
            **kwds,
        )

        if annot_geom is not None:
            if annot_geom in adata.uns["spatial"]["geom"]:
                # check annot_style is dict with correct values
                annot_kwargs = dict(ax=ax, color="blue", alpha=alpha, **kwds)
                annot_kwargs.update(annot_style or {})

                plg = adata.uns["spatial"]["geom"][annot_geom]
                gpd.GeoSeries(plg).plot(**annot_kwargs)
            else:
                raise ValueError(f"Cannot find {annot_geom!r} data in adata.uns['spatial']['geom']")

    axs = fig.get_axes()

    for ax in axs:
        if ax.properties()["label"] == "<colorbar>" and ax.properties()["ylabel"] != "":
            ax.set_title(ax.properties()["ylabel"], **cax_title_kwargs)
            ax.set_ylabel("")

    return axs


def assert_basic_spatial_features(adata, dimension="barcode", errors: str = "raise") -> Tuple[bool, str]:

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
    return True, ""

    if not isinstance(adata.obs, gpd.GeoDataFrame):
        error_msg = "adata.obs must be a geopandas.GeoDataFrame. " + get_geom_prompt
        if errors == "raise":
            raise TypeError(error_msg)
        return False, error_msg

    if adata.obs.geometry.name not in adata.obs:
        error_msg = f'"{adata.obs.geometry.name}" must be a column in adata.obs. ' + get_geom_prompt
        if errors == "raise":
            raise KeyError(error_msg)
        return False, error_msg
    if "geom" not in adata.uns["spatial"]:
        error_msg = '"geom" must be a key in adata.uns["spatial"]. ' + get_geom_prompt
        if errors == "raise":
            raise KeyError(error_msg)
        return False, error_msg

    return True, ""


def spatial_reduced_dim(
    adata: AnnData,
    dimred: str,
    ncomponents: Union[int, Sequence[int]],
    barcode_geom: Optional[str] = None,
    ncol: int = 2,
    annot_geom: Optional[str] = None,
    tissue: bool = True,
    cmap: Optional[str] = None,
    div_cmap: str = "roma",
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
    adata = adata.copy()
    if isinstance(ncomponents, list):
        dims = ncomponents
    elif isinstance(ncomponents, int):
        dims = list(range(ncomponents))
    else:
        raise TypeError("features must be a integer or a list of integers")
    dim_nr = len(dims)

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
        ncol=ncol,
        barcode_geom=barcode_geom,
        annot_geom=annot_geom,
        tissue=tissue,
        cmap=cmap,
        div_cmap=div_cmap,
        geom_style=geom_style,
        annot_style=annot_style,
        alpha=alpha,
        divergent=divergent,
        color=color,
        _ax=_ax,
        legend=legend,
        subplot_kwds=subplot_kwds,
        legend_kwds=legend_kwds,
        **kwds,
    )

    fig = axs[0].get_figure()
    fig.suptitle(dimred, x=0, ha="left", fontsize="xx-large", va="bottom")
    return axs


def add_colorbar_discrete(
    ax, fig, cmap, cbar_title: str, cat_names: list, scale: bool = False, title_kwargs: Optional[Dict] = None
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


def subplots_single_colorbar(nrow: int = 1, ncol: int = 1, **kwargs):
    fig_kwargs = {"layout": "tight"}
    fig_kwargs.update(kwargs)
    figsize = fig_kwargs.pop("figsize", None)
    if isinstance(figsize, tuple):
        figsize = (figsize[0] + 0.2, figsize[1])

    fig = plt.figure(figsize=figsize, **fig_kwargs)

    width_ratios = [1] * ncol + [0.2]
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

    if isinstance(figsize, (int, float)):
        ncol = min(ncol, len(dims))
        nrow = int(np.ceil(len(dims) / ncol))
        figsize = (nrow * figsize / ncol, figsize)

    fig, axs = configure_subplots(len(dims), ncol, figsize=figsize, sharex=True)

    show_symbol = show_symbol and adata.uns["config"]["var_names"] != "symbol"

    for i, (ax, dim) in enumerate(zip(axs.flat, dims)):
        ax.set_title(f"PC{dim}")

        idx = np.argsort(dat[:, i])
        idx = np.hstack([idx[:n_extremes], idx[-n_extremes:]])
        genes = adata.var.index[idx]
        if show_symbol:
            genes = adata.var.loc[genes, "symbol"]

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


def plot_barcode_data_with_reduction(
    adata: AnnData,
    features: Union[str, Sequence[str]],
    reduction: str = "X_pca",
    ncol: int = 2,
    layout="tight",
    cmap_continuous: str = "Blues",
    cmap_discrete: str = "dittoseq",
    cmap_divergent: str = "roma",
    divergent: bool = False,
    divergence_center: Union[float, Sequence[float]] = 0,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    feature_labels: Optional[Sequence[str]] = None,
):
    obs = adata.obs.copy()
    x, y = adata.obsm[reduction][:, :2].T
    del adata
    features = [features] if isinstance(features, str) else features[:]
    feature_labels = feature_labels[:] if feature_labels is not None else features[:]

    fig, axs = configure_subplots(
        len(features),
        ncol,
        layout=layout,
        sharex=True,
        sharey=True,
        figsize=figsize,
    )

    vcenters = [divergence_center] * len(features) if isinstance(divergence_center, (float, int)) else divergence_center[:]
    i_div = 0
    for feature, feature_label, ax in zip(features, feature_labels, axs.flat):
        values = obs[feature].values

        extra_kwargs = {}
        if values.dtype == "category":
            cmap = cmap_discrete
            alpha = 0.7
        else:
            cmap = cmap_divergent if divergent else cmap_continuous
            if divergent:
                vmin = values.min()
                vmax = values.max()
                vcenter = vcenters[i_div]
                i_div += 1
                extra_kwargs["norm"] = DivergentNorm(vmin, vmax, vcenter)
            alpha = 1

        scatter_kwargs = dict(
            s=4,
            cmap=cmap,
            alpha=alpha,
            **extra_kwargs,
        )

        _legend_kwargs = dict(title=feature_label)
        ax = scatter(
            x,
            y,
            ax=ax,
            color_by=feature,
            data=obs,
            legend_kwargs=_legend_kwargs,
            **scatter_kwargs,
        )

        ax.set_aspect("equal")
    fig.suptitle(title, ha="left", x=0)

    return axs


def plot_feature_data_with_reduction(
    adata: AnnData,
    features: Union[str, Sequence[str]],
    reduction: str = "X_pca",
    ncol: int = 2,
    layout="tight",
    cmap_continuous: str = "Blues",
    cmap_discrete: str = "dittoseq",
    cmap_divergent: str = "roma",
    divergent: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    adata = adata.copy()

    x, y = adata.obsm[reduction][:, :2].T
    features = [features] if isinstance(features, str) else features[:]

    fig, axs = configure_subplots(len(features), ncol, layout=layout, sharex=True, sharey=True, figsize=figsize)
    var = adata.var.copy()
    feature_idx = [var.index.get_loc(feature) for feature in features]

    cbar_kwargs = {"loc": "left", "fontsize": 8}

    for feature, ax in zip(features, axs.flat):
        feature_idx = var.index.get_loc(feature)
        feature = adata.var.loc[feature, adata.uns["config"]["secondary_var_names"]]
        values = adata.layers["logcounts"][:, feature_idx].toarray()
        is_discrete = values.dtype == "category"

        if is_discrete:
            norm = None
            cmap = cmap_discrete
            colors = values.astype(int)
            alpha = 0.5
        else:
            cmap = cmap_divergent if divergent else cmap_continuous
            norm = None
            if divergent:
                vmin = values.min()
                vmax = values.max()
                norm = DivergentNorm(vmin, vmax)
            colors = values
            alpha = 1

        scatter_kwargs = dict(s=4, c=colors, cmap=cmap, alpha=alpha, norm=norm)
        scat = ax.scatter(x, y, **scatter_kwargs)
        if is_discrete:
            ax.legend(*ax.collections[0].legend_elements(), bbox_to_anchor=(1.04, 0.5), loc="center left", frameon=False, title=feature)
        else:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(scat, cax=cax, shrink=0.4, label=feature)
            # cbar = fig.colorbar(scat, ax=ax, shrink=0.4)
            # cbar.ax.set_title(feature, **cbar_kwargs)

        ax.set_aspect("equal")
    fig.suptitle(title, ha="left", x=0)

    return axs


def elbow_plot(adata: AnnData, ndims: int = 20, reduction: str = "pca", ax: Optional[Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()
    var_ratio = adata.uns[reduction]["variance_ratio"][:ndims]
    ndims = var_ratio.size
    ax.scatter(np.arange(ndims, dtype=int), var_ratio * 100, s=8, c="k")
    ax.set_xticks(np.arange(0, ndims + 1, 3, dtype=int))
    ax.set_ylabel("Variance explained (%)")
    ax.set_xlabel("PC")
    return ax


@rcDecorator({"axes.edgecolor": "#00000050", "axes.grid.which": "both"})
def plot_pca(adata: AnnData, ndim: int = 5, cmap: str = "tab10", colorby: str = "cluster", figsize=None):

    data = adata.obsm["X_pca"]

    fig, axs, cax = subplots_single_colorbar(ndim, ndim, figsize=figsize)

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
        if ndim > 1:
            scatters_in_row = [ax for i, ax in enumerate(axs[row, :]) if i != row]
            scatters_in_row[0].get_shared_y_axes().join(scatters_in_row[0], *scatters_in_row)
            axs[1, row].get_shared_x_axes().join(axs[1, row], *axs[1:, row])

        for col in range(ndim):
            ax = axs[row, col]

            if row != col:
                ax.scatter(*data[:, (col, row)].T, **scatter_kwargs)
            if row < ndim - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])
            ax.set_frame_on(True)

        axs[0, row].set_xlabel(f"PC{row} ({var_expl[row]:d}%)")
        axs[0, row].xaxis.set_label_position("top")
        axs[row, -1].set_ylabel(f"PC{row} ({var_expl[row]:d}%)", rotation=270, labelpad=15)
        axs[row, -1].yaxis.set_label_position("right")

        density = gaussian_kde(data[:, row])
        xs = np.linspace(data[:, row].min(), data[:, row].max(), 200)
        axs[row, row].plot(xs, density(xs), c="k", linewidth=1)

    if ndim > 1:
        legend_elements = axs.flat[1].collections[0].legend_elements()
        cax.legend(*legend_elements, loc="center", title=colorby, frameon=False)
    else:
        cax.remove()

    fig.tight_layout(pad=0)
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

    ax.contour(X, Y, Z, levels=levels, colors=colors, linewidths=linewidths, origin=origin)

    return ax


def moran_plot(
    adata: AnnData,
    feature: str,
    distances: Union[None, str, np.ndarray],
    color_by: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    contour_kwargs: Optional[Dict[str, Any]] = None,
    # cmap: Optional[str] = None
    **scatter_kwargs,
) -> Axes:
    adata = adata.copy()

    lagged_feature = f"lagged_{feature}"
    if lagged_feature not in adata.obs:
        spatial.compute_spatial_lag(adata, feature, distances, inplace=True)

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
    points = adata.obs[[feature, lagged_feature]].values.T

    ax = scatter(
        x=feature,
        y=lagged_feature,
        color_by=color_by,
        labels=dict(y=y_label),
        rc_context=rc_context,
        fitline_kwargs=dict(color="b"),
        contour_kwargs={"colors": "cyan"},
        data=adata.obs,
        **scatter_kwargs,
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.axvline(adata.obs[feature].mean(), linestyle="--", c="k", alpha=0.5)
    ax.axhline(adata.obs[lagged_feature].mean(), linestyle="--", c="k", alpha=0.5)

    ax.set_aspect("equal")
    return ax



def plot_barcode_histogram(
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
    **hist_kwargs,
) -> np.ndarray[Axes]:

    features = [features] if isinstance(features, str) else features
    nplot = len(features)
    ncol = min(ncol, nplot)
    nrow = int(np.ceil(nplot / ncol))

    if fill_by is not None:
        fig, axs, cax = subplots_single_colorbar(nrow, ncol, figsize=figsize)
        all_feats = features + [fill_by]
        keys, groups = zip(*adata.obs[all_feats].groupby(fill_by).groups.items())

        colormap = plt.get_cmap(cmap)
        colors = [colormap(i) for i in range(len(keys))]

        for feat, ax in zip(features, axs.flat):
            hist_range = np.array([adata.obs[feat].min(), adata.obs[feat].max()])
            hist_mid = hist_range.mean()
            hist_range = tuple((hist_range - hist_mid) * 1.05 + hist_mid)

            # dat = [adata.obs.loc[group, feat].values for group in groups]

            hist_data = [
                np.histogram(
                    adata.obs.loc[group, feat].values,
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
                    range=hist_range,
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
                rects = ax.plot(centers.T, np.maximum(0.2, counts.T))

                if log:
                    ax.set_yscale("log")

                ax.set_ylim(0.2, None)

            ax.grid(False, "minor")
            ax.set_xlabel(feat, size=10)

        if histtype.startswith("step"):
            # Hack to get the actual handles
            handles = [patch[0] for patch in rects]
        else:
            handles = rects
        cax.legend(handles=handles, labels=keys, loc="center left", title=fill_by, frameon=False)
    else:
        fig, axs = configure_subplots(nplot, ncol)
        # TODO

    fig.supylabel("count")

    return axs


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
    **hist_kwargs,
) -> np.ndarray[Axes]:

    features = [features] if isinstance(features, str) else features
    nplot = len(features)
    ncol = min(ncol, nplot)
    nrow = int(np.ceil(nplot / ncol))

    if fill_by is not None:
        fig, axs, cax = subplots_single_colorbar(nrow, ncol, figsize=figsize)
        all_feats = features + [fill_by]
        keys, groups = zip(*adata.obs[all_feats].groupby(fill_by).groups.items())

        colormap = plt.get_cmap(cmap)
        colors = [colormap(i) for i in range(len(keys))]

        for feat, ax in zip(features, axs.flat):
            hist_range = np.array([adata.obs[feat].min(), adata.obs[feat].max()])
            hist_mid = hist_range.mean()
            hist_range = tuple((hist_range - hist_mid) * 1.05 + hist_mid)

            # dat = [adata.obs.loc[group, feat].values for group in groups]

            hist_data = [np.histogram(adata.obs.loc[group, feat].values, bins=bins, range=hist_range, **hist_kwargs) for group in groups]

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
                    **hist_kwargs,
                )
            else:
                ax.set_prop_cycle("color", plt.get_cmap(cmap).colors)

                centers = np.diff(bin_edges) / 2 + bin_edges[:, :-1]
                rects = ax.plot(centers.T, np.maximum(0.2, counts.T))

                if log:
                    ax.set_yscale("log")

                ax.set_ylim(0.2, None)

            ax.grid(False, "minor")
            ax.set_xlabel(feat, size=10)

        if histtype.startswith("step"):
            # Hack to get the actual handles
            handles = [patch[0] for patch in rects]
        else:
            handles = rects
        cax.legend(handles=handles, labels=keys, loc="center left", title=fill_by, frameon=False)
    else:
        fig, axs = configure_subplots(nplot, ncol)

        color = plt.get_cmap(cmap)(7 if cmap is None else 0)
        for ax, feat in zip(axs.flat, features):
            n, _, rects = ax.hist(adata.var[feat], bins=bins, log=log, histtype=histtype, color=color, **hist_kwargs)
            print("sum of bins", n.sum())

    fig.supylabel("count")

    return axs


def plot_fitline(
    x,
    y,
    alternative: str = "two-sided",
    ax: Optional[Axes] = None,
    color: Optional[str] = None,
    data: Any = None,
):

    xlim = ylim = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    if data is not None:
        x = data[x]
        y = data[y]

    non_nans = ~(np.isnan(x) | np.isnan(y))
    reg = linregress(x[non_nans], y[non_nans], alternative=alternative)

    ax.axline((0, reg.intercept), slope=reg.slope, color=color)

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

    is_categorical = False
    if data is None:
        if isinstance(color_by, str):
            raise ValueError("data must not be None if color_by is str")
        if isinstance(x, str):
            raise ValueError("data must not be None if x is str")
        if isinstance(y, str):
            raise ValueError("data must not be None if y is str")

    xdat, xstr = (data[x], x) if isinstance(x, str) else (x, None)
    ydat, ystr = (data[y], y) if isinstance(y, str) else (y, None)
    color_dat, color_str = (data[color_by], color_by) if isinstance(color_by, str) else (color_by, None)

    del color_by, x, y

    # if color_dat is not None and data is not None and data[color_dat].dtype == "category":
    if color_dat is not None and color_dat.dtype == "category":
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
    if legend and is_categorical:
        _legend_kwargs = dict(
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            frameon=False,
            title=color_str,
        )
        legend_kwargs.pop("label", None)
        _legend_kwargs.update(legend_kwargs)

        ax.legend(*scat.legend_elements(), **_legend_kwargs)
    elif legend and color_dat is not None:
        # TODO: compare:
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(scat, cax=cax, shrink=0.4, label=feature_label)
        """
        cmap_kwargs = dict(
            label=legend_kwargs.pop("title", color_str),
            ax=ax,
            cax=None,
        )

        cmap_kwargs.update(legend_kwargs or {})
        cbar_title = cmap_kwargs.pop("title", None)
        cbar = fig.colorbar(scat, **cmap_kwargs)
        if cbar_title is not None:
            cbar.ax.set_title(cbar_title, fontsize=8)
    return ax
