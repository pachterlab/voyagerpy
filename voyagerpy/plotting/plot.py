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
from pandas import DataFrame, options
from pandas.api.types import is_categorical_dtype
from scipy.stats import gaussian_kde
from scipy import sparse as sp

from voyagerpy import spatial as spt

options.mode.chained_assignment = None  # default='warn'

plt.style.use("ggplot")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "k"
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["axes.edgecolor"] = "#00000050"
plt.rcParams["axes.grid.which"] = "both"


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

    grouped_data = [df.loc[groups[key], y] for key in keys]
    violin_opts = dict(showmeans=False, showextrema=False, showmedians=False, vert=vert)
    violins = ax.violinplot(grouped_data, **violin_opts)
    configure_violins(violins, cmap)

    set_ticks = ax.set_xticks if vert else ax.set_yticks
    set_ticks(np.arange(len(keys)) + 1, labels=keys)

    if not vert:
        x, y = y, x

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if not legend:
        return ax

    colormap = plt.get_cmap(cmap)
    for i, key in enumerate(keys):
        ax.scatter([], [], label=key, color=colormap(i))
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
            ax.set_xlabel(x)
            ax.set_ylabel(y)
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
):
    x_features = x if isinstance(x, (list, tuple)) else [x]
    y_features = y if isinstance(y, (list, tuple)) else [y]

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

    fig, axs_arr = configure_subplots(nplots, ncol, figsize=figsize, sharex=sharex, sharey=sharey)

    feature_iterator = ((y_feat, x_feat) for y_feat in y_features for x_feat in x_features)

    for i_plot, (y_feat, x_feat) in enumerate(feature_iterator):
        i_col = i_plot % ncol
        add_legend = i_col == ncol - 1

        axs_arr.flat[i_plot] = plot_single_barcode_data(
            adata,
            y=y_feat,
            x=x_feat,
            cmap=cmap,
            ax=axs_arr.flat[i_plot],
            legend=add_legend,
            color_by=color_by,
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

    nplots = len(gene_ids)
    fig, axs = configure_subplots(nplots, ncol, sharey=True)

    for ax, gene_id in zip(axs.flat, gene_ids):
        if groupby is not None:
            grouped_violinplot(ax, obs, groupby, gene_id, legend=False)
        else:
            simple_violinplot(ax, obs, gene_id, legend=False)
        title = gene_id
        if show_symbol and secondary_column == "symbol":
            title = adata.var.at[gene_id, secondary_column]
        elif show_symbol:
            title = gene_id

        ax.set_title(title, fontsize=10)
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.supylabel("Expression" + f" ({layer})" if layer is not None else "", fontsize=14)
    if groupby is not None:
        fig.supxlabel(groupby, fontsize=14)
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
    **kwds,
) -> Union[np.ndarray, Any]:

    if isinstance(features, (list, tuple)):
        feat_ls = list(features)
    elif isinstance(features, str):
        feat_ls = [features]
    else:
        raise TypeError("features must be a string or a list of strings")

    assert_basic_spatial_features(adata, errors="raise")

    # copy observation dataframe so we can edit it without changing the inputs
    adata = adata.copy()

    secondary_gene_column = adata.uns["config"]["secondary_var_names"]
    added_features = []

    features_to_pop = []

    var_features = []
    labeled_features = []

    for feature in feat_ls:
        if feature in adata.obs:
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

        raise ValueError(f"Cannot find {feature!r} in adata.obs or gene names")

    # rename the features found via the secondary column
    feat_ls.extend(added_features)
    for feature in features_to_pop:
        feat_ls.remove(feature)

    # Remove duplicates
    for i_feature in range(len(feat_ls) - 1, -1, -1):
        feature = feat_ls[i_feature]
        if feat_ls.count(feature) > 1:
            feat_ls.pop(i_feature)

    # copy observation dataframe so we can edit it without changing the inputs
    # Select the spots to work with
    barcode_selection = slice(None) if not tissue else adata.obs["in_tissue"] == 1
    gene_selection = slice(None) if not var_features else utils.make_unique(var_features)

    # Retain the type of obs
    geo_obs = adata.obs.copy()

    adata = adata[barcode_selection, gene_selection]
    adata = AnnData(
        X=adata.X,
        layers=adata.layers,
        var=adata.var,
        varm=adata.varm,
        varp=adata.varp,
        obs=geo_obs,
        obsm=adata.obsm,
        obsp=adata.obsp,
        dtype=adata.X.dtype,
    )

    obs = adata.obs

    if var_features:
        if layer is None:
            columns = adata[:, var_features].X.toarray()
        else:
            columns = adata[:, var_features].layers[layer].toarray()

        obs[var_features] = columns

    # check if barcode geometry exists
    if barcode_geom is not None:
        if barcode_geom not in obs:
            raise ValueError(f"Cannot find {barcode_geom!r} data in adata.obs")

        # if barcode_geom is not spot polygons, change the default
        # geometry of the observation matrix, so we can plot it
        if barcode_geom != obs.geometry.name:
            obs.set_geometry(barcode_geom)

    # check if features are in rowdata

    n_features = len(feat_ls)
    # Check if too many subplots
    if n_features > 6:
        raise ValueError("Too many features to plot, reduce the number of features")
    ncol = min(ncol, n_features)

    if ncol > 3:
        raise ValueError("Too many columns for subplots, max 3 allowed.")

    nrow = int(ceil(n_features / ncol))
    # only work with spots in tissue
    if tissue:
        obs = obs[obs["in_tissue"] == 1]
    # use a divergent colormap
    if divergent:
        cmap = "Spectral_r"

    subplot_kwds = subplot_kwds or {}
    # create the subplots with right cols and rows

    if _ax is None:
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
    del nrow, ncol

    for ax, feature in zip(axs.flat, feat_ls):
        legend_kwds_ = deepcopy(legend_kwds)
        _legend = legend
        if tissue:
            tissue_selection = adata.obs["in_tissue"] == 1
            # if gene value
            if feature in adata.var.index:
                # col = adata.var[features])
                col = adata[tissue_selection, feature].X.todense().reshape((adata[tissue_selection, :].shape[0])).T
                col = np.array(col.ravel()).T
                obs[feature] = col
            if feature in obs.columns:
                # feat = features
                pass
        else:
            if feature in adata.var.index:
                # col = adata.var[features]
                col = adata[:, feature].X.todense().reshape((adata.shape[0])).T
                obs[feature] = col
            if feature in obs.columns:
                pass

        if feature in adata.var.index or adata.obs[feature].dtype != "category":
            legend_kwds_.setdefault("label", feature)
            legend_kwds_.setdefault("orientation", "vertical")
            legend_kwds_.setdefault("shrink", 0.3)
        else:
            #  colorbar for discrete categories if pandas column is categorical
            _legend = False
            add_colorbar_discrete(ax, fig, cmap, feature, adata.obs[feature].unique().shape[0], list(adata.obs[feature].cat.categories))

        if color is not None:
            cmap = None

        obs.plot(
            feature,
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

    # colorbar title
    if _ax is not None:
        fig = ax.get_figure()

    axs = fig.get_axes()

    # for i in range(len(axs)):
    for ax in axs:
        if ax.properties()["label"] == "<colorbar>" and ax.properties()["ylabel"] != "":
            ax.set_title(ax.properties()["ylabel"], ha="left")
            ax.set_ylabel("")

    return axs  # ,fig


def assert_basic_spatial_features(adata, errors: str = "raise") -> Tuple[bool, str]:

    ret = True
    errors_to_raise = []
    get_geom_prompt = "Consider run voyagerpy.spatial.get_geom(adata) first."

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
        cmap = "Spectral_r"

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


def add_colorbar_discrete(ax, fig, cmap, cbar_title: str, cat_nr: int, cat_names: list, scale: bool = False) -> Colorbar:
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
