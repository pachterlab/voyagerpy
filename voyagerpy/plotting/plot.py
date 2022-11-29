#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:08:42 2022

@author: sinant
"""


from math import ceil
from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import geopandas as gpd
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from voyagerpy import spatial as spt


def plot_spatial_features(
    adata: AnnData,
    features: Union[str, Sequence[str]],
    ncol: Optional[int] = None,
    barcode_geom: Optional[str] = None,
    annot_geom: Optional[str] = None,
    tissue: bool = True,
    colorbar: bool = False,
    color: Optional[str] = None,
    cmap: Optional[str] = "Blues",
    geom_style: Optional[dict] = {},
    annot_style: Optional[dict] = {},
    ax: Optional[Axes] = None,
    subplot_kwds: Optional[dict] = {},
    legend_kwds: Optional[dict] = {},
    **kwds,
) -> Tuple[Union[np.ndarray, Any], Figure]:
    sns.set_theme()

    # check input
    if ("geometry" not in adata.obs) or "geom" not in adata.uns["spatial"]:
        adata = spt.get_geom(adata)
    for i in features:
        if i not in adata.obs and i not in adata.var.index:
            raise ValueError(f"Cannot find {i!r} in adata.obs or gene names")
    # copy observation dataframe so we can edit it without changing the inputs
    obs = adata.obs

    # check if barcode geometry exists
    if barcode_geom is not None:
        if barcode_geom not in obs:
            raise ValueError(f"Cannot find {barcode_geom!r} data in adata.obs")

        # if barcode_geom is not spot polygons, change the default geometry of the observation matrix, so we can plot it
        if barcode_geom != "spot_poly":
            obs.set_geometry(barcode_geom)

    # check if features are in rowdata
    feat_ls = []
    if isinstance(features, list):
        feat_ls = features
    if isinstance(features, str):
        feat_ls = [features]

    # Check if too many subplots
    if len(feat_ls) > 6:
        raise ValueError("Too many features to plot, reduce the number of features")
    if ncol is not None:
        if ncol > 3:
            raise ValueError("Too many columns for subplots")

    # only work with spots in tissue
    if tissue is True:
        obs = obs[obs["in_tissue"] == 1]

    # create the subplots with right cols and rows
    if ax is None:
        plt_nr = len(feat_ls)
        nrows = 1
        # ncols = ncol if ncol is not None else 1

        # TODO: are ncol and ncols supposed to be the same variable?
        # defaults
        if ncol is None:
            if plt_nr < 4:
                ncols = plt_nr
            if plt_nr >= 4:
                nrows = 2
                ncols = 3
        else:
            nrows = ceil(plt_nr / ncols)

        # if(subplot_kwds is None):
        #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10,10))
        if subplot_kwds is None:
            subplot_kwds = {}
        if "figsize" in subplot_kwds:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **subplot_kwds)
        else:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10), **subplot_kwds)

        plt.subplots_adjust(wspace=1 / ncols + 0.2)

    # iterate over features to plot
    x = 0
    y = 0

    for i in range(len(feat_ls)):
        # TODO: deep copy?
        legend_kwds_ = legend_kwds or {}

        if tissue is True:

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
                col = adata[:, feat_ls[i]].X.todense().reshape((adata.shape[0]))
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

        # correct legend if feature is categorical and make sure title is in there
        if len(legend_kwds_) == 0:

            if feat_ls[i] in adata.var.index or adata.obs[feat_ls[i]].dtype != "category":
                legend_kwds_ = {"label": feat_ls[i], "orientation": "vertical", "shrink": 0.3}
            else:
                legend_kwds_ = {"title": feat_ls[i]}
        else:
            if feat_ls[i] in adata.var.index or adata.obs[feat_ls[i]].dtype != "category":
                legend_kwds_.setdefault("label", feat_ls[i])
                legend_kwds_.setdefault("orientation", "vertical")
                legend_kwds_.setdefault("shrink", 0.3)
            else:

                legend_kwds_.setdefault("title", feat_ls[i])

        if color is not None:
            cmap = None
        # change default behaviour in seaborn and have no color for edges of polygons
        if geom_style is None:
            geom_style = {}
        geom_style.setdefault("edgecolor", "none")

        obs.plot(feat_ls[i], ax=ax, color=color, legend=True, cmap=cmap, legend_kwds=legend_kwds_, **geom_style, **kwds)  # type: ignore
        if annot_geom is not None:
            if annot_geom in adata.uns["spatial"]["geom"]:

                # check annot_style is dict with correct values
                plg = adata.uns["spatial"]["geom"][annot_geom]
                if annot_style is not None:
                    gpd.GeoSeries(plg).plot(ax=ax, **annot_style, **kwds)
                else:

                    gpd.GeoSeries(plg).plot(color="blue", ax=ax, alpha=0.2, **kwds)
            else:
                raise ValueError(f"Cannot find {annot_geom!r} data in adata.uns['spatial']['geom']")

            pass

        y = y + 1
        if y >= ncols:
            y = 0
            x = x + 1

    # TODO: What if fig is None?
    # colorbar title
    axs = fig.get_axes()
    for i in range(len(axs)):
        if axs[i].properties()["label"] == "<colorbar>":

            axs[i].set_title(axs[i].properties()["ylabel"], ha="left")
            axs[i].set_ylabel("")

    if ax is None:
        return axs, fig
    else:
        return ax, fig
