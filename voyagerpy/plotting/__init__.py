#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import rcParams, rc_context

from voyagerpy.plotting.ditto_colors import register_dittoseq
from voyagerpy.plotting.plot import (
    elbow_plot,
    plot_barcode_data,
    plot_barcodes_bin2d,
    plot_bin2d,
    plot_dim_loadings,
    plot_features_bin2d,
    plot_pca,
    plot_spatial_feature,
    spatial_reduced_dim,
)


def set_default_cmap(cmap_name: str) -> None:
    rcParams["image.cmap"] = cmap_name


def with_colormap(cmap_name: str):
    return rc_context({"image.cmap": cmap_name})


with_cmap = with_colormap

set_default_cmap(register_dittoseq())


__all__ = [
    "elbow_plot",
    "plot_barcodes_bin2d",
    "plot_barcode_data",
    "plot_bin2d",
    "plot_dim_loadings",
    "plot_features_bin2d",
    "plot_pca",
    "plot_spatial_feature",
    "spatial_reduced_dim",
]
