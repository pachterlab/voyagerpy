#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import rcParams

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

ditto_colors = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#666666",
    "#AD7700",
    "#1C91D4",
    "#007756",
    "#D5C711",
    "#005685",
    "#A04700",
    "#B14380",
    "#4D4D4D",
    "#FFBE2D",
    "#80C7EF",
    "#00F6B3",
    "#F4EB71",
    "#06A5FF",
    "#FF8320",
    "#D99BBD",
    "#8C8C8C",
    "#FFCB57",
    "#9AD2F2",
    "#2CFFC6",
    "#F6EF8E",
    "#38B7FF",
    "#FF9B4D",
    "#E0AFCA",
    "#A3A3A3",
    "#8A5F00",
    "#1674A9",
    "#005F45",
    "#AA9F0D",
    "#00446B",
    "#803800",
    "#8D3666",
    "#3D3D3D",
]

dittoseq_name = "dittoseq"
if dittoseq_name not in colormaps:
    dittoseq_cmap = ListedColormap(ditto_colors, name=dittoseq_name)
    colormaps.register(dittoseq_cmap)
    colormaps.register(dittoseq_cmap.reversed())
    rcParams["image.cmap"] = dittoseq_name

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
