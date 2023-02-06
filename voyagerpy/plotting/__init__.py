#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.pyplot import rc_context, rcParams

from voyagerpy.plotting.ditto_colors import register_dittoseq
from voyagerpy.plotting.plot import (
    elbow_plot,
    grouped_violinplot,
    imshow,
    plot_barcode_data,
    plot_barcodes_bin2d,
    plot_bin2d,
    plot_dim_loadings,
    plot_expression,
    plot_features_bin2d,
    plot_pca,
    plot_spatial_feature,
    simple_violinplot,
    spatial_reduced_dim,
)


def set_default_cmap(cmap_name: str) -> None:
    rcParams["image.cmap"] = cmap_name


set_default_cmap(register_dittoseq())


__all__ = [
    "elbow_plot",
    "grouped_violinplot",
    "imshow",
    "plot_barcode_data",
    "plot_barcodes_bin2d",
    "plot_bin2d",
    "plot_dim_loadings",
    "plot_expression",
    "plot_features_bin2d",
    "plot_pca",
    "plot_spatial_feature",
    "simple_violinplot",
    "spatial_reduced_dim",
]
