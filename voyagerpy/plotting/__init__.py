#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.pyplot import rcParams

from voyagerpy.plotting.cmap_helper import register_listed_cmap, register_segmented_cmap
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


# https://github.com/thomasp85/scico
register_segmented_cmap("roma", "roma_colors.txt", reverse=True)

# https://github.com/dtm2451/dittoSeq
register_listed_cmap("dittoseq", "ditto_colors.txt", reverse=False)
set_default_cmap("dittoseq")

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
