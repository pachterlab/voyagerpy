#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.pyplot import rcParams

from voyagerpy.plotting.cmap_helper import register_listed_cmap, register_segmented_cmap
from voyagerpy.plotting.plot import (
    elbow_plot,
    grouped_violinplot,
    imshow,
    moran_plot,
    plot_barcode_data,
    plot_barcode_data_with_reduction,
    plot_barcode_histogram,
    plot_barcodes_bin2d,
    plot_bin2d,
    plot_dim_loadings,
    plot_expression,
    plot_features_bin2d,
    plot_feature_data_with_reduction,
    plot_features_histogram,
    plot_pca,
    plot_spatial_feature,
    simple_violinplot,
    spatial_reduced_dim,
    scatter,
    plot_fitline,
    plot_local_result,
)


def set_default_cmap(cmap_name: str) -> None:
    rcParams["image.cmap"] = cmap_name


# https://github.com/thomasp85/scico
from .roma_colors import roma_colors

register_segmented_cmap("roma", roma_colors, reverse=True)

# https://github.com/dtm2451/dittoSeq
from .ditto_colors import ditto_colors

register_listed_cmap("dittoseq", ditto_colors, reverse=False)
set_default_cmap("dittoseq")

__all__ = [
    "elbow_plot",
    "grouped_violinplot",
    "imshow",
    "moran_plot",
    "plot_barcode_data",
    "plot_barcode_data_with_reduction",
    "plot_barcode_histogram",
    "plot_barcodes_bin2d",
    "plot_bin2d",
    "plot_dim_loadings",
    "plot_expression",
    "plot_features_bin2d",
    "plot_feature_data_with_reduction",
    "plot_features_histogram",
    "plot_pca",
    "plot_spatial_feature",
    "simple_violinplot",
    "spatial_reduced_dim",
    "scatter",
    "plot_fitline",
    "plot_local_result",
]
