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
    plot_barcode_histogram,
    plot_barcodes_bin2d,
    plot_bin2d,
    plot_correlogram,
    plot_dim_loadings,
    plot_expression,
    plot_features_bin2d,
    plot_features_histogram,
    plot_fitline,
    plot_local_result,
    plot_moran_mc,
    plot_pca,
    plot_spatial_feature,
    scatter,
    simple_violinplot,
    spatial_reduced_dim,
)

# https://github.com/dtm2451/dittoSeq
from .ditto_colors import ditto_colors

# https://github.com/thomasp85/scico
from .roma_colors import roma_colors

# https://github.com/cran/RColorBrewer
from .blues_colors import blues3, blues4, blues5, blues6, blues7, blues8, blues9


def set_default_cmap(cmap_name: str) -> None:
    rcParams["image.cmap"] = cmap_name


register_segmented_cmap("roma", roma_colors, reverse=True)  # type: ignore
register_segmented_cmap("Blues3", blues3, reverse=False)  # type: ignore
register_segmented_cmap("Blues4", blues4, reverse=False)  # type: ignore
register_segmented_cmap("Blues5", blues5, reverse=False)  # type: ignore
register_segmented_cmap("Blues6", blues6, reverse=False)  # type: ignore
register_segmented_cmap("Blues7", blues7, reverse=False)  # type: ignore
register_segmented_cmap("Blues8", blues8, reverse=False)  # type: ignore
register_segmented_cmap("Blues9", blues9, reverse=False)  # type: ignore


register_listed_cmap("dittoseq", ditto_colors, reverse=False) # type: ignore
set_default_cmap("dittoseq")

__all__ = [
    "elbow_plot",
    "grouped_violinplot",
    "imshow",
    "moran_plot",
    "plot_barcode_data",
    "plot_barcode_histogram",
    "plot_barcodes_bin2d",
    "plot_bin2d",
    "plot_correlogram",
    "plot_dim_loadings",
    "plot_expression",
    "plot_features_bin2d",
    "plot_features_histogram",
    "plot_pca",
    "plot_spatial_feature",
    "simple_violinplot",
    "spatial_reduced_dim",
    "scatter",
    "plot_fitline",
    "plot_local_result",
    "plot_moran_mc",
]
