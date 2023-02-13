from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pathlib
from typing import Tuple


def register_segmented_cmap(cmap_name: str, cmap_colors: Tuple[str], reverse=False) -> None:
    if cmap_name in colormaps:
        return

    if reverse:
        cmap_colors = cmap_colors[::-1]

    cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=len(cmap_colors))
    colormaps.register(cmap)
    colormaps.register(cmap.reversed())


def register_listed_cmap(cmap_name: str, cmap_colors: Tuple[str], reverse: bool = False) -> None:
    if cmap_name in colormaps:
        return

    if reverse:
        cmap_colors = cmap_colors[::-1]

    dittoseq_cmap = ListedColormap(cmap_colors, name=cmap_name, N=len(cmap_colors))
    colormaps.register(dittoseq_cmap)
    colormaps.register(dittoseq_cmap.reversed())
