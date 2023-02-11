from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pathlib


def register_segmented_cmap(cmap_name: str, filename: str, reverse=False) -> None:
    if cmap_name in colormaps:
        return

    folder = pathlib.Path(__file__).parent
    fn = folder / filename

    cmap_colors = list(map(str.strip, fn.open().readlines()))
    if reverse:
        cmap_colors.reverse()

    cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=len(cmap_colors))
    colormaps.register(cmap)
    colormaps.register(cmap.reversed())


def register_listed_cmap(cmap_name: str, filename: str, reverse: bool = False) -> None:
    if cmap_name in colormaps:
        return
    fn = pathlib.Path(__file__).parent / filename
    cmap_colors = list(map(str.strip, fn.open().readlines()))

    if reverse:
        cmap_colors.reverse()

    dittoseq_cmap = ListedColormap(cmap_colors, name=cmap_name, N=len(cmap_colors))
    colormaps.register(dittoseq_cmap)
    colormaps.register(dittoseq_cmap.reversed())
