from typing import Tuple

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize


def register_segmented_cmap(cmap_name: str, cmap_colors: Tuple[str], reverse=False) -> None:
    if cmap_name in colormaps:
        return

    if reverse:
        cmap_colors = cmap_colors[::-1]

    # cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=len(cmap_colors))
    cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors)
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


class DivergentNorm(Normalize):
    def __init__(self, vmin, vmax, vcenter=0, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)

        # The largest interval defines the slope
        dx = max(abs(vmax - vcenter), abs(vcenter - vmin))

        # The y difference of the dx is always 0.5
        dy = 0.5
        slope = dy / dx
        intercept = 0.5 - slope * vcenter

        f = lambda x: slope * x + intercept
        self.x = [self.vmin, self.vmax]
        self.y = [f(self.vmin), f(self.vmax)]

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.interp(value, self.x, self.y))

    def inverse(self, value):
        return np.ma.masked_array(np.interp(value, [0, 1], self.x, left=self.vmin, right=self.vmax))
