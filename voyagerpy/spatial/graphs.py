from typing import Any, List, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from anndata import AnnData


def compute_weights(coords: gpd.GeoSeries, func='euclidean') -> Any:
    if func == 'euclidean':
        point_func = shapely.ops.BaseGeometry.distance

    else:
        point_func = func

    def func(point):
        return coords.apply(point_func, other=point)

    return coords.apply(func)


def knn(W: np.ndarray, k: int) -> np.ndarray:
    n = W.shape[0]
    nbrs = np.empty((n, k))
    for i, d in enumerate(W):
        idx = np.argpartition(d, k+1)[k+1]
        nbrs[i] = idx[idx != i]
    return nbrs


def dnn(W: np.ndarray, min_dist: float = 0, max_dist: float = 1, full: bool = True, include_low: bool = False, include_high: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
    pred00 = lambda d: (min_dist < d) & (d < max_dist)  # noqa E713
    pred10 = lambda d: (min_dist <= d) & (d < max_dist)  # noqa E713
    pred01 = lambda d: (min_dist < d) & (d <= max_dist)  # noqa E713
    pred11 = lambda d: (min_dist <= d) & (d <= max_dist)  # noqa E713
    
    pred = ((pred00, pred01), (pred10, pred11))[include_low][include_high]

    if full:
        return np.apply_along_axis(pred, 0, W)

    ret = []
    for d in W:
        ret.append(np.where(pred(d))[0])

    return ret

def graph2np(g: Any) -> List[List[Tuple[str, float]]]:
    neighbour_list = []

    return neighbour_list


_spatial_nb_methods = ['tri2nb', 'knn', 'dnn', 'gabriel', 'relative', 'soi', 'poly2nb']
_dist_types = ['none', 'idw', 'exp', 'dpd']


def findVisiumGraph(adata: AnnData) -> Any:

    centroids = gpd.GeoSeries(adata.obs[adata.obs['in_tissue'] == 1]['spot_poly']).centroid
    G = nx.Graph()
    dists = compute_weights(centroids)
    positions = list(zip(centroids.x, centroids.y))
    # TODO: max_dist by scale
    edge_mat = dnn(dists, max_dist=30, full=True)
    edges = list(zip(*np.where(edge_mat)))
    G.add_edges_from(edges)

    g = dnn(coords)
