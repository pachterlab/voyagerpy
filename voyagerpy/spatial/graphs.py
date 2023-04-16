from collections import defaultdict
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from anndata import AnnData
from shapely.geometry import Point, Polygon

from voyagerpy import utils
from voyagerpy.spatial import spatial

# from .spatial import get_default_graph


def compute_weights(coords: gpd.GeoSeries, func="euclidean") -> Any:
    if func == "euclidean":
        point_func = shapely.ops.BaseGeometry.distance  # type: ignore
    else:
        point_func = func

    def row_func(point):
        return coords.apply(point_func, other=point)

    return coords.apply(row_func)


def knn(W: np.ndarray, k: int) -> np.ndarray:
    n = W.shape[0]
    nbrs = np.empty((n, k))
    for i, d in enumerate(W):
        idx = np.argpartition(d, k + 1)[k + 1]
        nbrs[i] = idx[idx != i]
    return nbrs


def dnn(
    W: np.ndarray,
    min_dist: float = 0,
    max_dist: float = 1,
    full: bool = True,
    include_low: bool = False,
    include_high: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    pred00 = lambda d: (min_dist < d) & (d < max_dist)  # noqa E713
    pred10 = lambda d: (min_dist <= d) & (d < max_dist)  # noqa E713
    pred01 = lambda d: (min_dist < d) & (d <= max_dist)  # noqa E713
    pred11 = lambda d: (min_dist <= d) & (d <= max_dist)  # noqa E713

    pred = ((pred00, pred01), (pred10, pred11))[include_low][include_high]

    if full:
        return np.apply_along_axis(pred, 0, W)

    ret = [np.where(pred(d))[0] for d in W]
    return ret


def graph2np(g: Any) -> List[List[Tuple[str, float]]]:
    neighbour_list = []

    return neighbour_list


_spatial_nb_methods = ["tri2nb", "knn", "dnn", "gabriel", "relative", "soi", "poly2nb"]
_dist_types = ["none", "idw", "exp", "dpd"]


def compute_visium_dists(adata, filt):
    select = select_obs(adata, filt)

    coords = adata[select].obs[["array_col", "array_row"]] * np.array([1, np.sqrt(3)])
    positions = gpd.GeoSeries(map(Point, coords.values), index=coords.index)
    dists = compute_weights(positions)
    return dists


def select_obs(adata: AnnData, filt: Optional[Tuple[str, Any]]) -> Union[slice, pd.Series]:
    if filt is None:
        return slice(None)

    filt_key, filt_val = filt
    if not isinstance(filt_val, (tuple, list)):
        filt_val = (filt_val,)

    return adata.obs[filt_key].isin(filt_val)


def compute_visium_graph(adata, filt, dist_key: str) -> sp.csr_matrix:
    if dist_key not in adata.obsp:
        select = select_obs(adata, filt)
        dists = compute_visium_dists(adata, filt)

        if filt is not None:
            (idx,) = np.where(select)
            rows = np.tile(idx, len(idx))
            cols = np.repeat(idx, len(idx))
            matrix_data = (dists.values.ravel(), (rows, cols))

            adata.obsp[dist_key] = sp.csr_matrix(matrix_data, shape=(adata.n_obs, adata.n_obs))
        else:
            adata.obsp[dist_key] = sp.csr_matrix(dists.values, shape=(adata.n_obs, adata.n_obs))

    visium_graph = dnn(adata.obsp[dist_key].todense(), max_dist=3, full=True).astype(float)
    return sp.csr_matrix(visium_graph, shape=(adata.n_obs, adata.n_obs))


def find_visium_graph(
    adata: AnnData, filt=("in_tissue", 1), graph_key: str = "visium", dist_key: str = "dists", row_normalize: bool = True
) -> nx.Graph:
    if graph_key not in adata.obsp:
        adata.obsp[graph_key] = compute_visium_graph(adata, filt, dist_key)
        # adata.obsp[graph_key].eliminate_zeros()

    if row_normalize is True:
        adata.obsp[graph_key] = utils.normalize_csr(adata.obsp[graph_key], byrow=True)

    G = nx.Graph()
    nodes_out, nodes_in = np.where(adata.obsp[graph_key].todense())

    # Add node names or mappings
    edges = list(zip(nodes_out, nodes_in))
    G.add_edges_from(edges)
    nx.relabel_nodes(G, dict(enumerate(adata.obs.index)), copy=False)

    positions: pd.DataFrame = spatial.get_spot_coords(adata, tissue=True, as_df=True)
    positions = dict(zip(positions.index, positions.values))
    nx.set_node_attributes(G, positions, "pos")
    return G
