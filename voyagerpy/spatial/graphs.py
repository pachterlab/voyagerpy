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


def compute_visium_graph(adata, graph_key: str, inplace: bool = True, force: bool = False) -> sp.csr_matrix:
    from .spatial import get_default_graph

    if graph_key is None:
        graph_key = get_default_graph(adata)

    if graph_key in adata.obsp and not force:
        return adata.obsp[graph_key]

    coord_names = ["array_row", "array_col"]
    max_row, max_col = adata.obs[coord_names].max()
    coords = adata.obs[coord_names].values
    coord2barcode = dict(zip(map(tuple, coords), adata.obs_names))

    rows = []
    cols = []
    index = adata.obs_names
    for row, col in coord2barcode:
        barcode = coord2barcode[(row, col)]
        barcode_idx = index.get_loc(barcode)

        nbrs = [(row, col + 2), (row + 1, col - 1), (row + 1, col + 1)]
        for nbr_row, nbr_col in nbrs:
            nbr_barcode = coord2barcode.get((nbr_row, nbr_col))
            if nbr_barcode is None:
                continue
            nbr_idx = index.get_loc(nbr_barcode)
            rows.extend([barcode_idx, nbr_idx])
            cols.extend([nbr_idx, barcode_idx])
    vals = [1] * len(rows)
    visium_graph = sp.csr_matrix((vals, (rows, cols)), shape=(adata.n_obs, adata.n_obs))

    if inplace:
        adata.obsp[graph_key] = visium_graph

    return visium_graph


def find_visium_graph(
    adata: AnnData,
    subset: Union[None, pd.Series, List[str]] = None,
    graph_key: str = "visium",
    geom: Optional[str] = None,
    row_normalize: bool = True,
    inplace: bool = True,
    force: bool = False,
) -> nx.Graph:
    """Find Visium graph from AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing Visium data.
    subset : list, optional
        Subset of spots to include in graph, by default None
    graph_key : str, optional
        Key in `adata.obsp` to store graph, by default "visium"
    row_normalize : bool, optional
        Whether to row normalize the graph, by default True
    inplace : bool, optional
        Whether to store graph in `adata.obsp`, by default True
    force: bool, optional
        Whether to force recomputation of graph, by default False

    Returns
    -------
    nx.Graph
        NetworkX graph object.
    """

    visium_graph = compute_visium_graph(adata, graph_key, inplace=inplace, force=force)
    if row_normalize is True:
        visium_graph = utils.normalize_csr(visium_graph, byrow=True)

    if inplace is True and graph_key not in adata.obsp:
        adata.obsp[graph_key] = visium_graph

    if subset is None:
        subset = slice(None)

    barcodes = adata[subset].obs_names
    index = adata.obs_names.get_indexer(barcodes)

    adj_mat = visium_graph[index, :][:, index]
    G = nx.Graph(adj_mat)
    nx.relabel_nodes(G, dict(enumerate(barcodes)), copy=False)

    # Get positions of nodes
    geo = adata.obsm["geometry"]
    if not isinstance(geo, gpd.GeoDataFrame):
        geo = gpd.GeoDataFrame(geo, geometry=geo.columns[0])
    if geom is None or geom not in geo.columns:
        points = geo.geometry.centroid
    else:
        points = geo[geom].centroid

    geo_subset = pd.DataFrame({"x": points.x, "y": points.y}, index=geo.index).loc[subset]
    pos_dict = dict(zip(geo_subset.index, geo_subset[["x", "y"]].values))

    nx.set_node_attributes(G, pos_dict, "pos")
    return G
