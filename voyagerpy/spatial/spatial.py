#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from cv2 import (
    CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY,
    COLOR_RGB2BGR,
    RETR_TREE,
    arcLength,
    contourArea,
    cvtColor,
    findContours,
    pointPolygonTest,
    threshold,
)
from scipy import sparse
from shapely.geometry import Point, Polygon

from voyagerpy import utils as utl


# create spatial functions with shapely
def get_approx_tissue_boundary(adata: AnnData, size: str = "hires", paddingx: int = 0, paddingy: int = 0) -> Tuple[int, int, int, int]:
    if size == "hires":
        scl = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
    else:
        scl = adata.uns["spatial"]["scale"]["tissue_lowres_scalef"]
    bot = int(np.max(adata.obs[adata.obs["in_tissue"] == 1]["pxl_row_in_fullres"]) * scl)
    top = int(np.min(adata.obs[adata.obs["in_tissue"] == 1]["pxl_row_in_fullres"]) * scl)
    right = int(np.max(adata.obs[adata.obs["in_tissue"] == 1]["pxl_col_in_fullres"]) * scl)
    left = int(np.min(adata.obs[adata.obs["in_tissue"] == 1]["pxl_col_in_fullres"]) * scl)
    if paddingx != 0:
        left = left - paddingx
        right = right + paddingx
    if paddingy != 0:
        top = top - paddingy
        bot = bot + paddingy

    return top, bot, left, right


Contour = Any
# %%


## write a docstring for this function
def get_tissue_contour_score(cntr: Contour, adata: AnnData, size: str = "hires") -> float:
    # docstring
    """
    This function takes a contour and returns a score for how well it fits the tissue

    Parameters
    ----------
    cntr : Contour
        The contour to score, represented as a list of points from a cv2 findContours function
    adata: AnnData object describing the voyager dataset
    size: str = "hires" or "lowres", the resolution of the image to use

    Returns
    -------
    float the score of the contour
    """
    scl = utl.get_scale(adata, res=size)

    # tissue_barcodes = adata.obs[adata.obs["in_tissue"] == 1]
    # non_tissue_barcodes  = adata.obs[adata.obs["in_tissue"] != 1]
    # total = tissue_barcodes.shape[0]

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(adata.obs.shape[0]):
        # print([int(tissue_barcodes.iloc[i,3]*0.2),int(tissue_barcodes.iloc[i,4]*0.2)])
        # print(cv2.pointPolygonTest(big_cntrs[0], (int(tissue_barcodes.iloc[i,4]*0.2),int(tissue_barcodes.iloc[i,3]*0.2)), False) )
        test_pt = (int(adata.obs["pxl_col_in_fullres"][i] * scl), int(adata.obs["pxl_row_in_fullres"][i] * scl))
        polytest = pointPolygonTest(cntr, test_pt, False)
        if polytest == 1:
            if adata.obs["in_tissue"][i] == 1:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if adata.obs["in_tissue"][i] == 1:
                fn = fn + 1
            else:
                tn = tn + 1

    # method youden j....whynot
    J = (tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)) - 1
    return J


def detect_tissue_threshold(adata: AnnData, size: str = "hires", low: int = 200, high: int = 255) -> Tuple[int, Optional[Contour]]:
    bgr_img = cvtColor(adata.uns["spatial"]["img"][size], COLOR_RGB2BGR)
    bgr_img = (bgr_img * 255).astype("uint8")  # type: ignore
    imgray = cvtColor(bgr_img, COLOR_BGR2GRAY)
    px_thrsh = low
    thrsh_score_all = 0
    best_thrsh = 0
    best_cntr_all = None
    for i in range(px_thrsh, high):
        ret, thresh = threshold(imgray, i, 255, 0)
        # contours
        contours, contours2 = findContours(thresh, RETR_TREE, CHAIN_APPROX_SIMPLE)
        # filter contours by size

        big_cntrs = []
        # marked = bgr_img.copy();
        for contour in contours:
            area = contourArea(contour)
            if area > 10000:
                # print(area);
                big_cntrs.append(contour)
        # print(len(big_cntrs))
        score = 0
        best_cntr = None
        for j in range(len(big_cntrs)):
            new_score = get_tissue_contour_score(big_cntrs[j], adata, size=size)
            if new_score > score:
                best_cntr = big_cntrs[j]
                score = new_score

        if score > thrsh_score_all:
            # print("score is " ,thrsh_score_all)
            if best_cntr_all is not None:
                # print("ratio is " ,cv2.arcLength(best_cntr_all, True)/ cv2.arcLength(best_cntr, True))

                if arcLength(best_cntr_all, True) / arcLength(best_cntr, True) < 0.9:
                    if abs(thrsh_score_all - score) < 0.1:
                        break

            best_thrsh = i
            best_cntr_all = best_cntr
            thrsh_score_all = score
            # print("score is " ,thrsh_score_all)
            # if(best_cntr_all is not None):

            #    print(cv2.arcLength(best_cntr_all, True)/cv2.arcLength(best_cntr, True))

    return best_thrsh, best_cntr_all


def get_tissue_boundary(
    adata: AnnData,
    threshold_low: Optional[int] = None,
    size: Optional[str] = "hires",
    strictness: Optional[int] = None,
    inplace: bool = False,
    # detect_threshold: bool = False,
) -> Polygon:
    if size not in (None, "lowres", "hires"):
        raise ValueError('Expected size to be one of None, "lowres", or "hires", but got `{size}`')

    if size is None:
        res = "hires" if utl.is_highres(adata) else "lowres"
    else:
        res = size

    scl = utl.get_scale(adata, res=res)
    # load image

    bgr_img = cvtColor(adata.uns["spatial"]["img"][res], COLOR_RGB2BGR)
    bgr_img = (bgr_img * 255).astype("uint8")  # type: ignore

    # rescale
    # scale = 0.25
    # h, w = img.shape[:2]
    # h = int(h*scale)
    # w = int(w*scale)
    # img = cv2.resize(img, (w,h))

    # hsv
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    # h, s, v = cv2.split(hsv);

    # thresh
    # thresh = cv2.inRange(h, 140, 179);
    if threshold_low is not None:
        imgray = cvtColor(bgr_img, COLOR_BGR2GRAY)
        ret, thresh = threshold(imgray, threshold_low, 255, 0)
        # contours
        contours, contours2 = findContours(thresh, RETR_TREE, CHAIN_APPROX_SIMPLE)
        # filter contours by size
        big_cntrs = []
        # marked = bgr_img.copy();
        for contour in contours:
            area = contourArea(contour)
            if area > 10000:
                # print(area);
                big_cntrs.append(contour)
        # for all contours check if all points are within countour and all points outside it
        score = 0
        best_cntr: Optional[np.ndarray] = None
        for i in range(len(big_cntrs)):
            new_score = get_tissue_contour_score(big_cntrs[i], adata)
            if new_score > score:
                score = new_score
                best_cntr = big_cntrs[i]
    else:
        thrsh, best_cntr = detect_tissue_threshold(adata, size=size)

        # not_tissue_barcodes = adata.obs[adata.obs["in_tissue"] == 0]

        # ts_out_p = []

    # if(strictness == "strict"):

    # cv2.drawContours(marked, big_cntrs, -1, (0, 255, 0), 3);

    # # create a mask of the contoured image
    # mask = np.zeros_like(imgray);
    # mask = cv2.drawContours(mask, big_cntrs, -1, 255, -1);

    # # crop out
    # out = np.zeros_like(bgr_img) # Extract out the object and place into output image
    # out[mask == 255] = bgr_img[mask == 255];

    # if(plot):
    # # show
    #     cv2.imshow("Original", brg_img);
    #     cv2.imshow("thresh", thresh);
    #     cv2.imshow("Marked", marked);
    #     cv2.imshow("out", out);

    #     cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    assert best_cntr is not None

    contour = np.squeeze(best_cntr)
    polygon = Polygon(contour)
    # print(polygon.wkt)

    return polygon  # ,out

    # create outline of tissue sample


def set_geometry(
    adata: AnnData,
    geom: str,
    values: Optional[gpd.GeoSeries] = None,
    index: Optional[pd.Index] = None,
    dim: Union[str, Literal["barcode", "gene"]] = "barcode",
    inplace: bool = True,
) -> AnnData:
    if not inplace:
        adata = adata.copy()

    if dim == "barcode":
        geo = adata.obsm.setdefault("geometry", gpd.GeoDataFrame(index=adata.obs_names))
        if not isinstance(geo, gpd.GeoDataFrame):
            adata.obsm["geometry"] = gpd.GeoDataFrame(geo)
        geo_df: gpd.GeoDataFrame = adata.obsm["geometry"]  # type: ignore
    elif dim == "gene":
        geo = adata.varm.setdefault("geometry", gpd.GeoDataFrame(index=adata.var_names))
        if not isinstance(geo, gpd.GeoDataFrame):
            adata.varm["geometry"] = gpd.GeoDataFrame(geo)
        geo_df: gpd.GeoDataFrame = adata.varm["geometry"]  # type: ignore
    else:
        geo_dict = adata.uns["spatial"].setdefault("geometry", {})
        geo = geo_dict.setdefault(dim, gpd.GeoDataFrame(columns=[geom], index=index))
        if not isinstance(geo, gpd.GeoDataFrame):
            geo_dict[dim] = gpd.GeoDataFrame(geo)
        geo_df: gpd.GeoDataFrame = geo_dict[dim]

    if geom not in geo_df and values is None:
        raise ValueError("values must not be None when geom does not exist in the DataFrame")
    if values is not None:
        if sorted(values.index) == list(range(adata.n_obs)):
            values.index = adata.obs_names[values.index]

        geo_df[geom] = values

    geo_df.set_geometry(geom, inplace=True)

    return adata


def to_points(
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    data: Union[gpd.GeoDataFrame, pd.DataFrame, None] = None,
    scale: float = 1,
    radius: Optional[float] = None,
) -> gpd.GeoSeries:
    if data is None and (isinstance(x, str) or isinstance(y, str)):
        raise ValueError("data must not be None if either x or y is str")
    xdat = data[x] if isinstance(x, str) else x
    ydat = data[y] if isinstance(y, str) else y

    points = gpd.GeoSeries.from_xy(xdat, ydat).scale(scale, scale, origin=(0, 0))
    if radius:
        return points.buffer(radius)
    return points


def get_geom(adata: AnnData, threshold: Optional[int] = None, inplace: bool = False, res: Optional[str] = None) -> AnnData:
    if not inplace:
        adata = adata.copy()

    geom = adata.uns["spatial"].setdefault("geom", {})

    # Create a geometry column from x & ly
    scale = utl.get_scale(adata, res=res)
    spot_diam = adata.uns["spatial"]["scale"]["spot_diameter_fullres"]

    type_converter = {"float32": np.float32, "float64": np.float64, "float": np.float32}
    dtype_cast = type_converter.get(str(adata.X.dtype), np.float64)

    if False:

        def to_point(x) -> Point:
            return Point(
                dtype_cast(x.pxl_col_in_fullres * scale),
                dtype_cast(x.pxl_row_in_fullres * scale),
            ).buffer((spot_diam / 2) * 0.2)

        geometry_name: str = "spot_poly"
        if geometry_name not in adata.obs:
            # add spot points to geom
            adata.obs[geometry_name] = adata.obs.apply(to_point, axis=1)

        if not isinstance(adata.obs, gpd.GeoDataFrame):
            # Create a GeoDataFrame from adata.obs
            adata.obs = gpd.GeoDataFrame(
                adata.obs,
                geometry=geometry_name,
            )

    tissue_poly = geom.get("tissue_poly", None)
    boundary = geom.get("tissue_boundary", None)

    if not isinstance(tissue_poly, Polygon):
        # add boundary and tissue poly to geom
        tissue_poly = get_tissue_boundary(adata, threshold, size=res)
        geom["tissue_poly"] = tissue_poly
    if not isinstance(boundary, gpd.GeoSeries):
        geom["tissue_boundary"] = gpd.GeoSeries(tissue_poly).boundary

    return adata


def get_spot_coords(
    adata: AnnData,
    tissue: bool = True,
    as_tuple: bool = True,
    as_df: bool = False,
    res: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
    h_sc = utl.get_scale(adata, res)
    cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    if tissue:
        coords = adata.obs.loc[adata.obs["in_tissue"] == 1, cols] * h_sc
    else:
        coords = adata.obs.loc[:, cols] * h_sc

    if as_df:
        return coords
    coords = coords.values
    # if utl.is_highres(adata):
    #     h_sc = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
    # else:
    #     h_sc = adata.uns["spatial"]["scale"]["tissue_lowes_scalef"]
    # if tissue:
    #     return np.array(
    #         [h_sc * adata.obs[adata.obs["in_tissue"] == 1].iloc[:, 4], h_sc * adata.obs[adata.obs["in_tissue"] == 1].iloc[:, 3]]
    #     )
    # else:
    #     return np.array(h_sc * adata.obs.iloc[:, 4]), np.array(h_sc * adata.obs.iloc[:, 3])

    return (coords[:, 0], coords[:, 1]) if as_tuple else coords


def cancel_transforms(adata: AnnData) -> None:
    spatial_dict = adata.uns["spatial"]
    transforms = spatial_dict.get("transform", ([], []))
    pxl_coord_cols = ["pxl_col_in_fullres_tmp", "pxl_row_in_fullres_tmp"]

    transforms[1].clear()
    spatial_dict.pop("img_tmp", None)
    adata.obs.drop(pxl_coord_cols, axis="columns", inplace=True, errors="ignore")


def apply_transforms(adata) -> None:
    spatial_dict = adata.uns["spatial"]
    transforms = spatial_dict.get("transform", ([], []))
    pxl_coord_cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    pxl_coord_cols_tmp = [f"{colname}_tmp" for colname in pxl_coord_cols]

    if "img_tmp" in spatial_dict:
        spatial_dict["img"] = spatial_dict["img_tmp"]
        transforms[0].extend(transforms[1])

        if all(colname in adata.obs for colname in pxl_coord_cols_tmp):
            adata.obs[pxl_coord_cols] = adata.obs[pxl_coord_cols_tmp]

    # cleanup tmp image, coords and transform
    transforms[1].clear()
    spatial_dict.pop("img_tmp", None)
    adata.obs.drop(pxl_coord_cols_tmp, axis="columns", inplace=True, errors="ignore")


def _rotate_coordinate_system(img: np.ndarray, coords: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    rotation_mats = rotation_mats = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]]),
    ]
    img = np.rot90(img, k=k)
    n_rows, n_cols = img.shape[:2]

    # rotate all spot coordinates
    center = np.array([n_rows, n_cols]) / 2

    # we rotate around the center of the image:
    # 1. translate to origin
    # 2. rotate
    coords = np.matmul(coords - center, rotation_mats[k])

    # 3. translate back, maybe transposing the center
    coords += center[::-1] if k % 2 else center
    return img, coords


def _mirror_coordinate_system(img, coords, axis):
    n_rows, n_cols = img.shape[:2]

    if axis % 2 == 0:
        img = img[::-1, ...]
        coords[:, 1] = n_rows - 1 - coords[:, 1]

    if axis > 0:
        img = img[:, ::-1, ...]
        coords[:, 0] = n_cols - 1 - coords[:, 0]

    return img, coords


def get_transformation_function(which: str):
    def mirror_param_eval(k: Optional[int] = None, axis: Optional[int] = None) -> int:
        axis = axis or 0
        if axis not in range(2):
            raise ValueError("Invalid mirror axis, must be either 0 or 1")
        return axis

    def rotate_param_eval(k: Optional[int] = None, axis: Optional[int] = None) -> int:
        k = k or 1
        return k % 4

    if which == "mirror":
        param_evaluator = mirror_param_eval
        inner_transformer = _mirror_coordinate_system
    elif which == "rotate":
        param_evaluator = rotate_param_eval
        inner_transformer = _rotate_coordinate_system
    else:
        raise ValueError('`which` must be either "rotate" and "mirror"')

    def inner(adata: AnnData, apply: bool = True, k: Optional[int] = None, axis: Optional[int] = None):
        param = param_evaluator(k, axis)
        del k, axis
        imgs_ret = {}

        spatial_dict = adata.uns["spatial"]
        spatial_dict.setdefault("transform", ([], []))

        # Determine where to fetch and store the data
        img_key_fetch = "img" if (apply or "img_tmp" not in spatial_dict) else "img_tmp"
        img_key_store = "img" if apply else "img_tmp"

        if not apply:
            spatial_dict.setdefault("img_tmp", {})

        pxl_coord_cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        pxl_coord_cols_tmp = [f"{colname}_tmp" for colname in pxl_coord_cols]

        pxl_coord_cols_fetch = pxl_coord_cols
        pxl_coord_cols_store = pxl_coord_cols if apply else pxl_coord_cols_tmp

        if not apply and all(col in adata.obs for col in pxl_coord_cols_tmp):
            pxl_coord_cols_fetch = pxl_coord_cols_tmp

        spot_coords = adata.obs.loc[:, pxl_coord_cols_fetch].values
        transforms = spatial_dict["transform"]

        res_keys = spatial_dict[img_key_fetch].keys()

        for res in res_keys:
            scale = utl.get_scale(adata, res)

            img = spatial_dict[img_key_fetch][res]
            img, coords = inner_transformer(img, spot_coords * scale, param)
            coords = (coords / scale).astype(int)

            adata.obs[pxl_coord_cols_store] = coords
            spatial_dict[img_key_store][res] = img.copy()
            imgs_ret[res] = img

        transforms[not apply].append((which, param))

        if apply:
            # Cleanup tmp space
            transforms[1].clear()
            spatial_dict.pop("img_tmp", None)
            adata.obs.drop(pxl_coord_cols_tmp, inplace=True, errors="ignore")

        return imgs_ret

    return inner


# %%

rotate_img90 = get_transformation_function("rotate")
mirror_img = get_transformation_function("mirror")


def rollback_transforms(adata: AnnData, apply: bool = True):
    spatial_dict = adata.uns["spatial"]
    transforms = spatial_dict.get("transform", ([], []))
    if len(transforms[0]) == 0:
        return

    res_keys = spatial_dict["img"].keys()
    pxl_coord_cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    pxl_coord_cols_tmp = [f"{colname}_tmp" for colname in pxl_coord_cols]

    saved_transform = []
    if not apply:
        # Save applied state in the tmp space
        spatial_dict.setdefault("img_tmp", {})
        for res in res_keys:
            spatial_dict["img_tmp"][res] = spatial_dict["img"][res].copy()
        adata.obs[pxl_coord_cols_tmp] = adata.obs[pxl_coord_cols]
        saved_transform.extend(transforms[0])
    else:
        adata.obs.drop(pxl_coord_cols_tmp, axis="columns", inplace=True, errors="ignore")
        spatial_dict.pop("img_tmp", None)

    rollback = transforms[0][::-1]
    for transform, param in rollback:
        if transform == "mirror":
            mirror_img(adata, axis=param, apply=apply)
        else:
            rotate_img90(adata, k=-param, apply=apply)

    transforms[1].clear()
    transforms[1].extend(saved_transform)

    if not apply:
        spatial_dict["img"], spatial_dict["img_tmp"] = spatial_dict["img_tmp"], spatial_dict["img"]
        coords = adata.obs[pxl_coord_cols]
        adata.obs[pxl_coord_cols] = adata.obs[pxl_coord_cols_tmp]
        adata.obs[pxl_coord_cols_tmp] = coords

    transforms[0].clear()


# %%


def to_spatial_weights(adata: AnnData, graph_name: Optional[str] = None):
    import libpysal

    distances = adata.obsp[graph_name or get_default_graph(adata)].copy()
    if sparse.issparse(distances):
        distances = distances.A
    elif isinstance(distances, np.matrix):
        distances = distances.A

    assert isinstance(distances, np.ndarray)

    focal, neighbors = np.where(distances > 0)
    idx = adata.obs_names

    graph_df = pd.DataFrame(
        {
            "focal": idx[focal],
            "neighbor": idx[neighbors],
            "weight": distances[focal, neighbors],
        }
    )
    W = libpysal.weights.W.from_adjlist(graph_df)
    W.set_transform("r")

    adata.uns.setdefault("spatial", {})
    adata.uns["spatial"][graph_name] = W
    return adata


def compute_spatial_lag(
    adata: AnnData,
    feature: str,
    graph_name: Union[None, str, np.ndarray] = None,
    inplace: bool = False,
    layer: Optional[str] = None,
) -> AnnData:
    if not inplace:
        adata = adata.copy()

    if graph_name is None:
        graph_name = get_default_graph(adata)

    if isinstance(graph_name, str):
        dists = adata.obsp[graph_name]
    elif isinstance(graph_name, np.ndarray):
        dists = graph_name
    else:
        raise TypeError("Distances should be of type None, str, or np.ndarray.")
    del graph_name

    if sparse.issparse(dists) or isinstance(dists, np.matrix):
        dists = dists.A

    features = [feature] if isinstance(feature, str) else feature[:]
    X = adata.X if layer is None else adata.layers[layer].A
    for feat in features:
        lagged_feat = f"lagged_{feat}"
        if feat in adata.var_names:
            x = X[:, adata.var_names.get_loc(feat)]
        else:
            x = adata.obs[feat]

        adata.obs[lagged_feat] = dists.dot(x)

    return adata


def moran(
    adata: AnnData,
    feature: Union[str, Sequence[str]],
    graph_name: Optional[str] = None,
    layer: Optional[str] = None,
    dim: Literal["obs", "var"] = "obs",
    permutations: int = 0,
):
    # import esda
    ## write code to check if esda is installed, otherwise complain
    try:
        import esda
    except ImportError:
        raise ImportError(
            "Moran's I requires the `esda` package. Please install it with `pip install esda`."
        )

    if graph_name is None:
        graph_name = get_default_graph(adata)

    if graph_name not in adata.uns.get("spatial", {}):
        to_spatial_weights(adata, graph_name)

    W = adata.uns["spatial"][graph_name]

    features = [feature] if isinstance(feature, str) else list(feature[:])

    if dim == "obs":
        morans = [
            esda.Moran(adata.obs[feat], W, permutations=permutations)
            for feat in features
        ]
    elif dim == "var":
        feat_idx = list(map(adata.var.index.get_loc, features))
        X = adata.X.A if layer is None else adata.layers[layer].A
        morans = [esda.Moran(X[:, i], W, permutations=permutations) for i in feat_idx]
    else:
        raise ValueError('dim must either be "obs" or "var"')

    moran_dict = adata.uns["spatial"].setdefault("moran", {})
    df = moran_dict.setdefault(
        graph_name, pd.DataFrame(columns=["I", "EI"], dtype=("double", "double"))
    )

    for feat, moran in zip(features, morans):
        df.at[feat, "I"] = moran.I
        df.at[feat, "EI"] = moran.EI

    if permutations > 0:
        moran_sims = adata.uns["spatial"].setdefault("moran_mc", {})
        sims_dict = adata.uns["spatial"]["moran_mc"].setdefault(graph_name, {})
        for feat, moran in zip(features, morans):
            df = sims_dict.setdefault(feat, pd.DataFrame(columns=["sim", "p_sim"]))
            df["sim"] = moran.sim
            df["p_sim"] = moran.p_sim

    # TODO: What do we want to return?
    # return morans[0] if isinstance(feature, str) else morans


def set_default_graph(adata: AnnData, graph_name: str) -> None:
    adata.uns.setdefault("spatial", {})
    adata.uns["spatial"]["default_graph"] = graph_name


def get_default_graph(adata: AnnData) -> str:
    return adata.uns.get("spatial", {}).get("default_graph", "connectivities")


def losh(
    adata: AnnData,
    feature: Union[str, Sequence[str]],
    graph_name: Optional[str] = None,
    inference: Union[None, Literal["permutation"], Literal["chi-square"]] = None,
    inplace: bool = True,
    key_added: str = "losh",
    layer: Optional[str] = None,
) -> AnnData:
    import esda

    if not inplace:
        adata = adata.copy()
    if graph_name is None:
        graph_name = get_default_graph(adata)

    if graph_name not in adata.uns.get("spatial", {}):
        to_spatial_weights(adata, graph_name)

    W = adata.uns["spatial"][graph_name]
    features = [feature] if isinstance(feature, str) else list(feature)
    losh = esda.LOSH(W, inference=inference)

    adata.obsm.setdefault(key_added, pd.DataFrame(index=adata.obs_names))

    X = adata.X if layer is None else adata.layers[layer]
    if sparse.issparse(X) or isinstance(X, np.matrix):
        X = X.A

    for feat in features:
        if feat in adata.var_names:
            losh.fit(X[:, adata.var_names.get_loc(feat)])
        else:
            losh.fit(adata.obs[feat])
        adata.obsm[key_added][feat] = losh.Hi

    adata.uns["spatial"].setdefault(key_added, {})
    losh_dict = adata.uns["spatial"][key_added]
    losh_dict.setdefault("params", {})

    losh_dict["params"].update(losh.get_params())
    losh_dict["params"]["graph_name"] = graph_name

    return adata


def local_moran(
    adata: AnnData,
    feature: Union[str, Sequence[str]],
    inplace: bool = True,
    permutations: int = 0,
    key_added: str = "local_moran",
    graph_name: Optional[str] = None,
    keep_simulations: bool = False,
    layer: Optional[str] = None,
    **kwargs,
) -> AnnData:
    import esda

    if not inplace:
        adata = adata.copy()
    features = [feature] if isinstance(feature, str) else list(feature)

    adata.uns.setdefault("spatial", {})

    if graph_name is None:
        graph_name = get_default_graph(adata)

    if graph_name not in adata.uns["spatial"]:
        to_spatial_weights(adata, graph_name)

    X = adata.X if layer is None else adata.layers[layer].A
    get_loc = adata.var.index.get_loc
    W = adata.uns["spatial"][graph_name]

    local_morans = [
        esda.Moran_Local(
            X[:, get_loc(feat)] if feat in adata.var_names else adata.obs[feat],
            W,
            permutations=permutations,
            keep_simulations=keep_simulations,
            **kwargs,
        )
        for feat in features
    ]

    adata.obsm.setdefault(key_added, pd.DataFrame(index=adata.obs_names))
    moran_df = adata.obsm[key_added]
    for feat, lm in zip(features, local_morans):
        moran_df[f"{feat}"] = lm.Is

    # Keep metadata for this run
    adata.uns["spatial"].setdefault(key_added, {})

    # Simulations
    if permutations > 0 and keep_simulations:
        for feat, lm in zip(features, local_morans):
            adata.uns["spatial"][key_added].setdefault("sim", {})
            sims_df = pd.DataFrame(lm.sim.T, index=adata.obs_names)
            adata.uns["spatial"][key_added]["sim"][feat] = sims_df

    # Parameters
    adata.uns["spatial"][key_added].setdefault("params", {})
    param_dict = adata.uns["spatial"][key_added]["params"]
    param_dict["graph_name"] = graph_name
    param_dict["permutations"] = permutations
    param_dict["keep_simulations"] = keep_simulations
    param_dict["seed"] = kwargs.get("seed", None)
    return adata


def compute_higher_order_neighbors(
    adata, graph_name=None, force: bool = False, *, order: int
):
    if graph_name is None:
        graph_name = get_default_graph(adata)

    higher_order = adata.uns["spatial"].setdefault("higher_order", {})
    Ws = higher_order.setdefault(graph_name, [])
    if force:
        Ws.clear()

    W = adata.uns["spatial"][graph_name].sparse.copy()
    W.eliminate_zeros()
    W.prune()
    W.data = np.ones_like(W.data)
    n_order = len(Ws)
    if n_order == 0:
        Ws.append(W)
        n_order = 1

    if n_order >= order:
        return Ws

    # M is the original graph
    M = W.copy().tolil()

    # Mp is the order-k neighbourhood graph
    Mp = Ws[-1].copy().tolil()

    # The total graph is the union of all the graphs
    total_graph = M.copy().tolil()
    total_graph.setdiag(1)

    # Reconstruction of the <k order neighbourhood graph
    for w in Ws:
        total_graph = total_graph.maximum(w.tolil())

    for k in range(n_order, order):
        # M is the original graph
        # Mp is the order-k neighbourhood graph
        Mp = ((Mp * M).minimum(1) - total_graph).maximum(0).tolil()
        total_graph = total_graph.maximum(Mp)
        w = Mp.tocsr()
        w.eliminate_zeros()
        w.prune()
        Ws.append(w)

    return Ws


def compute_correlogram(
    adata: AnnData,
    feature: Union[str, Sequence[str]],
    method: str = "moran",
    graph_name: Optional[str] = None,
    order: Optional[int] = None,
    layer: Optional[str] = None,
    force: bool = False,
    key_added: str = "correlogram",
):
    # try importing libpysal and esda
    try:
        import libpysal
        import esda
    except ImportError:
        raise ImportError(
            "Please install libpysal and esda to use correlogram. "
            "See https://pysal.org/libpysal/installation.html and "
            "https://pysal.org/esda/installation.html for installation instructions."
        )

    if method not in ["moran", "losh", "corr"]:
        raise NotImplementedError(
            f"Correlogram is not implemented for method {method}."
        )

    if graph_name is None:
        graph_name = get_default_graph(adata)

    higher_order = adata.uns["spatial"].setdefault("higher_order", {})
    Ws = higher_order.setdefault(graph_name, [])
    if not order and len(Ws) == 0:
        raise RuntimeError(
            "Please provide a positive integer value for `order`, or run `compute_higher_order_neighbors` first."
        )

    if len(Ws) == 0 and order:
        Ws = compute_higher_order_neighbors(adata, graph_name=graph_name, order=order)

    features = [feature] if isinstance(feature, str) else feature[:]
    order = order or (len(Ws) + 1)

    correlogram_dict = adata.uns["spatial"][method].setdefault(key_added, {})
    correlogram_df = correlogram_dict.setdefault(
        graph_name, pd.DataFrame(columns=list(range(1, order + 1)), index=features)
    )
    W_og = adata.uns["spatial"][graph_name].sparse.copy()

    X = adata.X if layer is None else adata.layers[layer]
    if sparse.issparse(X) or isinstance(X, np.matrix):
        X = X.A

    for k_order, w in enumerate((Ws)[:order], 1):
        W = libpysal.weights.WSP(w, id_order=adata.obs_names.to_list()).to_W(
            silence_warnings=True
        )
        if (
            k_order in correlogram_df.columns
            and (not any(correlogram_df[k_order].isna()))
            and not force
        ):
            continue

        for feat in features:
            if not (np.isnan(correlogram_df.at[feat, k_order]) or force):
                continue

            if feat in adata.var_names:
                i_feature = adata.var_names.get_loc(feat)
                x = X[:, i_feature]
            else:
                x = adata.obs[feat]

            if method == "moran":
                val = esda.Moran(x, W, permutations=0).I
            else:
                y = w.dot(x)
                val = np.corrcoef(x, y)[0, 1]

            correlogram_df.at[feat, k_order] = val

    return correlogram_df
