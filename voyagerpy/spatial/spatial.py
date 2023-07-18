#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
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
    """Compute approx tissue. Get the approximate tissue boundary from an image in the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    size : str, optional
        The image resolution to use, by default "hires"
    paddingx : int, optional
        Horizontal padding for cropping the image, by default 0.
    paddingy : int, optional
        The vertical padding from cropping the image, by default 0

    Returns
    -------
    Tuple[int, int, int, int]
        The top, bottom, left, and right coordinates of the tissue boundary, in pixels.
    """

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


## write a docstring for this function
def get_tissue_contour_score(cntr: Contour, adata: AnnData, size: str = "hires") -> float:
    """Get the score of a contour. This function takes a contour and returns a score for how well it fits the tissue.

    Parameters
    ----------
    cntr : Contour
        The contour to score, represented as a list of points from a cv2 findContours function.
    adata : AnnData
        Annotated data matrix.
    size : {"hires", "lowres"}, optional
        The resolution of the image to use, by default "hires"

    Returns
    -------
    float
        Score of the contour of how well it fits the tissue.
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
    """Detect the tissue boundary using a thresholding method. The contours are evaluated from the image in ``adata.uns["spatial"]["img"][size]``

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    size : {"hires", "lowres"}, optional
        The resolution of the image to use for tissue detction, by default "hires"
    low : int, optional
        The minimum threshold values, by default 200
    high : int, optional
        The maximum threshold value, by default 255

    Returns
    -------
    Tuple[int, Optional[Contour]]
        The threshold value and the contour of the tissue boundary.
    """
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
    """Detect the tissue boundary. This function computes the tissue boundary from the image in ``adata.uns["spatial"]["img"][size]``

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    threshold_low : Optional[int], optional
        The minimum threshold by default None
    size : {"hires", "lowres"}, optional
        The resolution of the image to use, by default "hires"
    strictness : Optional[int], optional
        Not used.
    inplace : bool, optional
        Not used.

    Returns
    -------
    Polygon
        Polygon of the tissue boundary.
    """

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
    """Set the geometry of the AnnData object. This function sets the geometry of the AnnData object in ``adata.obsm["geometry"]``, which is a ``gpd.GeoDataFrame``.
    If the geometry dataframe does not exist, it will be created. If values is not None, it will be set as the geometry values.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    geom : str
        The name of the geometry column to set. If the column does not exist, it will be created and populated with ``values``.
    values : Optional[gpd.GeoSeries], optional
        The geometry series to populate the `geom` column with, by default None. Must not be `None` if the column `geom` does not exist.
    index : Optional[pd.Index], optional
        The index to associate with `values`. If `None`, will use the geometry dataframe's indexing, by default None`.
    dim : Union[str, Literal[&quot;barcode&quot;, &quot;gene&quot;]], optional
        The type of geometry to set, by default "barcode". If "barcode", will set the geometry of the barcodes. If "gene", will set the geometry of the genes.
        Otherwise, the geometry is assumed to be annotation geometry, residing in `adata.uns["spatial"]["geometry][dim]`. Defaults to "barcode".
    inplace : bool, optional
        Whether to modify `adata` inplace. If `False`, a copy is made, by default `True`.

    Returns
    -------
    AnnData
        The updated AnnData object. If `inplace` is `True`, returns a copy. This object will have
        `geom` column in the respective geometry dataframe set as its geometry.

    Raises
    ------
    ValueError
        If `values` is `None` and `geom` does not exist.
    """
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
    """Create a GeoSeries from x and y coordinates. If radius is not None, will create a circle with the given radius around each point.
    Otherwise, will create a point at each x,y coordinate.

    Parameters
    ----------
    x : Union[str, pd.Series, np.ndarray]
        The x coordinates of the points. If ``str``, will use ``data[x]``.
    y : Union[str, pd.Series, np.ndarray]
        The y coordinates of the points. If ``str``, will use ``data[y]``.
    data : Union[gpd.GeoDataFrame, pd.DataFrame, None], optional
        The dataframe to get `x` and `y` from if the are `str`, by default None
    scale : float, optional
        The scale to use for converting the coordinates to pixel coordinates, by default 1
    radius : Optional[float], optional
        If supplied, the radius of the circle at each coordinate, by default None

    Returns
    -------
    gpd.GeoSeries
        Geoseries of points at the given coordinates. If `radius` is not `None`, each coordinate will be a circle of type Polycon with the given radius.

    Raises
    ------
    ValueError
        If `data` is `None`, and `x` or `y` is `str`.
    """

    if data is None and (isinstance(x, str) or isinstance(y, str)):
        raise ValueError("data must not be None if either x or y is str")
    xdat = data[x] if isinstance(x, str) else x
    ydat = data[y] if isinstance(y, str) else y

    points = gpd.GeoSeries.from_xy(xdat, ydat).scale(scale, scale, origin=(0, 0))
    if radius:
        return points.buffer(radius)
    return points


def get_visium_spots(adata: AnnData, with_radius: bool = False, res: Optional[str] = None) -> gpd.GeoSeries:
    """Return a GeoSeries of the spots in the Visium slide. If ``with_radius`` is ``True``, will return circular polygons with the radius of the spot diameter, otherwise will return points.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    with_radius : bool, optional
        Whether to create Polygons with radius `radius`. If `False`, the dtype of the returned GeoSeries will be Point, by default False
    res : Optional[str], optional
        The resolution to use for the coordinate system. If `None`, this will be determined automatically, by default None

    Returns
    -------
    gpd.GeoSeries
        Points or Polygons of the spots in the Visium slide.
    """

    scale = utl.get_scale(adata, res=res)
    scale_dict = adata.uns["spatial"].get("scale", {})
    spot_diam = scale_dict.get("spot_diameter_fullres")
    return to_points(
        x="pxl_col_in_fullres",
        y="pxl_row_in_fullres",
        data=adata.obs,
        scale=scale,
        radius=scale * spot_diam / 2 if with_radius else None,
    )


def get_geom(
    adata: AnnData,
    threshold: Optional[int] = None,
    inplace: bool = False,
    res: Optional[str] = None,
) -> AnnData:
    """Get the tissue polygons and tissue boundary from the sample image. If they don't exist, they will be computed.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    threshold : Optional[int], optional
        The threshold for computind the tissue segmentation, by default None
    inplace : bool, optional
        If `True`, update the AnnData object inplace, otherwise return a copy, by default False
    res : Optional[str], optional
        The resolution of the image to use for tissue segmentation. If `None`, determine the resolution automatically, by default None

    Returns
    -------
    AnnData
        The updated or copied AnnData object. The geometry dataframe will contain the columns `"tissue_poly"` and `"tissue_boundary"`,
        representing the polygon and its boundary for the tissue. The geometry dataframe is accessed at
        `adata.uns["spatial"]["geom"]`.
    """

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
    subset: Union[None, pd.Series, slice] = None,
    as_tuple: bool = True,
    as_df: bool = False,
    res: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
    """Get the spot coordinates in the image.

    Parameters
    ----------
    adata : AnnData
        Annontated data matrix.
    subset : Union[None, pd.Series, slice], optional
        Subset of observations to get the spot coordinates from, by default `None`.
    as_tuple : bool, optional
        If `True`, return the coordinates as a tuple of arrays (x, y), by default True
    as_df : bool, optional
        If `True`, return the the coordinates as a dataframe, by default `False`.
    res : Optional[str], optional
        The resolution to scale the coordinates for. If `None`, this is determined automatically , by default `None`.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray], pd.DataFrame]
        The coordinates of the selected spots, as a tuple, dataframe, or Nx2 array.
    """
    h_sc = utl.get_scale(adata, res)
    cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]

    subset = slice(None) if subset is None else subset
    coords = adata.obs.loc[subset, cols] * h_sc

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
    """Cancel unapplied image transforms.

    These transforms will be lost.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.

    See Also
    --------
    :py:func:`apply_transforms`
    :py:func:`rollback_transforms`
    :py:func:`mirror_img`
    :py:func:`rotate_img90`

    Examples
    --------

    >>> import voyagerpy as vp
    ... adata = ...
    ... vp.spatial.rotate_img90(adata, k=2, apply=False)
    ... vp.spatial.cancel_transforms(adata)
    # The mirror transform is now lost. Coordinates and image are back to original.
    """

    spatial_dict = adata.uns["spatial"]
    transforms = spatial_dict.get("transform", ([], []))
    pxl_coord_cols = ["pxl_col_in_fullres_tmp", "pxl_row_in_fullres_tmp"]

    transforms[1].clear()
    spatial_dict.pop("img_tmp", None)
    adata.obs.drop(pxl_coord_cols, axis="columns", inplace=True, errors="ignore")


def apply_transforms(adata: AnnData) -> None:
    """Apply unapplied image transforms.

    This changes the image and coordinates in the AnnData object and removes any temporary transformations.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    Examples
    --------
    >>> import voyagerpy as vp
        adata = ...
        vp.spatial.rotate_img90(adata, k=2, apply=False)
        vp.spatial.mirror_img(adata, k=2, apply=False)
        # The transforms are still temporary and not applied.
        vp.spatial.apply_transforms(adata)
        # The transforms are now applied, with image and coordinates rotated first, then mirrored.

    See Also
    --------
    :py:func:`cancel_transforms`
    :py:func:`rollback_transforms`
    :py:func:`mirror_img`
    :py:func:`rotate_img90`
    """

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
    """Rotate the coordinate system.

    Rotates the image and the coordinates clockwise by 90 degrees `k` times.

    Parameters
    ----------
    img : np.ndarray
        The image to rotate.
    coords : np.ndarray
        The coordinates to rotate.
    k : int
        How many times to rotate the image and coordinates by 90 degrees clockwise.
        If negative, rotates counter-clockwise.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rotated image and coordinates.
    """

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


def _mirror_coordinate_system(img: np.ndarray, coords: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror the coordinate system.

    Mirrors the image and the coordinates along the given axis.

    Parameters
    ----------
    img : np.ndarray
        The image to mirror.
    coords : np.ndarray
        The coordinates to mirror.
    axis : int
        The axis to mirror along.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The mirrored image and coordinates.
    """
    n_rows, n_cols = img.shape[:2]

    if axis % 2 == 0:
        img = img[::-1, ...]
        coords[:, 1] = n_rows - 1 - coords[:, 1]

    if axis > 0:
        img = img[:, ::-1, ...]
        coords[:, 0] = n_cols - 1 - coords[:, 0]

    return img, coords


def get_transformation_function(which: str) -> Callable[[AnnData, bool, Optional[int], Optional[int]], Dict[str, np.ndarray]]:
    """Get a transformation function for mirroring or rotating the image and coordinates.

    Parameters
    ----------
    which : {"rotate", "mirror"}
        Which function to get.

    Returns
    -------
    Callable[[AnnData, bool, Optional[int], Optional[int]], Dict[str, np.ndarray]]
        The transformation function.
    """

    def mirror_param_eval(k: Optional[int] = None, axis: Optional[int] = None) -> int:
        """Evaluate the mirror parameters.

        Parameters
        ----------
        k : Optional[int], optional
            Rotation parameter - not used, by default `None`.
        axis : Optional[int], optional
            The axis to mirror along, by default `None`.

        Returns
        -------
        int
            The axis to use. If axis is None, returns 0. Otherwise, returns axis from the input.

        Raises
        ------
        ValueError
            If `axis` not in [None, 0, 1].
        """

        axis = axis or 0
        if axis not in range(2):
            raise ValueError("Invalid mirror axis, must be either 0 or 1")
        return axis

    def rotate_param_eval(k: Optional[int] = None, axis: Optional[int] = None) -> int:
        """Evaluate the rotation parameters.

        Parameters
        ----------
        k : Optional[int], optional
            The number of 90-degree rotations, by default None
        axis : Optional[int], optional
            Mirror axis - not used, by default None

        Returns
        -------
        int
            k modulo 4. If k is None, returns 1.
        """

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

    def inner(
        adata: AnnData,
        apply: bool = True,
        k: Optional[int] = None,
        axis: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Perform the transformation.

        Parameters
        ----------
        adata : AnnData
            The annotated data matrix.
        apply : bool, optional
            Apply the transformation. If `False`, store the temporary transformation, by default `True`.
        k : Optional[int], optional
            The number of times to rotate the image, if rotation is selected, by default `None`.
        axis : Optional[int], optional
            The axis to mirror along, if mirroring is selected, by default `None`.

        Returns
        -------
        Dict[str, np.ndarray]
            _description_
        """
        """The actual function performing the transformation.

        :param adata: The AnnData object to transform.
        :type adata: AnnData
        :param apply: Whether to apply the transformations or store them in a temporary object, defaults to True
        :type apply: bool, optional
        :param k: The number of 90-degree rotations, defaults to None
        :type k: Optional[int], optional
        :param axis: The axis to mirror along, defaults to None
        :type axis: Optional[int], optional
        :return: The transformed images
        :rtype: Dict[str, np.ndarray]
        """
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


def rotate_img90(
    adata: AnnData,
    apply: bool = True,
    k: Optional[int] = None,
    axis: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """\
Rotate the image by 90 degrees k times.

:param adata: The AnnData object to transform.
:type adata: AnnData
:param apply: Whether to apply the transformations or store them in a temporary object, defaults to True
:type apply: bool, optional
:param k: The number of 90-degree rotations, defaults to None
:type k: Optional[int], optional
:param axis: Mirror axis - not used, defaults to None
:type axis: Optional[int], optional
:return: The transformed images
:rtype: Dict[str, np.ndarray]
    """

    func = get_transformation_function("rotate")
    return func(adata, apply, k, axis)


def mirror_img(
    adata: AnnData,
    apply: bool = True,
    k: Optional[int] = None,
    axis: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """\
Mirror the image along an axis.

:param adata: The AnnData object to transform.
:type adata: AnnData
:param apply: Whether to apply the transformations or store them in a temporary object, defaults to True
:type apply: bool, optional
:param axis: The axis to mirror along, defaults to None
:type axis: Optional[int], optional
:return: The transformed images
:rtype: Dict[str, np.ndarray]
    """

    func = get_transformation_function("mirror")
    return func(adata, apply, k, axis)


def rollback_transforms(adata: AnnData, apply: bool = True) -> None:
    """\
Rollback all transformations. Use this function to cancel applied transformations.

:param adata: The annotated data matrix.
:type adata: AnnData
:param apply: If True, drop the temporary transformation, defaults to True
:type apply: bool, optional
    """
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


def to_spatial_weights(adata: AnnData, graph_name: Optional[str] = None) -> AnnData:
    """\
Convert a graph adjacency matrix to a spatial weights matrix.

:param adata: The AnnData object storing the graph
:type adata: AnnData
:param graph_name: The key in ``obsp`` storing the matrix, defaults to None. If None, the default graph is used.
:type graph_name: Optional[str], optional
:raises ImportError: If libpysal is not installed.
:return: The updated AnnData object. The libpysal spatial weights matrix is stored in ``adata.uns["spatial"][graph_name]``.
:rtype: AnnData
    """
    try:
        import libpysal
    except ImportError:
        raise ImportError("Spatial Weights require libpysal to be installed. Please install it with `pip install libpysal`.")

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
    """\
Compute the spatial lag of a feature. The spatial lag is the weighted average of the feature in the neighborhood of each spot.

:param adata: the AnnData object to compute the spatial lag for.
:type adata: AnnData
:param feature: The feature to compute the spatial lag for. Must be a column in ``adata.obs`` or in``adata.var_names``.
:type feature: str
:param graph_name: The neighborhood graph to use, defaults to None. If None, use the default graph. If a string, use the graph stored in ``adata.uns["spatial"][graph_name]``. If an array, use the array as the adjacency matrix.
:type graph_name: Union[None, str, np.ndarray], optional
:param inplace: Whether to add the lagged_features to the AnnData object in place or copy it, defaults to False
:type inplace: bool, optional
:param layer: If not None, use this layer for the feature if the feature is a gene, defaults to None
:type layer: Optional[str], optional
:raises TypeError: If the graph_name is not of type ``None``, ``str``, or ``np.ndarray``.
:return: The updated AnnData object. The spatial lag is stored in ``adata.obs["lagged_" + feature]``. If inplace is False, the return a copy of AnnData.
:rtype: AnnData
    """
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
) -> None:
    """\
Compute Moran's I.

:param adata: The AnnData object to compute Moran's I for.
:type adata: AnnData
:param feature: The feature(s) to compute Moran's I for. Must be a column in ``adata.obs`` or in``adata.var_names``.
:type feature: Union[str, Sequence[str]]
:param graph_name: The neighborhood graph defining the weights, defaults to None. If None, use the default graph. Must be a key in ``adata.uns["spatial"]`` or ``adata.obsp["spatial"]``.
:type graph_name: Optional[str], optional
:param layer: If the feature is a gene id, use this layer for its value, defaults to None
:type layer: Optional[str], optional
:param dim: Whether the feature is a gene or in obs. Will be deprecated, defaults to "obs"
:type dim: Literal[&quot;obs&quot;, &quot;var&quot;], optional
:param permutations: How many permutations to use when simulating Moran's I, defaults to 0
:type permutations: int, optional
:raises ImportError: If esda is not installed
:raises ValueError: If ``dim`` not in ``["obs", "var"]``.
    """
    try:
        import esda
    except ImportError:
        raise ImportError("Moran's I requires the `esda` package. Please install it with `pip install esda`.")

    if graph_name is None:
        graph_name = get_default_graph(adata)

    if graph_name not in adata.uns.get("spatial", {}):
        to_spatial_weights(adata, graph_name)

    W = adata.uns["spatial"][graph_name]

    features = [feature] if isinstance(feature, str) else list(feature[:])

    if dim == "obs":
        morans = [esda.Moran(adata.obs[feat], W, permutations=permutations) for feat in features]
    elif dim == "var":
        feat_idx = list(map(adata.var.index.get_loc, features))
        X = adata.X.A if layer is None else adata.layers[layer].A
        morans = [esda.Moran(X[:, i], W, permutations=permutations) for i in feat_idx]
    else:
        raise ValueError('dim must either be "obs" or "var"')

    moran_dict = adata.uns["spatial"].setdefault("moran", {})
    df = moran_dict.setdefault(graph_name, pd.DataFrame(columns=["I", "EI"], dtype=("double", "double")))

    for feat, moran in zip(features, morans):
        df.at[feat, "I"] = moran.I
        df.at[feat, "EI"] = moran.EI

    if permutations > 0:
        moran_sims = adata.uns["spatial"].setdefault("moran_mc", {})
        sims_dict = moran_sims.setdefault(graph_name, {})
        for feat, moran in zip(features, morans):
            df = sims_dict.setdefault(feat, pd.DataFrame(columns=["sim", "p_sim"]))
            df["sim"] = moran.sim
            df["p_sim"] = moran.p_sim

    # TODO: What do we want to return?
    # return morans[0] if isinstance(feature, str) else morans


def set_default_graph(adata: AnnData, graph_name: str) -> None:
    """\
Set the default graph for spatial operations.

:param adata: The AnnData object to set the default graph for.
:type adata: AnnData
:param graph_name: The name of the graph to set as default. Should be a key in ``adata.obsp``.
:type graph_name: str
    """
    adata.uns.setdefault("spatial", {})
    adata.uns["spatial"]["default_graph"] = graph_name


def get_default_graph(adata: AnnData) -> str:
    """\
Get the default graph for spatial operations. A shorthand for ``adata.uns["spatial"]["default_graph"]``.

:param adata: The AnnData object to get the default graph for.
:type adata: AnnData
:return: The name of the default graph. If none is set, return "connectivities".
:rtype: str
    """
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
    """\
Compute LOSH for a feature. Local spatial heterogeneity (LOSH) is a measure of how spatially clustered a feature is. It is defined as the ratio of the variance of the feature in the neighborhood of a cell to the variance of the feature in the entire dataset. The neighborhood is defined by the spatial graph.

:param adata: The AnnData object to compute LOSH for.
:type adata: AnnData
:param feature: The feature(s) to compute LOSH for. Must be a column in ``adata.obs`` or in``adata.var_names``.
:type feature: Union[str, Sequence[str]]
:param graph_name: The neighborhood graph name, defaults to None.
:type graph_name: Optional[str], optional
:param inference: The inference method to pass to ``esda.LOSH`` constructor, defaults to None
:type inference: Union[None, Literal[&quot;permutation&quot;], Literal[&quot;chi, optional
:param inplace: Whether to add the results to adata inplace or copy it, defaults to True
:type inplace: bool, optional
:param key_added: The key in ``adata.obsm`` to store the results in, defaults to "losh"
:type key_added: str, optional
:param layer: If not None, use this layer for gene features, defaults to None
:type layer: Optional[str], optional
:raises ImportError: If ``esda`` is not installed.
:return: The updated AnnData object. If inflace is False, returns a copy.
:rtype: AnnData
    """
    try:
        import esda
    except ImportError:
        raise ImportError("LOSH requires the `esda` package. Please install it with `pip install esda`.")

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
    keep_simulations: Optional[bool] = None,
    layer: Optional[str] = None,
    na_action: Literal["omit", "exclude", "fail", "pass"] = "fail",
    **kwargs: Any,
) -> AnnData:
    """\
Compute local Moran's I for a feature.

:param adata: The AnnData object to compute local Moran's I for.
:type adata: AnnData
:param feature: The feature(s) to compute local Moran's I for. Must be a column in ``adata.obs`` or in``adata.var_names``.
:type feature: Union[str, Sequence[str]]
:param inplace: Whether to add the results to the adata object inplace or copy it, defaults to True
:type inplace: bool, optional
:param permutations: The number of permutations to permform for simulations, defaults to 0
:type permutations: int, optional
:param key_added: The key in ``adata.obsm`` to add, defaults to "local_moran"
:type key_added: str, optional
:param graph_name: The neighborhood graph to use, defaults to None
:type graph_name: Optional[str], optional
:param keep_simulations: Whether to keep the simulations if permutations > 0, defaults to False
:type keep_simulations: bool, optional
:param layer: If not None, use this layer for gene features, defaults to None
:type layer: Optional[str], optional
:raises ImportError: if esda is not installed.
:return: The updated AnnData object. If inplace is False, returns a copy. The results are stored in ``adata.obsm[key_added]``. If permutations > 0, the simulations are stored in ``adata.uns["spatial"][key_added]["sim"][feature]``. If feature is a list of strings, stores each of the features as such.
:rtype: AnnData"""

    try:
        import esda
        import libpysal
    except ImportError:
        raise ImportError(
            "Local Moran's I requires the `esda` and `libpysal` packages. Please install it with `pip install libpysal esda`."
        )

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

    localmoran_df = adata.obsm.setdefault(key_added, pd.DataFrame(index=adata.obs_names, dtype="float64"))
    local_morans = []

    if keep_simulations is None:
        keep_simulations = permutations > 0

    exclude_nan = na_action in ("omit", "exclude")
    fail_on_nan = na_action == "fail"
    pass_on_nan = na_action == "pass"

    for feat in features:
        y = X[:, get_loc(feat)] if feat in adata.var_names else adata.obs[feat]
        y = y.A if sparse.issparse(y) else y
        localmoran_df[feat] = np.nan
        isna = (np.isnan(y) | np.isinf(y)).squeeze()

        if fail_on_nan and isna.any():
            raise ValueError(f"Feature {feat} contains NaN or inf values")

        if pass_on_nan:
            W1 = W
            idx_keep = adata.obs_names
        else:
            y = y[~isna]
            idx_keep = adata.obs_names[~isna]
            W1 = libpysal.weights.w_subset(W, idx_keep)

        lm = esda.Moran_Local(
            y,
            W1,
            permutations=permutations,
            keep_simulations=keep_simulations,
            **kwargs,
        )
        local_morans.append(lm)
        n = y.size
        correction = n / (n - 1)
        # TODO: correct wrt n-1 vs n
        localmoran_df.loc[idx_keep, feat] = lm.Is

    # Keep metadata for this run
    metadata = adata.uns["spatial"].setdefault(key_added, {})

    # Simulations
    if permutations > 0 and keep_simulations:
        for feat, lm in zip(features, local_morans):
            sim_dict = metadata.setdefault("sim", {})

            nan_idx = None
            if exclude_nan:
                index = localmoran_df[feat].dropna().index
                nan_idx = localmoran_df[localmoran_df[feat].isna()].index
            else:
                index = localmoran_df[feat].index

            sims_df = pd.DataFrame(index=localmoran_df.index, columns=np.arange(lm.sim.shape[0]), dtype="float64")
            sims_df.loc[index, :] = lm.sim.T

            if nan_idx is not None and na_action == 'omit':
                sims_df.drop(nan_idx, inplace=True, axis=0)

            sim_dict[feat] = sims_df

    # Parameters
    param_dict = metadata.setdefault("params", {})
    param_dict["graph_name"] = graph_name
    param_dict["permutations"] = permutations
    param_dict["keep_simulations"] = keep_simulations
    param_dict["seed"] = kwargs.get("seed", None)
    param_dict["na_action"] = na_action
    param_dict["layer"] = layer
    param_dict["features"] = features

    return adata


def compute_higher_order_neighbors(
    adata: AnnData,
    graph_name: Optional[str] = None,
    force: bool = False,
    *,
    order: int,
) -> List[sparse.csr_matrix]:
    """\
Compute higher order neighbors of graph. The first order neighbors is the graph itself.

:param adata: The AnnData object storing the graph.
:type adata: AnnData
:param order: The order of neighbors to compute. Will compute all orders up to this order.
:type order: int
:param graph_name: The key in ``adata.uns["spatial"]`` storing the spatial weights neighborhood graph, defaults to None.
:type graph_name: Optional[str], optional
:param force: If True, recompute all pre-computed orders. Otherwise, start with the highest pre-computed order. Defaults to False.
:type force: bool, optional
:return: List of libpysal spatial weights matrices. The first element is the original graph, the second element is the first order neighbors, etc.\
This list is a reference to ``adata.uns["spatial"]["higher_order"][graph_name]``.
:rtype: List[sparse.csr_matrix]
    """
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
) -> pd.DataFrame:
    """Compute the correlogram of a feature.

    :param adata: The AnnData object storing the graph.
    :type adata: AnnData
    :param feature: The feature(s) to compute the correlogram for.
    :type feature: Union[str, Sequence[str]]
    :param method: The metric to compute at each order of neighbors, defaults to "moran"
    :type method: str, optional
    :param graph_name: The name of the neighborhood graph, defaults to None. \
Should be a key in ``adata.uns["spatial"]["higher_order]`` or ``adata.uns["spatial"]``.
    :type graph_name: Optional[str], optional
    :param order: The order to use for computing the correlogram. If None, compute_higher_order_neighbors must have been called, defaults to None.
    :type order: Optional[int], optional
    :param layer: If not None, use this layer for gene features, defaults to None
    :type layer: Optional[str], optional
    :param force: Whether to recompute the correlogram for values computed in an earlier call to this function, defaults to False
    :type force: bool, optional
    :param key_added: The key to add to ``adata.uns["spatial"][method]`` for storing the correlogram, defaults to "correlogram"
    :type key_added: str, optional
    :raises ImportError: If either libpysal or esda are not installed.
    :raises NotImplementedError: If method is not "moran" or "corr".
    :raises RuntimeError: If order is None and compute_higher_order_neighbors has not been called.
    :return: The dataframe containing the correlogram. It is stored in ``adata.uns["spatial"][method][key_added][graph_name]``.
    :rtype: pd.DataFrame
    """
    try:
        import libpysal
        import esda
    except ImportError:
        raise ImportError(
            "Please install libpysal and esda to use correlogram. "
            "See https://pysal.org/libpysal/installation.html and "
            "https://pysal.org/esda/installation.html for installation instructions."
        )

    if method not in ["moran", "corr"]:
        raise NotImplementedError(f"Correlogram is not implemented for method {method}.")

    if graph_name is None:
        graph_name = get_default_graph(adata)

    higher_order = adata.uns["spatial"].setdefault("higher_order", {})
    Ws = higher_order.setdefault(graph_name, [])
    if not order and len(Ws) == 0:
        raise RuntimeError("Please provide a positive integer value for `order`, or run `compute_higher_order_neighbors` first.")

    if len(Ws) == 0 and order:
        Ws = compute_higher_order_neighbors(adata, graph_name=graph_name, order=order)

    features = [feature] if isinstance(feature, str) else feature[:]
    order = order or (len(Ws) + 1)

    correlogram_dict = adata.uns["spatial"][method].setdefault(key_added, {})
    correlogram_df = correlogram_dict.setdefault(graph_name, pd.DataFrame(columns=list(range(1, order + 1)), index=features))
    W_og = adata.uns["spatial"][graph_name].sparse.copy()

    X = adata.X if layer is None else adata.layers[layer]
    if sparse.issparse(X) or isinstance(X, np.matrix):
        X = X.A

    for k_order, w in enumerate((Ws)[:order], 1):
        W = libpysal.weights.WSP(w, id_order=adata.obs_names.to_list()).to_W(silence_warnings=True)
        if k_order in correlogram_df.columns and (not any(correlogram_df[k_order].isna())) and not force:
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
