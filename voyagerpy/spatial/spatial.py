#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import geopandas as gpd
import numpy as np
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


def get_tissue_contour_score(cntr: Contour, adata: AnnData, size: str = "hires") -> float:

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
    # print([tp,fp,tn,fn])
    J = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
    # print(J)
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
            new_score = get_tissue_contour_score(big_cntrs[j], adata)
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
    threshold_low: int = None,
    size: str = "hires",
    strictness: Optional[int] = None,
    inplace: bool = False,
    # detect_threshold: bool = False,
) -> Polygon:

    # TODO: Do we want assert that size is either 'lowres' or 'hires'?
    res = "hires" if size == "hires" else "lowres"
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


def get_geom(adata: AnnData, threshold: int = None, inplace: bool = True, res: str = "hires") -> AnnData:

    if "geom" not in adata.uns["spatial"]:
        adata.uns["spatial"]["geom"] = {}

    # add spot points to geom
    # Create a geometry column from x & ly
    scale = utl.get_scale(adata, res=res)
    spot_diam = adata.uns["spatial"]["spot_diameter_fullres"]

    adata.obs["spot_poly"] = adata.obs.apply(
        lambda x: Point(float(x.pxl_col_in_fullres * scale), float(x.pxl_row_in_fullres * scale)).buffer((spot_diam / 2) * 0.2),  # type: ignore
        axis=1,
    )

    # Create a GeoDataFrame from adata.obs
    adata.obs = gpd.GeoDataFrame(adata.obs, geometry=adata.obs.spot_poly)  # type: ignore

    # add boundary and tissue poly to geom
    tissue_poly = get_tissue_boundary(adata, threshold)
    adata.uns["spatial"]["geom"]["tissue_poly"] = tissue_poly
    adata.uns["spatial"]["geom"]["tissue_boundary"] = gpd.GeoSeries(tissue_poly).boundary

    return adata


# %%


def get_spot_coords(adata: AnnData, tissue: bool = True, as_tuple: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    h_sc = utl.get_scale(adata)
    cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    if tissue:
        coords = (adata.obs.loc[adata.obs["in_tissue"] == 1, cols] * h_sc).values
    else:
        coords = (adata.obs.loc[:, cols] * h_sc).values

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
def rotate_img90(adata: AnnData, k: int = 1, apply: bool = True, res: str = "all") -> bool:
    """Rotate the tissue image and the coordinates of the spots by k*90 degrees. If apply is True,
    then adata.uns['spatial']['rotation'][res] will contain the degrees between the original image (and coordinates)
    and the rotated version.

    Parameters
    ----------
    adata : AnnData
        The AnnData whose image and spot coordinates are to be rotated.
    k : int, optional
        The number of times the image should rotated by 90 degrees by default 1
    apply : bool, optional
        Whether to apply the rotation to the image and coordinates, by default True. If False, the
        rotated image will be stored under adata.uns['spatial']['img'] with a key "{res}_rot{k}" for all
        resolutions `res` that exist. The rotated coordinates are stored under adata.uns with keys
        "pxl_col_in_fullres_rot{k}" and "pxl_row_in_fullres_rot{k}" if `apply` is False.

    res : str, optional
        One of 'lowres', 'hires', 'all', the resolution to rotatae, by default "all". If "all", all existing resolutions of the
        image are rotated.

    Returns
    -------
    bool
        True if any image was rotated. False if no image with resolution `res` exists.
    """
    rotation_mats = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]]),
    ]

    k = k % 4

    rot = rotation_mats[k]
    res_keys = adata.uns["spatial"]["img"].keys()
    res_vals = ("lowres", "hires") if res == "all" else (res,)

    def rotator(res):
        img = adata.uns["spatial"]["img"][res]
        img = np.rot90(img, k=k)
        n_rows, n_cols, _ = img.shape

        # Rotate all spot coordinates
        coords = get_spot_coords(adata, tissue=False, as_tuple=False)
        center = np.array([n_rows, n_cols]) / 2

        # We rotate around the center of the image: translate to Origin > rotate > translate back
        coords = np.matmul(coords - center, rot) + center

        return img, coords

    ret = False
    rot_dict = adata.uns["spatial"].get("rotation", {})
    for res in res_vals:
        if res in res_keys:
            img, coords = rotator(res)
            scale = utl.get_scale(adata, res)

            coords = (coords / scale).astype(int)
            col_pos, row_pos = coords[:, 0], coords[:, 1]

            if apply:
                adata.uns["spatial"]["img"][res] = img
                adata.obs["pxl_col_in_fullres"] = col_pos
                adata.obs["pxl_row_in_fullres"] = row_pos
                rot_dict[res] = (rot_dict.get(res, 0) + 90 * k) % 360
            else:
                adata.uns["spatial"]["img"][f"{res}_rot{k}"] = img
                adata.obs[f"pxl_col_in_fullres_rot{k}"] = col_pos
                adata.obs[f"pxl_row_in_fullres_rot{k}"] = row_pos
            ret = True
    adata.uns["spatial"]["rotation"] = rot_dict
    return ret
