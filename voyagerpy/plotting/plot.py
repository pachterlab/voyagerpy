#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:08:42 2022

@author: sinant
"""


import numpy as np
import pandas as pd
from anndata import AnnData


from typing import Optional, Union, Mapping,Sequence, Collection, Iterable, List    

from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry import Point

import geopandas as gpd


from cv2 import(
    pointPolygonTest,
    cvtColor,
    findContours,
    contourArea,
    arcLength,
    threshold,
    COLOR_BGR2GRAY,
    RETR_TREE,
    CHAIN_APPROX_SIMPLE,
    COLOR_RGB2BGR
    )

from math import ceil

from .. import utils as utl

        
        

    
def calculate_metrics(adata:AnnData):
    adata.var_names_make_unique()
    #forcells
    #n_genes_by_counts
    adata.obs["n_genes_by_counts"] = np.diff(adata.X.tocsr().indptr)
    #total_counts
    adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).reshape((adata.X.shape[0]))
    #prop_mito
    
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    adata.obs['pct_counts_mt'] = np.sum(adata[:,adata.var["mt"]].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    
    
    #for_genes
    #detected in nr of barcodes
    adata.var["n_cells_by_counts"] = np.diff(adata.X.tocsc().indptr)
    #barcodes per feature
    adata.var["total_counts"] = np.array(adata.X.sum(axis=0)).reshape((adata.X.shape[1]))
    return adata
    


#%%

def get_spot_coords(adata:AnnData,tissue = True):
    
    if(utl.is_highres(adata)):
        h_sc = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
    else:
        h_sc = adata.uns["spatial"]["scale"]["tissue_lowes_scalef"]
    if(tissue):
        return np.array([h_sc*adata.obs[adata.obs["in_tissue"] == 1].iloc[:,4],h_sc *adata.obs[adata.obs["in_tissue"] == 1].iloc[:,3]])
    else:
        return np.array(h_sc*adata.obs.iloc[:,4]),np.array(h_sc *adata.obs.iloc[:,3])
    
    
    
# def plot_overlay(adata:AnnData,tissue=True,col="red",colorbar=False):
#     if(is_highres(adata)):
#         img_size = 2000
#         plot_loc = "hires"
#     else:
#         img_size= 600
#         plot_loc = "lowres"
#     fig = plt.figure(figsize = (10,10))
#     plt.imshow(adata.uns["spatial"]["img"][plot_loc])
#     #fig = plt.figure()
#     size = fig.get_size_inches()*fig.dpi
#     size = size/img_size
#     #scaling = adata.uns["spatial"]["scale"]
#     #plt.plot(200, 300, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
#     h_sc = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
#     circ = adata.uns["spatial"]["scale"]["fiducial_diameter_fullres"]
#     #circ = adata.uns["spatial"]["scale"]["spot_diameter_fullres"]
    
#     # if(col is not None):
#     #     col = "red"
    
#     x,y = get_spot_coords(adata,tissue)
    
#     #plt.plot(x, y, marker="o", markersize=circ*h_sc*size[0], markeredgecolor=sp_col, markerfacecolor=sp_col,alpha=.5)
#     plt.scatter(x,y,marker="o",c=col,s=circ*h_sc*size[0],alpha= 0.5)
#     if(colorbar):
#         plt.colorbar(orientation="horizontal",shrink=1)
        
    
#     # for i in range(adata.obs.shape[0]):
#     #     #print(adata.obs.iloc[i,3])
        
#     #     x = h_sc *  adata.obs.iloc[i,4]
#     #     y = h_sc *  adata.obs.iloc[i,3]
#     #     if(adata.obs.iloc[i,0] == 1):
#     #         sp_col = "red"
#     #         plt.plot(x, y, marker="o", markersize=circ*h_sc*size[0], markeredgecolor=sp_col, markerfacecolor=sp_col,alpha=.5)
#     #     else:
#     #         sp_col = "blue"
        
#     plt.show()
    
    
def plot_spatial_features(
        adata:AnnData,
        features: Union[str, Sequence[str]],
        ncol = None,
        barcode_geom:str = None,
        annot_geom = None,
        tissue=True,
        colorbar=False,
        color = "blue",
        cmap = "Blues",
        annot_style = None,
        ax: Optional[Axes] = None,
        **kwds,
):
    
    
    #check input
    if(("geometry" not in adata.obs) or "geom" not in adata.uns["spatial"]):
        adata = get_geom(adata)
    
    #copy observation dataframe so we can edit it without changing the inputs
    obs = adata.obs
    
    #check if barcode geometry exists
    if(barcode_geom is not None):
        if(barcode_geom not in obs):
           raise ValueError(
                f"Cannot find {barcode_geom!r} data in adata.obs"
            )
    

        # if barcode_geom is not spot polygons, change the default geometry of the observation matrix, so we can plot it
        if(barcode_geom != "spot_poly"):
            obs.set_geometry(barcode_geom)
    
    #check if features are in rowdata
    if isinstance(features, list):
        feat_ls = features
    if(isinstance(features,str)):
        feat_ls = [features]
    
    #Check if too many subplots
    if(len(feat_ls) > 6):
        raise ValueError(
            "Too many features to plot, reduce the number of features"
        )
    if(ncol is not None):
        if(ncol > 3):
            raise ValueError(
                "Too many columns for subplots"
            )
    
    
    #only work with spots in tissue
    if(tissue==True):
        obs = obs[obs["in_tissue"] == 1]
    
    
    #create the subplots with right cols and rows
    if(ax is None): 
        plt_nr = len(feat_ls)
        nrows = 1
        #ncols = ncol if ncol is not None else 1
        
        #defaults
        if(ncol is None):
            if(plt_nr < 4):
                ncols = plt_nr
            if(plt_nr >= 4):
                nrows = 2
                ncols = 3
        else:
            nrows = ceil(plt_nr/ncols)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 10))
        plt.subplots_adjust(wspace = 1/ncols +  0.2)
    
    #iterate over features to plot
    x = 0
    y = 0
    for i in range(len(feat_ls)):
    
    
        if(tissue==True):

            #if gene value
            if(feat_ls[i] in adata.var.index):
                #col = adata.var[features]
                col = adata[adata.obs["in_tissue"] == 1,feat_ls[i]].X.todense().reshape((adata[adata.obs["in_tissue"] == 1,:].shape[0])).T
    
                col = np.array(col.ravel()).T
                obs[feat_ls[i]] = col
            if(feat_ls[i] in obs.columns):
                #feat = features
                pass
        else:
            
                if(feat_ls[i] in adata.var.index):
                    #col = adata.var[features]
                    col = adata[:,feat_ls[i]].X.todense().reshape((adata.shape[0]))
                    obs[feat_ls[i]] = col
                if(feat_ls[i] in obs.columns):
                    pass
            
        if(ncols > 1 and nrows > 1):
            ax = axs[x,y]
        if(ncols == 1 and nrows > 1):
            ax = axs[x]
        if(nrows == 1 and ncols > 1):
            ax = axs[y]
        if(ncols == 1 and nrows == 1):
            ax = axs
            
            
            
            
            
            
            
        obs.plot(feat_ls[i], ax=ax, legend=True,cmap=cmap,legend_kwds={'label': feat_ls[i],'orientation': "vertical","shrink":0.3})    
        if(annot_geom is not None):
            if(annot_geom in adata.uns["spatial"]["geom"]):
                
                #check annot_style is dict with correct values
                plg = adata.uns["spatial"]["geom"][annot_geom]
                if(annot_style is not None):
                    gpd.GeoSeries(plg).plot(ax=ax,**annot_style,**kwds) 
                else:
                    
                
                    gpd.GeoSeries(plg).plot(color="blue",ax=ax,alpha=0.2,**kwds) 
            else:
                raise ValueError(
                    f"Cannot find {annot_geom!r} data in adata.uns['spatial']['geom']"
                )
                
            pass
        y = y + 1
        if(y >= ncols):
            y = 0
            x = x + 1

            
    
    if(ax is None):
        return axs
    else:
        return ax
    
    
    
    

        
#create spatial functions with shapely

def get_approx_tissue_boundary(adata:AnnData,size="hires",paddingx = 0,paddingy=0):
    if(size == "hires"):
        scl = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
    else:
        scl = adata.uns["spatial"]["scale"]["tissue_lowres_scalef"]
    bot = int(np.max(adata.obs[adata.obs["in_tissue"] == 1]["pxl_row_in_fullres"])*scl)
    top = int(np.min(adata.obs[adata.obs["in_tissue"] == 1]["pxl_row_in_fullres"])*scl)
    right = int(np.max(adata.obs[adata.obs["in_tissue"] == 1]["pxl_col_in_fullres"])*scl)
    left = int(np.min(adata.obs[adata.obs["in_tissue"] == 1]["pxl_col_in_fullres"])*scl)
    if(paddingx !=0):
        left = left - paddingx
        right = right + paddingx
    if(paddingy != 0):
        top = top-paddingy
        bot = bot + paddingy
    
    return [top,bot,left,right]



#%%

def get_tissue_contour_score(cntr,adata,size="hires"):
    if(size == "hires"):
        scl = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
    if(size == "lowres"):
        scl = adata.uns["spatial"]["scale"]["tissue_lowres_scalef"]
    # tissue_barcodes = adata.obs[adata.obs["in_tissue"] == 1]
    # non_tissue_barcodes  = adata.obs[adata.obs["in_tissue"] != 1]
    # total = tissue_barcodes.shape[0]
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(adata.obs.shape[0]):
        #print([int(tissue_barcodes.iloc[i,3]*0.2),int(tissue_barcodes.iloc[i,4]*0.2)])
        #print(cv2.pointPolygonTest(big_cntrs[0], (int(tissue_barcodes.iloc[i,4]*0.2),int(tissue_barcodes.iloc[i,3]*0.2)), False) )
        if(pointPolygonTest(cntr, (int(adata.obs["pxl_col_in_fullres"][i]*scl),int(adata.obs["pxl_row_in_fullres"][i]*scl)), False) == 1):
            if(adata.obs["in_tissue"][i] == 1):
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if(adata.obs["in_tissue"][i] == 1):
                fn = fn + 1
            else:
                tn = tn + 1
    
    #method youden j....whynot
    #print([tp,fp,tn,fn])
    J = (tp/(tp+fn)) + (tn/(tn+fp)) - 1
    #print(J)
    return J
            
def detect_tissue_threshold(adata,size="hires",low=200,high=255):

    
    bgr_img = cvtColor(adata.uns["spatial"]["img"][size], COLOR_RGB2BGR)
    bgr_img = (bgr_img*255).astype("uint8")
    imgray = cvtColor(bgr_img, COLOR_BGR2GRAY)
    px_thrsh = low
    thrsh_score_all = 0
    best_thrsh = 0
    best_cntr_all = None
    for i in range(px_thrsh,high):
        ret, thresh = threshold(imgray, i,255, 0)
        # contours
        contours, contours2= findContours(thresh, RETR_TREE, CHAIN_APPROX_SIMPLE);
        # filter contours by size
        
        big_cntrs = [];
        #marked = bgr_img.copy();
        for contour in contours:
            area = contourArea(contour);
            if(area > 10000):
                #print(area);
                big_cntrs.append(contour);
        #print(len(big_cntrs))
        score = 0
        best_cntr = None
        for j in range(len(big_cntrs)):
            new_score = get_tissue_contour_score(big_cntrs[j], adata)
            if(new_score > score):
                best_cntr = big_cntrs[j]
                score = new_score
                
                
                
                
        if(score > thrsh_score_all):
            #print("score is " ,thrsh_score_all)
            if(best_cntr_all is not None):
                #print("ratio is " ,cv2.arcLength(best_cntr_all, True)/ cv2.arcLength(best_cntr, True))
                
                if(arcLength(best_cntr_all, True)/ arcLength(best_cntr, True)< 0.9):
                    if(abs(thrsh_score_all - score) < 0.1):
                        break

            best_thrsh = i
            best_cntr_all = best_cntr
            thrsh_score_all = score
            #print("score is " ,thrsh_score_all)
            #if(best_cntr_all is not None):
                
            #    print(cv2.arcLength(best_cntr_all, True)/cv2.arcLength(best_cntr, True))
            
            
        
    return best_thrsh,best_cntr_all
    
    
    


def get_tissue_boundary(adata:AnnData,threshold=222,size = "hires",strictness=None,inplace=False,detect_treshold = False):
    if(size == "hires"):
        scl = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
        res = "hires"
    else:
        scl = adata.uns["spatial"]["scale"]["tissue_lowres_scalef"]
        res = "lowres"
    
    # load image
    
    bgr_img = cvtColor(adata.uns["spatial"]["img"][res], COLOR_RGB2BGR)
    bgr_img = (bgr_img*255).astype("uint8")
    
    

    # rescale
    #scale = 0.25
    #h, w = img.shape[:2]
    #h = int(h*scale)
    #w = int(w*scale)
    #img = cv2.resize(img, (w,h))
    
    # hsv
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    #h, s, v = cv2.split(hsv);
    
    # thresh
    #thresh = cv2.inRange(h, 140, 179);

    imgray = cvtColor(bgr_img, COLOR_BGR2GRAY)
    ret, thresh = threshold(imgray, threshold,255, 0)
    
    # contours
    contours, contours2= findContours(thresh, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    # filter contours by size
    
    big_cntrs = [];
    #marked = bgr_img.copy();
    for contour in contours:
        area = contourArea(contour);
        if(area > 10000):
            #print(area);
            big_cntrs.append(contour);
    
    #for all contours check if all points are within countour and all points outside it
    
    
    
    score = 0
    best_cntr = None
    for i in range(len(big_cntrs)):
        new_score = get_tissue_contour_score(big_cntrs[i], adata)
        if(new_score > score):
            score = new_score
            best_cntr = big_cntrs[i]
        
        #not_tissue_barcodes = adata.obs[adata.obs["in_tissue"] == 0]

        
        #ts_out_p = []
        
    
    
    #if(strictness == "strict"):
        
    
    
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
    
    
    
    
    contour = np.squeeze(best_cntr)
    polygon = Polygon(contour)
    #print(polygon.wkt)


    return polygon#,out

    #create outline of tissue sample  
def get_geom(adata:AnnData,inplace="True"):
    
    if("geom" not in adata.uns["spatial"]):
        adata.uns["spatial"]["geom"] = {}
    
    #add spot points to geom
    # Create a geometry column from x & ly
    adata.obs['spot_poly'] = adata.obs.apply(lambda x: Point(float(x.pxl_col_in_fullres*0.2), float(x.pxl_row_in_fullres*0.2)).buffer((73.61433/2)*0.2), axis=1)

    # Create a GeoDataFrame from adata.obs 
    adata.obs = gpd.GeoDataFrame(adata.obs, geometry = adata.obs.spot_poly)
    
    
    #add boundary and tissue poly to geom
    
    tissue_poly = get_tissue_boundary(adata)
    adata.uns["spatial"]["geom"]["tissue_poly"] = tissue_poly
    adata.uns["spatial"]["geom"]["tissue_boundary"] = gpd.GeoSeries(tissue_poly).boundary
    # if(os.path.exists(path+spatial_path)):
        
    # else:
    #     raise ValueError(
    #         "Cannot read file tissue_positions.csv"
    #     )

    #total feature counts in spots under tissue
    
    return adata
    