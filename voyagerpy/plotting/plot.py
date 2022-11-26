#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:08:42 2022

@author: sinant
"""


import numpy as np
#import pandas as pd
from anndata import AnnData


from typing import Optional, Union, Mapping,Sequence, Collection, Iterable, List    

from matplotlib.axes import Axes
from matplotlib import pyplot as plt



import geopandas as gpd



from math import ceil

from .. import utils as utl
from .. import spatial as spt
        
        



    
    
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
        adata = spt.get_geom(adata)
    
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
    
    
    
    

        
