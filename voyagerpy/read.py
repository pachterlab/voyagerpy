#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from os import PathLike
from os.path import exists
import json
import h5py
from matplotlib.pyplot import imread
from anndata import AnnData
from anndata import read_mtx
import traceback
#import numpy as np
import pandas as pd


def read_img_data(path:PathLike,adata,res="high"):
    path = Path(path)
    
    if(exists(path / "spatial" / "tissue_hires_image.png") and exists(path / "spatial" / "scalefactors_json.json")):
        if(res == "high"):
            readpath = path  / "spatial" / "tissue_hires_image.png"
            loc = "hires"
        else:
            readpath = path / "spatial" / "tissue_lowres_image.png"
            loc = "lowres"
        image = imread(readpath)
        #adata.uns = {}
        adata.uns["spatial"] = {}
        adata.uns["spatial"]["img"] = {}
        #adata.uns["spatial"]["img"] = image
        adata.uns["spatial"]["img"][loc] = image
        adata.uns["spatial"]["scale"]  = json.load(open(path / "spatial" / "scalefactors_json.json"))
    else:
        raise ValueError(
            "Cannot read tissue image or scaling file"
        )
    return adata
    
    adata.uns
    
    

def _read_10x_h5(path:PathLike):
    """
    Parameters
    ----------
    path : String
        String that designates where the h5 file can be found.

    Returns
    -------
    adata : Anndata object
        Returns anndata object with features and barcode information.

    """
    from scipy.sparse import csr_matrix
    
    try:
        with h5py.File(path, "r") as f:
        
            for i in f.keys():
                data = f[i]["data"][()]
                barcodes = f[i]["barcodes"][()].astype("str")
                features = f[i]["features"]
                shape = f[i]["shape"][()]
                indices = f[i]["indices"][()]
                indptr = f[i]["indptr"][()]
        
             
            # arr = np.zeros((shape[1],shape[0]))
            # cell_nr = indptr[0]
            # cell_ind = 0
            # for i in range(indices.shape[0]):
                
                
            #     arr[cell_ind,indices[i]] = data[i]
            #     if(i > indptr[cell_ind+1]):
            #         cell_ind = cell_ind + 1
                    #cell_nr = indptr[cell_ind+1] 
            cm = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]),dtype="float32")
            #df_feat = pd.DataFrame(np.column_stack((features["id"][()].astype("str"),features["feature_type"][()].astype("str"),
                                                    #features["genome"][()].astype("str"))),index=features["name"][()].astype("str"))
            adata = AnnData(cm,obs=dict(obs_names = barcodes),var = dict(
                                                                        var_names = features["name"][()].astype("str"),
                                                                        gene_ids = features["id"][()].astype("str"),
                                                                        feature_types = features["feature_type"][()].astype("str"),
                                                                        genome = features["genome"][()].astype("str")))

        return adata
    except Exception:
        traceback.print_exc()
        return None
    return adata
def _read_10x_mtx(path:PathLike):
    path = Path(path)
    
    genes = pd.read_csv(path / 'features.tsv.gz', header=None, sep='\t')
    cells = pd.read_csv(path / 'barcodes.tsv.gz', header=None, sep='\t')

    data = read_mtx(path / 'matrix.mtx.gz').T
    
    adata = AnnData(data.X,obs=dict(obs_names = cells[0].to_numpy()),var = dict(
                                                                var_names = genes[1].to_numpy(),
                                                                gene_ids = genes[0].to_numpy(),
                                                                feature_types = genes[2].to_numpy()))
    return adata

def read_10x_visium(path : PathLike,datatype=None,raw = True,prefix=None):
    """
    
    Parameters
    ----------
    path : PathLike
        Path to visium "out" directory.
    datatype : String, optional
        Either h5 or mtx. If h5 the function will load raw_feature_bc_matrix.h5 or 
        filtered_feature_bc_matrix.h5 in the path directory.
        If it is mtx then the function will look for the approprate directory for .gz files. The default is None.
    raw : Boolean, optional
        If false loads only barcodes detected in tissue, otherwise loads the raw barcodes. The default is True.
    prefix : Str, optional
        If there is a prefix to the data this appends it when loading the files. The default is None.

    Raises
    ------
    ValueError
        If the function cannot find files in designated paths, it raises an error.

    Returns
    -------
    adata : Anndata object
        Complete anndata object with spatial information in adata.uns["spatial"] .

    """
    
    path = Path(path)
    if(not exists(path)):
        raise ValueError(
            f'Reading with path {path!r} failed, '
        )
    #path = os.path.normpath(path)
    if(prefix is not None):
        prefix_str = prefix
    else:
        prefix_str = ""
    if(raw):
        h5_file_path = prefix_str + "raw_feature_bc_matrix.h5"
        mtx_dir_path = "raw_feature_bc_matrix"
    else:
        h5_file_path = prefix_str + "filtered_feature_bc_matrix.h5"
        mtx_dir_path = "filtered_feature_bc_matrix"
        
    
    #wait with testing outs
    # if path.endswith("outs"):
    #     pass
    # else:
    #     if(os.path.exists(path+"/outs")):
            
    if(datatype is None or datatype == "h5"):
        
        adata = _read_10x_h5(path / h5_file_path)
    if(datatype == "mtx"):
        adata = _read_10x_mtx(path / mtx_dir_path)
        
        
        
        
    #spatial
    spatial_path =  "spatial/tissue_positions.csv"
    if(exists(path / spatial_path)):
        adata.obs = pd.concat([adata.obs,pd.read_csv(path / "spatial" / "tissue_positions.csv").set_index(["barcode"])],axis = 1)
    else:
        raise ValueError(
            "Cannot read file tissue_positions.csv"
        )
    adata = read_img_data(path,adata)
    return adata
    