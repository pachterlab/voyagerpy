#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import traceback
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import h5py
import pandas as pd
from anndata import AnnData, read_mtx
from matplotlib.pyplot import imread
from .spatial import set_geometry, to_points
from .utils import get_scale


def read_img_data(path: Union[Path, PathLike], adata: AnnData, res: str = "high") -> AnnData:
    path = Path(path)

    loc = "hires" if res == "high" else "lowres"
    spatial_path = path / "spatial"
    scalefactors_path = spatial_path / "scalefactors_json.json"
    img_path = spatial_path / f"tissue_{loc}_image.png"

    if img_path.exists() and scalefactors_path.exists():
        image = imread(str(img_path))
        adata.uns.setdefault("spatial", {}).setdefault("img", {})
        adata.uns["spatial"]["img"][loc] = image
        adata.uns["spatial"]["scale"] = json.load(scalefactors_path.open("rt"))
    else:
        raise ValueError("Cannot read tissue image or scaling file")
    return adata


def _read_10x_h5(path: PathLike, symbol_as_index: bool = False, dtype: str = "float64") -> Optional[AnnData]:
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
            # cell_nr = indptr[cell_ind+1]
            cm = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]), dtype=dtype)
            # df_feat = pd.DataFrame(np.column_stack((features["id"][()].astype("str"),features["feature_type"][()].astype("str"),
            # features["genome"][()].astype("str"))),index=features["name"][()].astype("str"))

            var_names_key = "id"
            gene_name_key = "name"
            secondary_gene_column_name: str = "symbol"
            if symbol_as_index:
                var_names_key, gene_name_key = gene_name_key, var_names_key
                secondary_gene_column_name = "gene_ids"

            adata = AnnData(
                cm,
                obs=dict(obs_names=barcodes),
                var={
                    "var_names": features[var_names_key][()].astype("str"),
                    secondary_gene_column_name: features[gene_name_key][()].astype("str"),
                    "feature_types": features["feature_type"][()].astype("str"),
                    "genome": features["genome"][()].astype("str"),
                },
                dtype=dtype,
            )

        return adata
    except Exception:
        traceback.print_exc()
        return None
    return adata


def read_10x_counts(
    path: PathLike,
    datatype: Optional[str] = None,
    raw: bool = True,
    prefix: Optional[str] = None,
    symbol_as_index: bool = False,
    dtype: str = "float64",
) -> AnnData:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Reading with path {path!r} failed, ")

    prefix_str = prefix or ""

    raw_qualifier = "raw" if raw else "filtered"
    h5_file_path = path / f"{prefix_str}{raw_qualifier}_feature_bc_matrix.h5"
    mtx_dir_path = path / f"{prefix_str}{raw_qualifier}_feature_bc_matrix"

    adata: Optional[AnnData] = None

    if (datatype is None and h5_file_path.exists()) or datatype == "h5":
        adata = _read_10x_h5(h5_file_path, symbol_as_index=symbol_as_index, dtype=dtype)
    elif (datatype is None and mtx_dir_path.exists()) or datatype == "mtx":
        adata = _read_10x_mtx(mtx_dir_path, symbol_as_index=symbol_as_index, dtype=dtype)

    if adata is None:
        raise ValueError("Invalid datatype for bc_matrix")

    adata.uns["config"] = {
        "var_names": "symbol" if symbol_as_index else "gene_ids",
        "secondary_var_names": "gene_ids" if symbol_as_index else "symbol",
    }

    return adata


def _read_10x_mtx(path: PathLike, symbol_as_index: bool = False, dtype: str = "float64") -> AnnData:
    path = Path(path)
    genes = pd.read_csv(path / "features.tsv.gz", header=None, sep="\t")
    cells = pd.read_csv(path / "barcodes.tsv.gz", header=None, sep="\t")

    data = read_mtx(path / "matrix.mtx.gz", dtype=dtype).T

    varname_pos, geneids_pos = 0, 1
    gene_column_key = "symbol"
    if symbol_as_index:
        varname_pos, geneids_pos = geneids_pos, varname_pos
        gene_column_key = "gene_ids"

    adata = AnnData(
        data.X,
        obs=dict(obs_names=cells[0].to_numpy()),
        var={
            "var_names": genes[varname_pos].to_numpy(),
            gene_column_key: genes[geneids_pos].to_numpy(),
            "feature_types": genes[2].to_numpy(),
        },
        dtype=dtype,
    )

    return adata


def read_10x_visium(
    path: PathLike,
    datatype: Optional[str] = None,
    raw: bool = True,
    prefix: Optional[str] = None,
    symbol_as_index: bool = False,
    dtype: str = "float64",
    res: str = "hires",
) -> AnnData:
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
    adata = read_10x_counts(path, datatype, raw, prefix, symbol_as_index, dtype=dtype)

    # spatial
    tissue_pos_path = path / "spatial" / "tissue_positions.csv"
    tissue_alt_path = tissue_pos_path.with_stem("tissue_positions_list")

    if tissue_pos_path.exists():
        version = 2
        adata.obs = pd.concat([adata.obs, pd.read_csv(tissue_pos_path).set_index(["barcode"])], axis=1)
    elif tissue_alt_path.exists():
        version = 1
        colnames = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        adata.obs = pd.concat([adata.obs, pd.read_csv(tissue_pos_path, header=None, names=colnames).set_index(["barcode"])], axis=1)
    else:
        raise ValueError("Cannot read file tissue_positions.csv")

    adata = read_img_data(path, adata, res=res)
    metadata = {
        "data_source": "visium",
        "img_res": [res for res in ("lowres", "hires") if (path / "spatial" / f"tissue_{res}_image.png").exists()],
        "version": version,
    }
    adata.uns["metadata"] = metadata
    return adata
