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
import warnings


def read_img_data(path: Union[Path, PathLike], adata: AnnData, res: str = "hires") -> AnnData:
    path = Path(path)

    scalefactors_path = path / "scalefactors_json.json"
    img_path = path / f"tissue_{res}_image.png"

    spatial_dict = adata.uns.setdefault("spatial", {})

    if img_path.exists():
        image = imread(str(img_path))
        img_dict = spatial_dict.setdefault("img", {})
        img_dict[res] = image
    else:
        warnings.warn(f"Could not find image {img_path}")

    if scalefactors_path.exists():
        spatial_dict["scale"] = json.load(scalefactors_path.open("rt"))
    else:
        warnings.warn(f"Could not find scalefactors_json.json in {scalefactors_path}")

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
    tissue_alt_path = tissue_pos_path.with_name("tissue_positions_list").with_suffix(".csv")

    if tissue_pos_path.exists():
        version = 2
        adata.obs = pd.concat([adata.obs, pd.read_csv(tissue_pos_path).set_index(["barcode"])], axis=1)
    elif tissue_alt_path.exists():
        version = 1
        colnames = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        adata.obs = pd.concat([adata.obs, pd.read_csv(tissue_pos_path, header=None, names=colnames).set_index(["barcode"])], axis=1)
    else:
        raise ValueError("Cannot read file tissue_positions.csv")

    adata = read_img_data(path / "spatial", adata, res=res)
    metadata = {
        "data_source": "visium",
        "img_res": [res for res in ("lowres", "hires") if (path / "spatial" / f"tissue_{res}_image.png").exists()],
        "version": version,
    }
    adata.uns["metadata"] = metadata
    return adata



from shapely.geometry import Polygon, MultiPolygon
import pandas as pd
import h5py
import anndata as ad

def create_polygon(group):
    """
    Create a Polygon from a group of vertices.

    Parameters:
    group (pd.DataFrame): A DataFrame containing the vertices of the polygon with columns 'vertex_x' and 'vertex_y'.

    Returns:
    Polygon: A Shapely Polygon object created from the provided vertices.

    This function takes a group (typically a subset of a DataFrame grouped by some criteria)
    and creates a Polygon by zipping together the 'vertex_x' and 'vertex_y' columns.

    Example usage:
    polygon = create_polygon(group)
    """
    return Polygon(zip(group['vertex_x'], group['vertex_y']))

def create_multipolygon(group):
    """
    Create a MultiPolygon from a group of geometries.

    Parameters:
    group (pd.DataFrame): A DataFrame containing geometries in a column 'geometry' and corresponding cell IDs in 'cell_id'.

    Returns:
    pd.Series: A Series with a MultiPolygon object and unique cell IDs.

    This function takes a group (typically a subset of a DataFrame grouped by some criteria)
    and creates a MultiPolygon by combining the geometries in the 'geometry' column.
    It also extracts the unique cell IDs from the 'cell_id' column.

    Example usage:
    multipolygon_series = create_multipolygon(group)
    multipolygon = multipolygon_series['geometry']
    cell_ids = multipolygon_series['cell_id']
    """
    multipolygon = MultiPolygon(group['geometry'].tolist())
    cell_ids = group['cell_id'].unique()
    return pd.Series({'geometry': multipolygon, 'cell_id': cell_ids})

def combine_polygons(group):
    """
    Combine multiple geometries into a single MultiPolygon.

    Parameters:
    group (pd.DataFrame): A DataFrame containing geometries in a column 'geometry' and corresponding cell IDs in 'cell_id'.

    Returns:
    pd.Series: A Series with a combined MultiPolygon object and the first cell ID.

    This function takes a group (typically a subset of a DataFrame grouped by some criteria)
    and combines the geometries in the 'geometry' column into a single MultiPolygon.
    It also extracts the first cell ID from the 'cell_id' column.

    Example usage:
    combined_polygon_series = combine_polygons(group)
    combined_multipolygon = combined_polygon_series['geometry']
    cell_id = combined_polygon_series['cell_id']
    """
    combined_multipolygon = MultiPolygon([geom for geom in group['geometry']])
    return pd.Series({'geometry': combined_multipolygon, 'cell_id': group['cell_id'].iloc[0]})

def read_xenium(data_folder):
    """
    Load and process Xenium spatial transcriptomics data from a specified folder.

    Parameters:
    data_folder (str): Path to the folder containing the Xenium data files.

    Returns:
    AnnData: An AnnData object containing the spatial transcriptomics data with cell and nucleus boundary information.

    The function performs the following steps:
    1. Reads the cell feature matrix from an HDF5 file and constructs a sparse matrix.
    2. Creates DataFrames for observation (obs) and variable (var) data.
    3. Reads additional cell information from a Parquet file and merges it with obs.
    4. Reads cell boundary information from a Parquet file and constructs geometries.
    5. Reads nucleus boundary information from a Parquet file, creates polygons, and combines them.
    6. Merges the cell and nucleus geometries into the obs metadata.
    7. Reads transcript information from a Parquet file and stores it in uns.
    8. Sets the technology type in uns.

    Example usage:
    adata = read_xenium("/path/to/data_folder")
    """
    from scipy.sparse import csr_matrix
    import geopandas as gpd


    matrix_file = f"{data_folder}/cell_feature_matrix.h5"

    # Open the HDF5 file
    with h5py.File(matrix_file, "r") as f:
        # Function to recursively print the structure of the HDF5 file
        def print_structure(name, obj):
            print(name)
        
        # Print the structure of the file
        f.visititems(print_structure)

        barcodes = f['matrix/barcodes'][:].astype(str)
        data = f['matrix/data'][:]
        feature_names = f['matrix/features/name'][:].astype(str)
        feature_types = f['matrix/features/feature_type'][:].astype(str)
        indices = f['matrix/indices'][:]
        indptr = f['matrix/indptr'][:]
        shape = f['matrix/shape'][:]

        X = csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

        # Create DataFrames for obs and var
        obs = pd.DataFrame(index=barcodes)
        var = pd.DataFrame(index=feature_names)
        var['feature_type'] = feature_types
        
        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs, var=var)

    cell_information_file = f"{data_folder}/cells.parquet"

    cell_df = pd.read_parquet(cell_information_file)

    cell_df.set_index('cell_id', inplace=True)

    aligned_cell_df = cell_df.reindex(adata.obs.index)
    adata.obs = adata.obs.join(aligned_cell_df)

    cell_boundary_information_file = f"{data_folder}/cell_boundaries.parquet"

    cell_boundaries_df = pd.read_parquet(cell_boundary_information_file)

    polygons = cell_boundaries_df.groupby('cell_id').apply(lambda group: Polygon(zip(group['vertex_x'], group['vertex_y'])))

    # Convert the Series of polygons to a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, columns=['geometry'])

    adata.obsm['geometry'] = pd.DataFrame(index=gdf.index)

    adata.obsm['geometry']['cellSeg'] = gdf

    nucleus_boundary_information_file = f"{data_folder}/nucleus_boundaries.parquet"

    nucleus_boundaries_df = pd.read_parquet(nucleus_boundary_information_file)

    # Group by 'label_id' and 'cell_id' to create individual polygons
    polygons_df = nucleus_boundaries_df.groupby(['label_id', 'cell_id']).apply(create_polygon).reset_index()

    # Rename the resulting column for clarity
    polygons_df = polygons_df.rename(columns={0: 'geometry'})

    # Ensure 'cell_id' is of a hashable type (string)
    polygons_df['cell_id'] = polygons_df['cell_id'].astype(str)

    # Group by 'cell_id' and apply the combination function
    combined_gdf = polygons_df.groupby('cell_id').apply(combine_polygons).reset_index(drop=True)

    # Convert to GeoDataFrame
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry')
    combined_gdf.rename(columns={'geometry': 'nucSeg'}, inplace=True)

    merged_df = pd.merge(adata.obsm['geometry'].reset_index(), combined_gdf, on='cell_id', how='left', suffixes=('', '_combined'))

    # Set the index back to the original index of adata.obsm['geometry']
    merged_df.set_index('cell_id', inplace=True)

    # Assign the merged DataFrame back to adata.obsm['geometry']
    adata.obsm['geometry'] = merged_df


    transcripts_information_file = f"{data_folder}/transcripts.parquet"

    transcripts_df = pd.read_parquet(transcripts_information_file)

    adata.uns['transcripts'] = transcripts_df

    adata.uns['technology'] = "xenium"

    return adata