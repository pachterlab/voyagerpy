# VoyagerPy

This repo manages the VoyagerPy Python package, a Python implementation of the R package [Voyager](https://github.com/pachterlab/voyager)

## Installation

To install the latest release of VoyagerPy, you can install it via `pip`:

```pip install voyagerpy```

### Clone the repo
Clone this repo either using SSH:

```git clone git@github.com:pmelsted/voyagerpy.git```

or HTTPS:

```git clone https://github.com/pmelsted/voyagerpy.git```.

To get the bleeding edge version, change your branch to `dev` by running

```git checkout dev```

once inside the `voyagerpy` directory.

### Install using `pip`

To install VoyagerPy, run 

```pip install .```

Some users may experince problems with installing GeoPandas, which VoyagerPy depends on. We refer to the [GeoPandas installation page](https://geopandas.org/en/stable/getting_started.html) if this is the case.

## Structure of VoyagerPy

VoyagerPy uses [AnnData](https://anndata.readthedocs.io/) as its internal datastructure. An AnnData object, `adata`, holds the following attributes:

- `adata.X`: the main data matrix of size $N_{obs} \times N_{vars}$. It holds the count data for each observation and feature (e.g. barcodes x genes), which may have gone under some transformation. Data type may be a `scipy.sparse.csr_matrix`, `numpy.ndarray`, or `numpy.matrix`. This is yet to be set in stone.
- `adata.layers`: A dictionary-like data structure with the values being matrices of the same shape as `adata.X`. These can hold transformations of `adata.X`, such as log-normalized counts.
- `adata.obs`: A `pandas.DataFrame` object where the rows represent the barcodes, and the columns are features of the barcodes.
- `adata.obsp`: This is a dictionary-based object, where each value is a `pandas.DataFrame` of size $N_{obs}\times N_{obs}$, representing a pairwise metric on the observation. For instance, `adata.obsp["distances"]` can hold the pairwise distances between the positions of origin for the barcodes. This can be handy to store graphs over the barcodes.
- `adata.obsm`: This is a dictionary-based object where each value is a `pandas.DataFrame` or `geopandas.GeoDataFrame`. The number of rows in these data frames must be $N_{obs}$. Example of data frames to be stored here:
	- `geometry`: a `geopandas.GeoDataFrame` where each column is of `geopandas.GeoSeries` or `pandas.Series`, used for plotting spatial objects, such as points or polygons. To use GeoPandas for plotting a column, it must be a `geopandas.GeoSeries`. These will represent the geometries of the barcodes.
	- `local_*`: a `pandas.DataFrame` which contains spatial results over features in `obs`. These can be e.g. local Moran's I, local spatial heteroscedasticity (LOSH) over some features `x, y, z`. The columns of `local_moran` and `local_losh` would then be `x, y, z`.
- `adata.var`: A `pandas.DataFrame` object where the rows represent the features from the columns of `X` (e.g. genes), and the columns are features of the genes (or whatever the columns of `X` represent).
- `adata.varp`, `adata.varm`: These are not used for the time being, but these objects can be used similarly to `adata.obsp` and `adata.obsm` but for feature (gene) data.

- `adata.uns`: This is a dictionary containing data that cannot be stored in the above objects:
	- `config`: These can contain config or metadata about this object. By using VoyagerPy to read the scRNA-seq data, this dictionary has the following items by default:
		- `"var_names": "gene_ids"`, meaning that the index of the variables (genes) are the standardized ENSG gene IDs.
		- `"secondary_var_names": "symbol"`, meaning that the column `"symbol"` in `adata.var` contains the symbol names for the genes.
	- `spatial`: This dictionary contains various spatial data, including:
		- `img`: a dictionary with key-values as resolution-image
		- `scale`: a dictionary describing the scales of the images.
		- `transform`: metadata that describes the transforms applied to the images (rotation, mirror) such that the originals can be recovered.
		- `local_results`: This is a dictionary which can contain Monte-Carlo simulations of spatial autocorrelation statistics, such as for local Moran statistics.