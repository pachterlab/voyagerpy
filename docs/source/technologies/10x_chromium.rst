==================================================================
10X Chromium Single Cell 3' V3 Processing Workflows with VoyagerPy
==================================================================

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :hidden:

   10X basic <../examples/nonspatial.ipynb>

The 10X Genomics Chromium Single Cell 3' v3 solution is a droplet-based single cell sequencing technology 
that enables gene expression profiling of hundreds to tens of thousands of cells. In a typical protocol, 
cells are dissociated and captured in Gel Beads-in-emulsion (GEMs). All cDNA from a single cell share 
the same barcode and additional 10X barcodes help associate individual reads from generated libraries 
back to the originating GEMs. Though spatial information is lost, the technology has diverse applications, 
including cell surface protein profiling and assessing the effects of cellular pertubations.

Getting Started
---------------

Download data
^^^^^^^^^^^^^

Several publicly available datasets are available from 10X Genomics on their
`website <https://www.10xgenomics.com/resources/datasets>`_.

Analysis Workflows
------------------

The notebooks below demonstrate workflows that can be implemented with VoyagerPy using a 10X v3 dataset. 
The analysis tasks include basic quality control, spatial exploratory data analysis, identification of
spatially variable genes, and computation of global and local spatial statistics.

.. list-table:: Available notebooks for 10X Chromium V3 datasets.
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `10X basic <../examples/nonspatial.ipynb>`_
      - Standard analyses for non-spatial scRNA-seq data.

