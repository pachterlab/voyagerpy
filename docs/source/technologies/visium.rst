==========================================
Visium Processing Workflows with VoyagerPy
==========================================

.. toctree::
  :maxdepth: 1
  :titlesonly:
  :hidden:

  Preprocessing of 10X example Visium dataset <../examples/preprocess_rna-visium-spatial.ipynb>
  Visium 10X Basic QC <../examples/qc_rna-visium-spatial.ipynb>
  Basic analysis of 10X example Visium dataset <../examples/visium_10x.ipynb>

The Visium spatial transcriptomics platform by 10X Genomics is based on |st_technology|_ 
that was originally published in 2016. In both methods, mRNA from tissue sections is captured on 
spatially barcoded spots that are immobilized on a microarray slide. Following construction of a barcoded 
cDNA library, mRNA transcripts can be mapped to specific spots on the microarray slide and overlayed 
with a high-resolution image of the tissue, allowing for visualization and analysis of gene expression 
in a spatial context.

Getting Started
---------------

Download data and preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several publicly available Visium datasets are available from 10X Genomics on their
`website <https://www.10xgenomics.com/resources/datasets>`_.

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * - Notebook
      - Description
    * - `Preprocessing of 10X example Visium dataset <../examples/preprocess_rna-visium-spatial.ipynb>`_
      - Download and preprocess the 10X example Visium dataset.

Analysis Workflows
------------------

The notebooks below demonstrate workflows that can be implemented with VoyagerPy using a variety of 
Visium datasets. The analysis tasks include basic quality control, spatial exploratory data analysis, 
identification of spatially variable genes, and computation of global and local spatial statistics.

.. list-table:: Available notebooks for Visium
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `Visium 10X Basic QC <../examples/qc_rna-visium-spatial.ipynb>`_
      - Perform basic QC and standard non-spatial scRNA-seq analysis.
    * - `Basic analysis of 10X example Visium dataset <../examples/visium_10x.ipynb>`_
      - Perform basic QC and standard non-spatial scRNA-seq analysis, and some spatial visualization.

.. References

.. |st_technology| replace:: Spatial Transcriptomics (ST) technology
.. _st_technology: https://www.nature.com/articles/nmeth.3901
