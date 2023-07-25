==================================================================
10X Chromium Single Cell CRISPR Screening Workflows with VoyagerPy
==================================================================

.. toctree::
    :titlesonly:
    :hidden:
    :maxdepth: 1

    Preprocessing raw data <../examples/preprocess_crispr-10xcrispr.ipynb>
    10X CRISPR Basic QC <../examples/qc_crispr-10xcrispr.ipynb>

The `Chromium Single Cell CRISPR Screening from 10x Genomics <https://www.10xgenomics.com/products/single-cell-crispr-screening>`_ 
enables researchers to assess the effects of CRISPR-mediated pertubations on cellular phenotypes and 
gene expression. The workflow relies on harvesting cells that have been transduced with CRISPR guides 
and stimulated before single cell preparation and library generation. Workflows typically include 2-5 
sgRNAs per gene and 100-250 cells can be recovered per guide. Currently, the 5’ CRISPR screening assay 
is compatible with most pre-existing Cas9 guide RNA vectors while the 3’ assay requires a specific capture 
sequence for capture of the guide.


Getting Started
---------------

Download Data and Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The notebooks below provide examples of processing raw data using a workflow that includes |seqspec|_, 
|gget|_, and |kb_tools| to generate a count matrix.

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `Preprocess raw data <../examples/preprocess_crispr-10xcrispr.ipynb>`_
      - Fetch reference data with ``gget``, process raw data with ``seqspec``, generate a count matrix with ``kallisto-bustools``

Analysis Workflows
------------------

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `10X CRISPR Basic QC <../examples/qc_crispr-10xcrispr.ipynb>`_
      - Basic QC and preprocessing


.. |seqspec| replace:: seqspec
.. _seqspec: https://github.com/IGVF/seqspec
.. |gget| replace:: gget
.. _gget: https://github.com/pachterlab/gget
.. |kb_tools| replace:: kallisto / bustools
.. _kb_tools: https://www.kallistobus.tools/