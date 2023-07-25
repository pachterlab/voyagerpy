=================================================================
10X Chromium Single Cell ATAC Processing Workflows with VoyagerPy
=================================================================

.. toctree::
  :titlesonly:
  :maxdepth: 1
  :hidden:

  10X ATAC Preprocessing <../examples/preprocess_atac-10xatac.ipynb>
  10X ATAC Basic QC <../examples/qc_atac-10xatac.ipynb>

The `Chromium Single Cell ATAC solution from 10x Genomics <https://www.10xgenomics.com/products/single-cell-atac>`_
allows researchers to interrogate chromatin accessibility at the single cell level. In contrast to 
other methods that rely on chromatin being bound by specific factors, ATAC-seq captures all open chromatin 
regions. The workflow requires a nuclei suspension isolated from cells or tissue and is optimized for 
use with the `Chromium Nuclei Isolation Kit <https://www.10xgenomics.com/products/nuclei-isolation>`_. 
Nuclei are suspended in a solution that includes transposase, which preferentially fragments regions 
of open chromatin. Individual nuclei and DNA fragments are sequestered into Gel Beads-in-emulsion 
(GEMs) where a pool of barcodes is used to index the transposed DNA of each individual nucleus. 
The workflow results in prepared libraries that are compatible with sequencing on Illumina sequencers. 
Analysis from this data supports discovery of gene regulatory elements and mechanisms, cell types and 
states, and epigenetic changes in response to disease or drug exposure.


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
    * - `Preprocess raw data <../examples/preprocess_atac-10xatac.ipynb>`_
      - Fetch reference data with ``gget``, process raw data with ``seqspec``, generate a count matrix with ``kallisto-bustools``

Analysis Workflows
------------------

.. list-table:: Available notebooks for 10X ATAC datasets.
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `10X ATAC Basic QC <../examples/preprocess_atac-10xatac.ipynb>`_
      - Basic QC and preprocessing


.. |seqspec| replace:: seqspec
.. _seqspec: https://github.com/IGVF/seqspec
.. |gget| replace:: gget
.. _gget: https://github.com/pachterlab/gget
.. |kb_tools| replace:: kallisto / bustools
.. _kb_tools: https://www.kallistobus.tools/