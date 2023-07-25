=====================================================================
10X Chromium Single Cell Multiome Processing Workflows with VoyagerPy
=====================================================================

.. toctree::
	:maxdepth: 2
	:hidden:
	:titlesonly:

	Preprocess raw data <../examples/preprocess_atac-10xmultiome.ipynb>
	10X Multiome Basic QC <../examples/qc_atac-10xmultiome.ipynb>

The `Chromium Single Cell Multiome solution from 10x Genomics <https://www.10xgenomics.com/products/single-cell-multiome-atac-plus-gene-expression>`_ 
allows researchers to simultaneously 
interrogate chromatin accessibility and gene expression at the single cell level. Many of the workflow 
objectives in the standalone :ref:`Single Cell ATAC <10xatac>` assay are mirrored in the multiome assay. Both assays 
require a nuclei suspension that is incubated in a solution that includes transposase. Nuclei and 
fragmented DNA are partitioned into Gel Beads-in-emulsion (GEMs) where a pool of barcodes is used to 
index the transposed DNA of each individual nucleus. In the multiome assay, gel beads include a poly(dT) 
sequence that facilitates capture of mRNA for gene expression and a spacer sequence that enables barcode 
attachment to transposed DNA fragments for ATAC library generation. Sequenced libraries support 
investigations into the connections between open chromatin peaks and gene expression.

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
    * - `Preprocess raw data <../examples/preprocess_atac-10xmultiome.ipynb>`_
      - Fetch reference data with ``gget``, process raw data with ``seqspec``, generate a count matrix with ``kallisto-bustools``

Analysis Workflows
------------------

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `10X Multiome Basic QC <../examples/qc_atac-10xmultiome.ipynb>`_
      - Basic QC and preprocessing


.. |seqspec| replace:: seqspec
.. _seqspec: https://github.com/IGVF/seqspec
.. |gget| replace:: gget
.. _gget: https://github.com/pachterlab/gget
.. |kb_tools| replace:: kallisto / bustools
.. _kb_tools: https://www.kallistobus.tools/