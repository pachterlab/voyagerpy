==================================================================
10X Chromium Single Cell 3' V3 Processing Workflows with VoyagerPy
==================================================================

.. toctree::
  :titlesonly:
  :maxdepth: 1
  :hidden:
  :caption: Preprocessing

  Preprocess raw data <../examples/preprocess_rna-10xv3.ipynb>
  Preprocess raw data (ClickTag) <../examples/preprocess_clicktag.ipynb>
  10X v3 Basic QC <../examples/qc_rna-10xv3.ipynb>
  ClickTag Basic QC <../examples/qc_tag-clicktag.ipynb>
  10X v3 Basic Analysis <../examples/nonspatial.ipynb>


The 10X Genomics Chromium Single Cell 3’ v3 solution is a droplet-based single cell sequencing technology 
that enables gene expression profiling in hundreds to tens of thousands of cells. In a typical protocol, 
cells are dissociated and captured in Gel Beads-in-emulsion (GEMs). All cDNA from a single cell share 
a cellular barcode. Additional 10X barcodes help associate individual reads from a cDNA library back 
to the originating GEMs. The gel beads in the v3 chemistry are also modified to enable feature barcoding, 
so that orthogonal molecules can be profiled alongside RNA. This, in addition to greater detection of 
RNA molecules, represents major advancements over the 10X Chromium v2 chemistry.

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
  * - `Preprocess raw data <../examples/preprocess_rna-10xv3.ipynb>`_
    - Fetch reference data with ``gget``, process raw data with ``seqspec``, generate a count matrix with ``kallisto-bustools``
  * - `Preprocess raw data (ClickTag) <../examples/preprocess_clicktag.ipynb>`_
    - Fetch reference data with ``gget``, process raw data with ``seqspec``, generate a count matrix with ``kallisto-bustools``


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
    * - `10X v3 Basic QC <../examples/qc_rna-10xv3.ipynb>`_
      - Standard QC for scRNA-seq data
    * - `ClickTag Basic QC <../examples/qc_tag-clicktag.ipynb>`_
      - Standard QC for scRNA-seq data
    * - `10X v3 Basic <../examples/nonspatial.ipynb>`_
      - Standard analyses for non-spatial scRNA-seq data.

.. |seqspec| replace:: seqspec
.. _seqspec: https://github.com/IGVF/seqspec
.. |gget| replace:: gget
.. _gget: https://github.com/pachterlab/gget
.. |kb_tools| replace:: kallisto / bustools
.. _kb_tools: https://www.kallistobus.tools/