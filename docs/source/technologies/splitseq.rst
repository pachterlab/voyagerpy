===========================================
SPLiT-seq Processing Workflows with Voyager
===========================================

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Contents:

    Preprocess raw data <../examples/preprocess_rna-splitseq.ipynb>
    SPLiT-seq Basic QC <../examples/qc_rna-splitseq.ipynb>


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
    * - `Preprocess raw data <../examples/preprocess_rna-splitseq.ipynb>`_
      - Fetch reference data with ``gget``, process raw data with ``seqspec``, generate a count matrix with ``kallisto-bustools``

Analysis Workflows
------------------

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `SPLiT-seq Basic QC <../examples/qc_rna-splitseq.ipynb>`_
      - Basic QC and preprocessing


.. |seqspec| replace:: seqspec
.. _seqspec: https://github.com/IGVF/seqspec
.. |gget| replace:: gget
.. _gget: https://github.com/pachterlab/gget
.. |kb_tools| replace:: kallisto / bustools
.. _kb_tools: https://www.kallistobus.tools/