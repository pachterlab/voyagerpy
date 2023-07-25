===============================================================
10X Chromium Nuclei Isolation Processing Workflows with Voyager
===============================================================

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    Preprocessing raw data <../examples/preprocess_rna-10xv3-nuclei>
    Nuclei Basic QC <../examples/qc_rna-10xv3-nuclei>

The `nuclei isolation solution <https://www.10xgenomics.com/products/nuclei-isolation>`_ 
from 10X genomics is a protocol for isolating nuclei from frozen samples for single nuclei sequencing.
The sample prep workflow has the advantage of avoiding long enzymatic incubations and being accessible 
without specialized equipment. The workflow includes cell lysis, nuclei isolation in a spin column, and 
debris removal. Importantly, isolated nuclei are compatible with Chromium Single Cell ATAC, Single Cell 
Multiome ATAC + Gene Expression, and Single Cell Gene Expression assays without modifications to the 
single cell workflows. Nuclei suspension concentration, viability, and quality can be assessed with 
a nucleic acid fluorescent dye and follow-up with an fluorescent automated counter or microscope.

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
    * - `Preprocess raw data <../examples/preprocess_rna-10xv3-nuclei.ipynb>`_
      - Fetch reference data with ``gget``, process raw data with ``seqspec``, generate a count matrix with ``kallisto-bustools``

Analysis Workflows
------------------

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * - Jupyter Notebook
      - Description
    * - `Nuclei Basic QC <../examples/qc_rna-10xv3-nuclei.ipynb>`_
      - Basic QC and preprocessing


.. |seqspec| replace:: seqspec
.. _seqspec: https://github.com/IGVF/seqspec
.. |gget| replace:: gget
.. _gget: https://github.com/pachterlab/gget
.. |kb_tools| replace:: kallisto / bustools
.. _kb_tools: https://www.kallistobus.tools/