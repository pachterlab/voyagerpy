.. VoyagerPy documentation master file, created by
   sphinx-quickstart on Wed Jun 21 10:43:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VoyagerPy - From geospatial to spatial omics
============================================

.. sidebar:: Data structure

    VoyagerPy uses `AnnData <https://anndata.readthedocs.io/en/latest/>`_ as the data structure for spatial omics data.
    Users who are familiar with the AnnData data structure should feel at home when using VoyagerPy.

VoyagerPy is the Python implementation of the `Voyager package for R <https://pachterlab.github.io/voyager>`_. 
It brings the tradition of geospatial statistics to spatial omics data analysis.

This documentation will help you get started with VoyagerPy and learn how to use it.
The :ref:`technologies` section lists several different technologies for scRNA-seq data. There you can
find notebooks for each technology.
These notebooks aim to illustrate preprocessing of raw data, basic QC, exploratory spatial data analysis and visualization with VoyagerPy.
They are designed to follow the tutorial vignettes for the R package Voyager.

Please head to our :ref:`installation page <installation>` to get started with VoyagerPy.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   api
   technologies


.. sidebar:: Indices and tables

   :ref:`genindex`
   :ref:`modindex`

   .. * :ref:`search`
