.. _installation:

============
Installation
============

Environment
-----------

We recommend using a fresh Conda environment for VoyagerPy. To create a new environment with the name ``voyagerpy`` and Python 3.8, run

.. code-block:: shell

	$ conda create -n voyagerpy python=3.8

To activate the environment, run

.. code-block:: shell

	$ conda activate voyagerpy

Since VoyagerPy is currently only published to PyPI, we recommend installing VoyagerPy and its dependencies with pip.

Install from PyPI
-----------------

To install VoyagerPy from PyPI run:

.. code-block:: shell

	$ pip install voyagerpy

We use Scanpy in the tutorials. Since VoyagerPy shares many dependencies with Scanpy, we recommend installing Scanpy as well with ``pip``. To install Scanpy, run

.. code-block:: shell

	$ pip install "scanpy[leiden]"

Development version
-------------------

You can also install VoyagerPy from GitHub by running 

.. code-block:: shell

	$ pip install git+https://github.com/pmelsted/voyagerpy.git


.. To get the version, use |version|.
