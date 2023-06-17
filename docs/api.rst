.. module:: voyagerpy

.. automodule:: voyagerpy
	:noindex:

API
===

Import VoyagerPy as follows:

.. code-block:: python

	import voyagerpy as vp

spatial
+++++++

.. module:: voyagerpy.spatial

.. currentmodule:: voyagerpy.spatial

Performing spatial analysis

.. autosummary::
	:toctree: generated/

	get_approx_tissue_boundary

	get_tissue_contour_score

	get_tissue_boundary
	
	detect_tissue_threshold
	
	set_geometry
	
	to_points
	
	get_visium_spots
	
	get_geom
	
	get_spot_coords
	
	to_spatial_weights
	
	compute_spatial_lag
	
	moran
	
	set_default_graph
	
	get_default_graph
	
	local_moran
	
	losh
	
	compute_higher_order_neighbors
	
	compute_correlogram

Coordinate transforms
---------------------

.. autosummary::
	
	mirror_img
	
	cancel_transforms

	rotate_img90

	apply_transforms
	
	rollback_transforms


utils
+++++

.. module:: voyagerpy.utils

.. currentmodule:: voyagerpy.utils

.. autosummary::
	:toctree: generated/

	log_norm_counts


.. .. automodule:: voyagerpy.plotting
.. 	:members:

.. .. automodule:: voyagerpy.spatial
..	:members:

.. .. autofunction:: voyagerpy.plotting.plot.imshow
.. .. autofunction:: voyager.plotting.plot.configure_violins

..    :toctree: generated