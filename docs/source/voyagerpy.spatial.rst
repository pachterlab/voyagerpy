Spatial functions
=================

.. module:: voyagerpy.spatial
   :synopsis: Spatial functions for Voyager data

.. currentmodule:: voyagerpy.spatial

Tissue stuff
************

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

Image transformation
********************

These functions allow rotating and mirroring the tissue images.
These functions perform the same transformations on the spot coordinates.

.. autosummary::
   :toctree: generated/

   apply_transforms
   cancel_transforms
   mirror_img
   rollback_transforms
   rotate_img90


Other stuff
***********

.. autosummary::
   :toctree: generated/

   to_spatial_weights
   compute_spatial_lag
   moran
   local_moran
   compute_correlogram