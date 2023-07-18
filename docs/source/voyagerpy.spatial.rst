Spatial functions
=================

.. module:: voyagerpy.spatial
   :synopsis: Spatial functions for Voyager data

.. currentmodule:: voyagerpy.spatial

Tissue segmentation
*******************

.. autosummary::
   :toctree: generated/spatial/tissue/

    detect_tissue_threshold
    get_approx_tissue_boundary
    get_tissue_boundary
    get_tissue_contour_score

Geometries
**********

.. autosummary::
    :toctree: generated/spatial/geometry/
    
    get_geom
    get_spot_coords
    get_visium_spots
    set_geometry
    to_points
    to_spatial_weights

Image transformation
********************

These functions allow rotating and mirroring the tissue images.
These functions perform the same transformations on the spot coordinates.

.. autosummary::
   :toctree: generated/spatial/transforms/

   apply_transforms
   cancel_transforms
   mirror_img
   rollback_transforms
   rotate_img90

Metrics and neighbours
**********************

.. autosummary::
   :toctree: generated/spatial/metrics/

   compute_spatial_lag
   local_moran
   losh
   moran

Graphs
******

.. autosummary::
   :toctree: generated/spatial/graphs/

   compute_correlogram
   compute_higher_order_neighbors
   get_default_graph
   set_default_graph