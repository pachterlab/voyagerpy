#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# need morean for data on obs, spatial and non spatial.
# then need moran for 
def moran(adata, gene):
    # get graph object
    vg = get_visium_graph(adata)
    # y = get gene data from adata.X.todense()

    # calculate for one, for all or a subset.
    # mi = Moran(y, w)

    # return for one or many genes


function_map = {
  'moran': calcChurn,
  'geary': calcEngagement,
  'moran.mc':,
  'geary.mc',
  'moran.test',
  'geary.test',
  'globalG.test',
  'sp.correlogram',
  'moran.plot',
  'localmoran',
  'localmoran_perm',
  'localC',
  'localC_perm',
  'localG',
  'localG_perm',
  'LOSH',
  'LOSH.mc',
  'LOSH.cs'
  
}

def calculate_univariate(
        adata,
        fun_type,
        features,
        barcode_graph_name,
        zero_policy=None,
        in_place=False,
        include_self=None,
        p_adjust_method=None
):
    