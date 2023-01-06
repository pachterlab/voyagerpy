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
  'moran': moran,
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
    if(fun_type in function_map):
        val = function_map[fun_type](adata, features)

    return val

import libpysal

w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
y = np.array(f.by_col["HR8893"])
from esda.moran import Moran

y
w
w1, w2 = w.full()
mi = Moran(y, w)
mi.I