#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import libpysal
# need morean for data on obs, spatial and non spatial.
# then need moran for 
def moran(adata, gene):
    from esda.moran import Moran
    # get graph object
    wg = get_visium_graph(adata)
    # y = get gene data from adata.X.todense()

    y = get_correct_gene_form_from_genes(gene)
    mi = Moran(y, wg)
    mi.I


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



