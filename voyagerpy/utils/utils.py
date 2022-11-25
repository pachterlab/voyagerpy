#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def is_highres(adata):
    if("hires" in adata.uns["spatial"]["img"]):
        return True
    if("lowres" in adata.uns["spatial"]["img"]):
        return False
    raise ValueError(
        "Cannot find image data in .uns['spatial']"
    )


