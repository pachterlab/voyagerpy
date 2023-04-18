---
title: 10X Chromium Single Cell 3' V3 Processing Workflows with VoyagerPy
layout: technology
name: 10X Chromium Single Cell 3'
published: True
order: 7
---


<p class="text-justify">
The 10X Genomics Chromium Single Cell 3' v3 solution is a droplet-based single cell sequencing technology that enables gene expression profiling of hundreds to tens of thousands of cells. In a typical protocol, cells are dissociated and captured in Gel Beads-in-emulsion (GEMs). All cDNA from a single cell share the same barcode and additional 10X barcodes help associate individual reads from generated libraries back to the originating GEMs. Though spatial information is lost, the technology has diverse applications, including cell surface protein profiling and assessing the effects of cellular pertubations.</p>

### Getting started
#### Download Data

Several publicly available datasets are available from 10X Genomics on their [website](https://www.10xgenomics.com/resources/datasets).

### Analysis Workflows

<p class="text-justify">
The notebooks below demonstrate workflows that can be implemented with VoyagerPy using a 10X v3 datasets. The analysis tasks include basic quality control, spatial exploratory data analysis, identification of spatially variable genes, and computation of global and local spatial statistics.</p>

<table class="table table-hover">
<caption>Available notebooks</caption>
<thead>
    <tr>
      <th scope="col">Jupyter Notebook</th>
      <th scope="col">Description</th>
    </tr>
  </thead>
  <tbody>
  <tr>
  <td>
  <a href="{{ site.baseurl }}/notebooks/nonspatial.html" class="link-primary">10X v3 Basic</a>
  </td>
	<td>"Spatial" analyses for non-spatial scRNA-seq data.</td>
  </tr>
  </tbody>
</table>
