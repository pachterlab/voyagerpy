---
title: Visium Processing Workflows with VoyagerPy
layout: technology
name: Visium
published: true
---

The Visium spatial transcriptomics platform by 10X Genomics is based on Spatial Transcriptomics (ST) technology that was originally published in 2016. In both methods, mRNA from tissue sections is captured on spatially barcoded spots that are immobilized on a microarray slide. Following construction of a barcoded cDNA library, mRNA transcripts can be mapped to specific spots on the microarray slide and overlayed with a high-resolution image of the tissue, allowing for visualization and analysis of gene expression in a spatial context.

### Getting started
#### Download Data

Several publicly available Visium datasets are available from 10X Genomics on their [website](https://www.10xgenomics.com/resources/datasets).

### Analysis Workflows

The notebooks below demonstrate workflows that can be implemented with VoyagerPy using a variety of Visium datasets. The analysis tasks include basic quality control, spatial exploratory data analysis, identification of spatially variable genes, and computation of global and local spatial statistics.

<table class="table">
<thead>
    <tr>
      <th scope="col">Jupyter Notebook</th>
      <th scope="col">Description</th>
    </tr>
  </thead>
  <tbody>
  <tr>
  <td>
  <a href="{{ site.baseurl }}/notebooks/visium_10x.html" class="link-primary">Basic analysis of 10X example Visium dataset</a>
  </td>
	<td>Perform basic QC and standard non-spatial scRNA-seq analysis, and some spatial visualization</td>
  </tr>
  </tbody>
</table>