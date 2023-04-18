# This is the GitHub pages branch for VoyagerPy

This document is to serve as a guide to editing this GitHub page. 

## General overview of the repository structure

```
.
├── 404.html
├── Gemfile
├── Gemfile.lock
├── README.md # <- Your are here!
├── _config.yml # The config file for this page
├── _includes # Here are all the html document we want to embed in a page
│   ├── navbar.html
│   └── notebooks
│       ├── nonspatial.html
│       ├── nonspatial_files
│       │   ├── ...
│       │   ├── nonspatial_95_0.png
│       │   └── nonspatial_98_0.png
│       ├── visium_10x.html
│       └── visium_10x_files
│           ├── ...
│           ├── visium_10x_40_0.png
│           └── visium_10x_6_0.png
├── _layouts  # Here we define our layouts/templates for the pages
│   ├── basic.md  # we don't really use this
│   ├── default.html # this what we use as as base for the others
│   ├── notebook.html # use this layout for notebooks
│   └── technology.html # Use this layout for the landing pages of each technology
├── _navitems # Published pages of this directory is shown in the navbar
│   ├── about.markdown
│   ├── changelog.md
│   ├── docs.md
│   ├── install.md  # The installation page
│   ├── notebooks.md # Was supposed to be the page with links to all the notebooks...deprecated
│   └── technologies.html # This is a special one, it only needs to be there
├── _site # this is auto-generated
│   └── ...
├── _technologies # These are the landing pages for each tech
│   ├── 10x_chromium_v3.md
│   ├── cosmx.md
│   ├── merfish.md
│   ├── seqfish.md
│   ├── slideseq_v2.md
│   ├── visium.md
│   └── xenium.md
├── index.md  # the front page
├── jekyll_toc  # This is for the table of contents
│   ├── LICENSE.txt
│   ├── README.md
│   └── toc.js
├── notebooks  # These are the actual pages for each notebook. These should embed the actual notebook.html files.
│   ├── nonspatial.html
│   └── visium_10x.html
└── styles  # These are the styles, the ipynb*.css were extracted from the converted notebooks and saved as files.
    ├── ipynb1.css
    ├── ipynb2.css
    ├── ipynb3.css
    ├── ipynb4.css
    └── main.css  # we could probably get rid of this one.
```

## Adding a notebook for an existing technology

In the `notebooks` directory, create a HTML file with the following lines:

```
---
layout: notebook
name: The name of the notebook
title: The title of the notebook
tech: the technology used. e.g. chromium, visium, etc. This probably has no use any more.
---

Add some content here if you really need to. Otherwise, it should be in the notebook.

{% include notebooks/notebookfile.html %}

Same as above...

```

## Adding a landing page for a new technology

In the `_technologies` directory, create a markdown file for your landing page. At the top, include the following front matter:

```
---
title: The title for your landing page
layout: technology
name: The name of the technology
published: true
order: 42
---
```

Make sure to have `published: true` for the page to be created and included in the technologies drowdown. The `order` field is for the order in which the page appears in the technology dropdown.

## Running things locally

Follow the instructions for [Jekyll on GitHub pages]() to install Ruby and Jekyll. Anyway, I used RVM since MacOS and ruby are not the best of friends, and installed Ruby 3.1.3.