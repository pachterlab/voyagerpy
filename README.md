# This is the GitHub pages branch for VoyagerPy

This document is to serve as a guide to editing this GitHub page. 

## General overview of the repository structure

```
./docs/
├── .nojekyll                    # Let GitHub know we're not using Jekyll
├── make.bat                     # Makefile for windows
├── Makefile                     # Makefile
├── requirements.txt             # Requirements file for the docs
└── source
    ├── _static
    │   ├── css
    │   │   └── custom.css        # Custom CSS. Defines mostly just color variables
    │   └── js
    │       └── custom.js         # Change the ugly pandas tables generated.
    ├── _templates
    ├── api.rst                   # The main document for the API page
    ├── conf.py                 # Configuration file for sphinx
    ├── examples                  # These are synced from the main branch where they are populated
    │   └── ...ipynb
    ├── index.rst                 # The index page
    ├── installation.rst          # Installation page
    ├── technologies              # Here we have the landing pages for each technology
    │   ├── 10xatac.rst
    │   ├── 10xchromium.rst
    │   ├── 10xcrispr.rst
    │   ├── 10xmultiome.rst
    │   ├── 10xnuclei.rst
    │   ├── codex.rst
    │   ├── cosmx.rst
    │   ├── merfish.rst
    │   ├── seqfish.rst
    │   ├── slideseqV2.rst
    │   ├── splitseq.rst
    │   ├── visium.rst
    │   └── xenium.rst
    ├── technologies.rst          # The technology page, referencing all the technologies
    ├── voyagerpy.plotting.rst    # plotting docs
    ├── voyagerpy.spatial.rst     # spatial docs
    ├── voyagerpy.utils.hvg.rst      # hvg docs
    ├── voyagerpy.utils.markers.rst  # markers docs
    ├── voyagerpy.utils.rst          # utils overview
    └── voyagerpy.utils.utils.rst    # utils.utils docs
```

## Adding a notebook for an existing technology

Add the `.ipynb` file to the `examples` folder on the `main` branch. Then, reference the notebook in the 
corresponding technology document (e.g. `` `name for link <../examples/notebook.ipynb>`_`` for link and `name in toc <../examples/notebook.ipynb>` for the toctree). Make sure you add it both to the `toctree`, and in the `list-table`.

## Adding a landing page for a new technology

In the `docs/source/technologies` directory, create a `.rst` file for your landing page. You can follow the general structure for the other landing pages. Make sure to add the new technology in the toctree of `docs/source/technologies.rst`

## Running things locally

You can install the requirements by running

```pip install -r docs/requirements```

You might have to install [pandoc](https://pandoc.org/installing.html), installable via binaries, `pip`, `conda`, or from source.
You can also check out their [GitHub page](https://github.com/jgm/pandoc).

To build the documentation, you can run `make html` in the `docs` directory, or just `make -C docs html` in the root.

To view the results, simply open the `docs/_site/html/index.html`. If you want to automatically build the docs, install
`sphinx-autobuild` with

```python3 -m pip install sphinx-autobuild```

and run it via

```python3 -m sphinx_autobuild docs/source docs/_site```.
