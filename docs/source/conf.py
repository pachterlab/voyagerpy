# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "VoyagerPy"
copyright = "2023, Pall Melsted, Sindri Emmanuel Antonsson, Petur Helgi Einarsson"
author = "Pall Melsted, Sindri Emmanuel Antonsson, Petur Helgi Einarsson"
release = "0.1"
version = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "matplotlib": ("https://matplotlib.org/3.6.3/", None),
}

intersphinx_disabled_domains = ["std"]

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_align": "content",
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 6,
    "external_links": [
        # {"name": "Notebooks", "url": "https://pmelsted.github.io/voyagerpy"},
        {"name": "Voyager - R", "url": "https://pachterlab.github.io/voyager"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pmelsted/voyagerpy",  # required
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}

html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_js_files = [
    "js/custom.js",
]
