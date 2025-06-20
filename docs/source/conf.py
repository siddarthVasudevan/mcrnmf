# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "mcrnmf"
copyright = "2025, Siddarth A. Vasudevan"
author = "Siddarth A. Vasudevan"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "numpydoc",
]

# Specify how to identify the prompt when copying code snippets
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_exclude = "style"

numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "../tests/**"]
source_suffix = ".rst"
master_doc = "index"
language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "show_nav_level": 3,
    "show_toc_level": 2,
    "navbar_align": "left",
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "show_version_warning_banner": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}
html_logo = "_static/logo.svg"

autosummary_generate = True
