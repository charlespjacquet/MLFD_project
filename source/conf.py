# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath(r'C:\Users\charl\OneDrive\VKI\MLFD\Project\MLFD_project'))

project = 'Reconstruction of flow field behind wind turbine with PIV data using radial basis functions'
copyright = '2025, Charles Jacquet'
author = 'Charles Jacquet'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autoapi.extension',    "sphinx.ext.mathjax"]
autoapi_type = 'python'
autoapi_dirs = ['../../']  # on remonte depuis source/ vers MLFD_project/
autoapi_ignore = ['*conf.py']
#autoapi_add_toctree_entry = False
autoapi_generate_api_docs = True



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
