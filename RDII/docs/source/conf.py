import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

project = 'RDII Analysis'
author = 'Grace Inman'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',      # pulls docstrings automatically
    'sphinx.ext.napoleon',     # handles NumPy style docstrings
    'sphinx.ext.viewcode',     # adds links to source code
    'sphinx.ext.autosummary',  # generates summary tables
]

html_theme = 'sphinx_rtd_theme'  # the nice ReadTheDocs theme

napoleon_numpy_docstring = True
napoleon_google_docstring = False