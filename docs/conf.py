# SPDX-License-Identifier: Apache-2.0


# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import warnings
import skl2onnx
import pydata_sphinx_theme

sys.path.append(os.path.abspath('exts'))
from github_link import make_linkcode_resolve  # noqa


# -- Project information -----------------------------------------------------

project = 'sklearn-onnx'
copyright = '2018-2022, Microsoft'
author = 'Microsoft'
version = skl2onnx.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx_skl2onnx_extension',
    'matplotlib.sphinxext.plot_directive',
    'pyquickhelper.sphinxext.sphinx_cmdref_extension',
    'pyquickhelper.sphinxext.sphinx_collapse_extension',
    'pyquickhelper.sphinxext.sphinx_docassert_extension',
    'pyquickhelper.sphinxext.sphinx_epkg_extension',
    'pyquickhelper.sphinxext.sphinx_exref_extension',
    'pyquickhelper.sphinxext.sphinx_faqref_extension',
    'pyquickhelper.sphinxext.sphinx_gdot_extension',
    'pyquickhelper.sphinxext.sphinx_runpython_extension',
    "sphinxcontrib.blockdiag",
]

templates_path = ['_templates']
source_suffix = ['.rst']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_mo"
html_static_path = ['_static']
html_theme = "pydata_sphinx_theme"
html_theme_path = pydata_sphinx_theme.get_html_theme_path()
html_logo = "logo_main.png"

# -- Options for graphviz ----------------------------------------------------

graphviz_output_format = "svg"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for Sphinx Gallery ----------------------------------------------

linkcode_resolve = make_linkcode_resolve(
    'skl2onnx',
    'https://github.com/onnx/skl2onnx/blob/{revision}/'
    '{package}/{path}#L{lineno}')

intersphinx_mapping = {
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'mlinsights': (
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/', None),
    'mlprodict': (
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pyquickhelper': (
        'http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
    'onnxruntime': ('https://onnxruntime.ai/docs/api/python/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'scikit-learn': (
        'https://scikit-learn.org/stable/',
        None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'skl2onnx': ('https://onnx.ai/sklearn-onnx/', None),
    'sklearn-onnx': ('https://onnx.ai/sklearn-onnx/', None),
}

sphinx_gallery_conf = {
    'examples_dirs': ['examples', 'tutorial'],
    'gallery_dirs': ['auto_examples', 'auto_tutorial'],
    'capture_repr': ('_repr_html_', '__repr__'),
    'ignore_repr_types': r'matplotlib.text|matplotlib.axes',
    'binder': {
        'org': 'onnx',
        'repo': 'onnx.ai/sklearn-onnx/',
        'notebooks_dir': 'auto_examples',
        'binderhub_url': 'https://mybinder.org',
        'branch': 'master',
        'dependencies': './requirements.txt'
    },
}

epkg_dictionary = {
    'C': 'https://en.wikipedia.org/wiki/C_(programming_language)',
    'C++': 'https://en.wikipedia.org/wiki/C%2B%2B',
    'cython': 'https://cython.org/',
    'DOT': 'https://www.graphviz.org/doc/info/lang.html',
    'ImageNet': 'http://www.image-net.org/',
    'LightGBM': 'https://lightgbm.readthedocs.io/en/latest/',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/latest/',
    'mlprodict':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'NMF':
        'https://scikit-learn.org/stable/modules/generated/'
        'sklearn.decomposition.NMF.html',
    'numpy': 'https://numpy.org/',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'ONNX operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators.md',
    'ONNX ML operators':
        'https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md',
    'onnxmltools': 'https://github.com/onnx/onnxmltools',
    'OnnxPipeline':
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/'
        'sklapi/onnx_pipeline.html?highlight=onnxpipeline',
    'onnxruntime': 'https://microsoft.github.io/onnxruntime/',
    'openmp': 'https://en.wikipedia.org/wiki/OpenMP',
    'pyinstrument': 'https://github.com/joerick/pyinstrument',
    'python': 'https://www.python.org/',
    'pytorch': 'https://pytorch.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'skorch': 'https://skorch.readthedocs.io/en/stable/',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
    'sphinx-gallery': 'https://github.com/sphinx-gallery/sphinx-gallery',
    'xgboost': 'https://xgboost.readthedocs.io/en/latest/',
    'XGBoost': 'https://xgboost.readthedocs.io/en/latest/',
}

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Setup actions -----------------------------------------------------------


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    return app
