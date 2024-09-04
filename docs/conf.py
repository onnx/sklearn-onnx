# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.

import os
import sys
import logging
import warnings
import skl2onnx

sys.path.append(os.path.abspath("exts"))
from github_link import make_linkcode_resolve


# -- Project information -----------------------------------------------------

project = "sklearn-onnx"
copyright = "2018-2023, Microsoft"
author = "Microsoft"
version = skl2onnx.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx_skl2onnx_extension",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.blocdefs.sphinx_exref_extension",
    "sphinx_runpython.blocdefs.sphinx_faqref_extension",
    "sphinx_runpython.blocdefs.sphinx_mathdef_extension",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
    "sphinxcontrib.blockdiag",
]

templates_path = ["_templates"]
source_suffix = [".rst"]

master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_theme = "furo"
html_logo = "logo_main.png"

# -- Options for graphviz ----------------------------------------------------

graphviz_output_format = "svg"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/": None}

# -- Options for Sphinx Gallery ----------------------------------------------

linkcode_resolve = make_linkcode_resolve(
    "skl2onnx",
    "https://github.com/onnx/skl2onnx/blob/{revision}/{package}/{path}#L{lineno}",
)

intersphinx_mapping = {
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skl2onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "sklearn-onnx": ("https://onnx.ai/sklearn-onnx/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": ["examples", "tutorial"],
    "gallery_dirs": ["auto_examples", "auto_tutorial"],
    "capture_repr": ("_repr_html_", "__repr__"),
    "ignore_repr_types": r"matplotlib.text|matplotlib.axes",
    # 'binder': {
    #     'org': 'onnx',
    #     'repo': 'onnx.ai/sklearn-onnx/',
    #     'notebooks_dir': 'auto_examples',
    #     'binderhub_url': 'https://mybinder.org',
    #     'branch': 'main',
    #     'dependencies': './requirements.txt'
    # },
}

epkg_dictionary = {
    "C": "https://en.wikipedia.org/wiki/C_(programming_language)",
    "C++": "https://en.wikipedia.org/wiki/C%2B%2B",
    "cython": "https://cython.org/",
    "DOT": "https://www.graphviz.org/doc/info/lang.html",
    "ImageNet": "http://www.image-net.org/",
    "LightGBM": "https://lightgbm.readthedocs.io/en/latest/",
    "lightgbm": "https://lightgbm.readthedocs.io/en/latest/",
    "NMF": "https://scikit-learn.org/stable/modules/generated/"
    "sklearn.decomposition.NMF.html",
    "numpy": "https://numpy.org/",
    "onnx": "https://github.com/onnx/onnx",
    "ONNX": "https://onnx.ai/",
    "ONNX operators": "https://onnx.ai/onnx/operators/",
    "ONNX ML operators": "https://onnx.ai/onnx/operators/",
    "ONNX ML Operators": "https://onnx.ai/onnx/operators/",
    "onnxmltools": "https://github.com/onnx/onnxmltools",
    "onnxruntime": "https://microsoft.github.io/onnxruntime/",
    "openmp": "https://en.wikipedia.org/wiki/OpenMP",
    "pyinstrument": "https://github.com/joerick/pyinstrument",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "skorch": "https://skorch.readthedocs.io/en/stable/",
    "skl2onnx": "https://github.com/onnx/sklearn-onnx",
    "sklearn-onnx": "https://github.com/onnx/sklearn-onnx",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "xgboost": "https://xgboost.readthedocs.io/en/latest/",
    "XGBoost": "https://xgboost.readthedocs.io/en/latest/",
}

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Setup actions -----------------------------------------------------------


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    logger = logging.getLogger("skl2onnx")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("matplotlib.font_manager")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("matplotlib.ticker")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("PIL.PngImagePlugin")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("graphviz._tools")
    logger.setLevel(logging.WARNING)
    return app
