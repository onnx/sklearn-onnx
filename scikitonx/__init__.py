# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Main entry point to the converter from the scikit-learn to onnx.
"""
__version__ = "1.3.1"
__author__ = "Microsoft"
__producer__ = "scikitonx"
__producer_version__ = __version__
__domain__ = "onnxml"
__model_version__ = 0

from .convert import convert
from .common import utils

def convert_sklearn(model, name=None, initial_types=None, doc_string='',
                    target_opset=None, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.sklearn_installed():
        raise RuntimeError('scikit-learn is not installed. Please install scikit-learn to use this feature.')

    from scikitonx.scikitonx.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset,
                   custom_conversion_functions, custom_shape_calculators)


from .utils import load_model
from .utils import save_model
from .utils import save_text
