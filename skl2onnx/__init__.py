# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Main entry point to the converter from the scikit-learn to onnx.
"""
__version__ = "1.4.0"
__author__ = "Microsoft"
__producer__ = "skl2onnx"
__producer_version__ = __version__
__domain__ = "onnxml"
__model_version__ = 0


from .convert import convert
