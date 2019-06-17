# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Common errors.
"""


_missing_converter = """
It usually means the pipeline being converted contains a
transformer or a predictor with no corresponding converter
implemented in sklearn-onnx. If the converted is implemented
in another library (ie: onnxmltools), you need to register
the converted so that it can be used by sklearn-onnx (function
update_registered_converter). If the model is not yet covered
by sklearn-onnx, you may raise an issue to
https://github.com/onnx/sklearn-onnx/issues
to get the converter implemented or even contribute to the
project. If the model is a custom model, a new converter must
be implemented. Examples can be found in the gallery.
"""


class MissingShapeCalculator(RuntimeError):
    """
    Raised when there is no registered shape calculator
    for a machine learning operator.
    """
    def __init__(self, msg):
        super().__init__(msg + _missing_converter)


class MissingConverter(RuntimeError):
    """
    Raised when there is no registered converter
    for a machine learning operator. If the model is
    part of scikit-learn, you may raise an issue at
    https://github.com/onnx/sklearn-onnx/issues.
    """
    def __init__(self, msg):
        super().__init__(msg + _missing_converter)
