# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .onnx_operator import OP, OnnxOperator


class Div(OnnxOperator):
    "See `Div <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div>`_"
    pass


class Gemm(OnnxOperator):
    "See `Gemm <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm>`_"
    pass


class Mul(OnnxOperator):
    "See `Mul <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul>`_"
    pass


class Sub(OnnxOperator):
    "See `Sub <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub>`_"
    pass

