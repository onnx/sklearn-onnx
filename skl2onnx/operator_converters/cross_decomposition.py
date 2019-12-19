# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_converter
from ..common.data_types import Int64TensorType
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxCast, OnnxDiv, OnnxMatMul, OnnxSub,
)


def convert_pls_regression(scope, operator, container):
    X = operator.inputs[0]
    op = operator.raw_operator
    opv = container.target_opset

    if type(X.type) == Int64TensorType:
        X = OnnxCast(X, to=container.proto_dtype, op_version=opv)

    norm_x = OnnxDiv(
                OnnxSub(X, op.x_mean_.astype(container.dtype),
                        op_version=opv),
                op.x_std_.astype(container.dtype),
                op_version=opv)
    dot = OnnxMatMul(norm_x, op.coef_.astype(container.dtype),
                     op_version=opv)
    pred = OnnxAdd(dot, op.y_mean_.astype(container.dtype),
                   op_version=opv, output_names=operator.outputs)
    pred.add_to(scope, container)


register_converter('SklearnPLSRegression', convert_pls_regression)
