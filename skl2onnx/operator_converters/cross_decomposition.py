# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common.data_types import (
    Int64TensorType, guess_numpy_type, guess_proto_type)
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxCast, OnnxDiv, OnnxMatMul, OnnxSub)


def convert_pls_regression(scope, operator, container):
    X = operator.inputs[0]
    op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    if type(X.type) == Int64TensorType:
        X = OnnxCast(X, to=proto_dtype, op_version=opv)

    norm_x = OnnxDiv(
                OnnxSub(X, op.x_mean_.astype(dtype),
                        op_version=opv),
                op.x_std_.astype(dtype),
                op_version=opv)
    dot = OnnxMatMul(norm_x, op.coef_.astype(dtype),
                     op_version=opv)
    pred = OnnxAdd(dot, op.y_mean_.astype(dtype),
                   op_version=opv, output_names=operator.outputs)
    pred.add_to(scope, container)


register_converter('SklearnPLSRegression', convert_pls_regression)
