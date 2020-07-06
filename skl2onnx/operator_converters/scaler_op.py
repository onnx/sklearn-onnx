# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
from ..algebra.onnx_ops import OnnxSub, OnnxDiv, OnnxCast
from ..common._registration import register_converter
from ..common.data_types import guess_numpy_type, guess_proto_type
from ..proto import onnx_proto
from .common import concatenate_variables


def convert_sklearn_scaler(scope, operator, container):
    # If there are multiple input variables, we need to combine them as a
    # whole tensor. Integer(s) would be converted to float(s).
    # Options div use true division instead of Scaler operator
    # which replaces a division by a multiplication.
    # This leads to discrepencies in some cases.
    if len(operator.inputs) > 1:
        feature_name = concatenate_variables(scope, operator.inputs, container)
    else:
        feature_name = operator.inputs[0].full_name

    op = operator.raw_operator
    op_type = 'Scaler'
    attrs = {'name': scope.get_unique_operator_name(op_type)}

    if isinstance(op, StandardScaler):
        C = operator.inputs[0].type.shape[1]
        attrs['offset'] = op.mean_ if op.with_mean else [0.0] * C
        attrs['scale'] = 1.0 / op.scale_ if op.with_std else [1.0] * C
        inv_scale = op.scale_ if op.with_std else None
    elif isinstance(op, RobustScaler):
        C = operator.inputs[0].type.shape[1]
        attrs['offset'] = op.center_ if op.with_centering else [0.0] * C
        attrs['scale'] = 1.0 / op.scale_ if op.with_scaling else [1.0] * C
        inv_scale = op.scale_ if op.with_scaling else None
    elif isinstance(op, MinMaxScaler):
        attrs['scale'] = op.scale_
        # Add 1e-8 to avoid divided by 0
        attrs['offset'] = -op.min_/(op.scale_ + 1e-8)
        inv_scale = None
    elif isinstance(op, MaxAbsScaler):
        C = operator.inputs[0].type.shape[1]
        attrs['scale'] = 1.0 / op.scale_
        attrs['offset'] = [0.] * C
        inv_scale = op.scale_
    else:
        raise ValueError('Only scikit-learn StandardScaler and RobustScaler '
                         'are supported but got %s. You may raise '
                         'an issue at '
                         'https://github.com/onnx/sklearn-onnx/issues.'
                         '' % type(op))

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    for k in attrs:
        v = attrs[k]
        if isinstance(v, np.ndarray) and v.dtype != dtype:
            attrs[k] = v.astype(dtype)

    if inv_scale is not None:
        options = container.get_options(op, dict(div='std'))
        div = options['div']
        if div == 'div':
            opv = container.target_opset
            sub = OnnxSub(
                feature_name, attrs['offset'].astype(dtype),
                op_version=opv)
            div = OnnxDiv(sub, inv_scale.astype(dtype),
                          op_version=opv,
                          output_names=[operator.outputs[0].full_name])
            div.add_to(scope, container)
            return
        if div == 'div_cast':
            opv = container.target_opset
            cast = OnnxCast(feature_name, to=onnx_proto.TensorProto.DOUBLE,
                            op_version=opv)
            sub = OnnxSub(cast, attrs['offset'].astype(np.float64),
                          op_version=opv)
            div = OnnxDiv(sub, inv_scale.astype(np.float64), op_version=opv)
            cast = OnnxCast(div, to=proto_dtype, op_version=opv,
                            output_names=[operator.outputs[0].full_name])
            cast.add_to(scope, container)
            return

    container.add_node(
        op_type, feature_name, operator.outputs[0].full_name,
        op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnRobustScaler', convert_sklearn_scaler,
                   options={'div': ['std', 'div', 'div_cast']})
register_converter('SklearnScaler', convert_sklearn_scaler,
                   options={'div': ['std', 'div', 'div_cast']})
register_converter('SklearnMinMaxScaler', convert_sklearn_scaler)
register_converter('SklearnMaxAbsScaler', convert_sklearn_scaler,
                   options={'div': ['std', 'div', 'div_cast']})
