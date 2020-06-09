# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
from ..common._registration import register_converter
from ..common._apply_operation import apply_cast
from ..proto import onnx_proto
from .common import concatenate_variables


def convert_sklearn_scaler(scope, operator, container):
    # If there are multiple input variables, we need to combine them as a
    # whole tensor. Integer(s) would be converted to float(s).
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
    elif isinstance(op, RobustScaler):
        C = operator.inputs[0].type.shape[1]
        attrs['offset'] = op.center_ if op.with_centering else [0.0] * C
        attrs['scale'] = 1.0 / op.scale_ if op.with_scaling else [1.0] * C
    elif isinstance(op, MinMaxScaler):
        attrs['scale'] = op.scale_
        # Add 1e-8 to avoid divided by 0
        attrs['offset'] = -op.min_/(op.scale_ + 1e-8)
    elif isinstance(op, MaxAbsScaler):
        C = operator.inputs[0].type.shape[1]
        attrs['scale'] = 1.0 / op.scale_
        attrs['offset'] = [0.] * C
    else:
        raise ValueError('Only scikit-learn StandardScaler and RobustScaler '
                         'are supported but got %s. You may raise '
                         'an issue at '
                         'https://github.com/onnx/sklearn-onnx/issues.'
                         '' % type(op))

    # ONNX does not convert arrays of float32.
    for k in attrs:
        v = attrs[k]
        if isinstance(v, np.ndarray) and v.dtype == np.float32:
            attrs[k] = v.astype(np.float64)

    options = container.get_options(op, dict(double=False))
    cast = options['double'] and container.dtype != np.float64
    if cast:
        feature_name_double = scope.get_unique_variable_name(
            'scaler_double')
        apply_cast(scope, feature_name, feature_name_double,
                   container, to=onnx_proto.TensorProto.DOUBLE)
        feature_name = feature_name_double

        scaled_feature_name_double = scope.get_unique_variable_name(
            'scaled_double')
        container.add_node(
            op_type, feature_name, scaled_feature_name_double,
            op_domain='ai.onnx.ml', **attrs)

        apply_cast(scope, scaled_feature_name_double,
                   operator.outputs[0].full_name,
                   container, to=container.proto_dtype)
    else:
        container.add_node(
            op_type, feature_name, operator.outputs[0].full_name,
            op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnRobustScaler', convert_sklearn_scaler,
                   options={'double': [False, True]})
register_converter('SklearnScaler', convert_sklearn_scaler,
                   options={'double': [False, True]})
register_converter('SklearnMinMaxScaler', convert_sklearn_scaler,
                   options={'double': [False, True]})
register_converter('SklearnMaxAbsScaler', convert_sklearn_scaler,
                   options={'double': [False, True]})
