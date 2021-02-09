# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
try:
    import collections.abc as cabc
except ImportError:
    import collections as cabc
import numpy as np
from ..common._apply_operation import (
    apply_cast, apply_add, apply_sqrt, apply_div, apply_sub,
    apply_reshape)
from ..common.data_types import (
    BooleanTensorType, Int64TensorType, DoubleTensorType,
    guess_numpy_type, guess_proto_type)
from ..common._registration import register_converter
from ..proto import onnx_proto


def convert_sklearn_linear_regressor(scope, operator, container):
    op = operator.raw_operator

    if type(operator.inputs[0].type) in (DoubleTensorType, ):
        proto_dtype = guess_proto_type(operator.inputs[0].type)
        coef = scope.get_unique_variable_name('coef')
        model_coef = op.coef_.T
        container.add_initializer(
            coef, proto_dtype, model_coef.shape, model_coef.ravel().tolist())
        intercept = scope.get_unique_variable_name('intercept')
        container.add_initializer(
            intercept, proto_dtype, op.intercept_.shape,
            op.intercept_.ravel().tolist())
        multiplied = scope.get_unique_variable_name('multiplied')
        container.add_node(
            'MatMul', [operator.inputs[0].full_name, coef], multiplied,
            name=scope.get_unique_operator_name('MatMul'))
        resh = scope.get_unique_variable_name('resh')
        apply_add(scope, [multiplied, intercept],
                  resh, container)
        last_dim = 1 if len(model_coef.shape) == 1 else model_coef.shape[-1]
        apply_reshape(scope, resh, operator.outputs[0].full_name,
                      container, desired_shape=(-1, last_dim))
        return

    op_type = 'LinearRegressor'
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype not in (np.float32, np.float64):
        dtype = np.float32
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs['coefficients'] = op.coef_.astype(dtype).ravel()
    attrs['intercepts'] = (op.intercept_.astype(dtype)
                           if isinstance(op.intercept_, cabc.Iterable)
                           else np.array([op.intercept_], dtype=dtype))
    if len(op.coef_.shape) == 2:
        attrs['targets'] = op.coef_.shape[0]

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container,
                   to=(onnx_proto.TensorProto.DOUBLE
                       if dtype == np.float64
                       else onnx_proto.TensorProto.FLOAT))
        input_name = cast_input_name
    container.add_node(op_type, input_name,
                       operator.outputs[0].full_name, op_domain='ai.onnx.ml',
                       **attrs)


def convert_sklearn_bayesian_ridge(scope, operator, container):
    convert_sklearn_linear_regressor(scope, operator, container)

    op = operator.raw_operator
    options = container.get_options(op, dict(return_std=False))
    return_std = options['return_std']
    if not return_std:
        return

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if op.normalize:
        # if self.normalize:
        #     X = (X - self.X_offset_) / self.X_scale_
        offset = scope.get_unique_variable_name('offset')
        container.add_initializer(
            offset, proto_dtype, op.X_offset_.shape,
            op.X_offset_.ravel().tolist())
        scale = scope.get_unique_variable_name('scale')
        container.add_initializer(
            scale, proto_dtype, op.X_scale_.shape,
            op.X_scale_.ravel().tolist())
        centered = scope.get_unique_variable_name('centered')
        apply_sub(scope, [operator.inputs[0].full_name, offset],
                  centered, container)
        scaled = scope.get_unique_variable_name('scaled')
        apply_div(scope, [centered, scale], scaled, container)
        input_name = scaled
    else:
        input_name = operator.inputs[0].full_name

    # sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
    sigma = scope.get_unique_variable_name('sigma')
    container.add_initializer(
        sigma, proto_dtype, op.sigma_.shape, op.sigma_.ravel().tolist())
    sigmaed0 = scope.get_unique_variable_name('sigma0')
    container.add_node(
        'MatMul', [input_name, sigma], sigmaed0,
        name=scope.get_unique_operator_name('MatMul'))
    sigmaed = scope.get_unique_variable_name('sigma')
    if container.target_opset < 13:
        container.add_node(
            'ReduceSum', sigmaed0, sigmaed, axes=[1],
            name=scope.get_unique_operator_name('ReduceSum'))
    else:
        axis_name = scope.get_unique_variable_name('axis')
        container.add_initializer(
            axis_name, onnx_proto.TensorProto.INT64, [1], [1])
        container.add_node(
            'ReduceSum', [sigmaed0, axis_name], sigmaed,
            name=scope.get_unique_operator_name('ReduceSum'))

    # y_std = np.sqrt(sigmas_squared_data + (1. / self.alpha_))
    # return y_mean, y_std
    std0 = scope.get_unique_variable_name('std0')
    alphainv = scope.get_unique_variable_name('alphainv')
    container.add_initializer(alphainv, proto_dtype, [1], [1 / op.alpha_])
    apply_add(scope, [sigmaed, alphainv], std0, container)
    apply_sqrt(scope, std0, operator.outputs[1].full_name, container)


register_converter('SklearnLinearRegressor', convert_sklearn_linear_regressor)
register_converter('SklearnLinearSVR', convert_sklearn_linear_regressor)
register_converter('SklearnBayesianRidge', convert_sklearn_bayesian_ridge,
                   options={'return_std': [True, False]})
