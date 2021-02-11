# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._apply_operation import apply_cast
from ..common._registration import register_converter
from .._supported_operators import sklearn_operator_name_map


def convert_sklearn_cast(scope, operator, container):
    inp = operator.inputs[0]
    exptype = operator.outputs[0]
    res = exptype.type.to_onnx_type()
    et = res.tensor_type.elem_type
    apply_cast(scope, inp.full_name, exptype.full_name,
               container, to=et)


def convert_sklearn_cast_regressor(scope, operator, container):

    op = operator.raw_operator
    estimator = op.estimator

    op_type = sklearn_operator_name_map[type(estimator)]
    this_operator = scope.declare_local_operator(op_type, estimator)
    this_operator.inputs = operator.inputs

    cls = operator.inputs[0].type.__class__
    var_name = scope.declare_local_variable('cast_est', cls())
    this_operator.outputs.append(var_name)
    var_name = var_name.onnx_name

    exptype = operator.outputs[0]
    res = exptype.type.to_onnx_type()
    et = res.tensor_type.elem_type
    apply_cast(scope, var_name, exptype.full_name,
               container, to=et)


register_converter('SklearnCastTransformer', convert_sklearn_cast)
register_converter('SklearnCastRegressor', convert_sklearn_cast_regressor)
register_converter('SklearnCast', convert_sklearn_cast)
