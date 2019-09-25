# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._topology import FloatTensorType
from ..common._registration import register_converter
from ..common._apply_operation import apply_mul
from .._supported_operators import sklearn_operator_name_map
from ..proto import onnx_proto


def convert_voting_regressor(scope, operator, container):
    """
    Converts a *VotingRegressor* into *ONNX* format.
    """
    op = operator.raw_operator

    vars_names = []
    for i, estimator in enumerate(op.estimators_):
        if estimator is None:
            continue

        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type)
        this_operator.raw_operator = estimator
        this_operator.inputs = operator.inputs

        var_name = scope.declare_local_variable(
            'var_%d' % i, FloatTensorType())
        this_operator.outputs.append(var_name)
        var_name = var_name.onnx_name

        if op.weights is not None:
            val = op.weights[i] / op.weights.sum()
        else:
            val = 1. / len(op.estimators_)

        weights_name = scope.get_unique_variable_name('w%d' % i)
        container.add_initializer(
            weights_name, onnx_proto.TensorProto.FLOAT, [1], [val])
        wvar_name = scope.get_unique_variable_name('wvar_%d' % i)
        apply_mul(scope, [var_name, weights_name],
                  wvar_name, container, broadcast=1)

        flat_name = scope.get_unique_variable_name('fvar_%d' % i)
        container.add_node('Flatten', wvar_name, flat_name)
        vars_names.append(flat_name)

    container.add_node('Sum', vars_names,
                       operator.outputs[0].full_name,
                       name=scope.get_unique_operator_name('Sum'))


register_converter('SklearnVotingRegressor', convert_voting_regressor)
