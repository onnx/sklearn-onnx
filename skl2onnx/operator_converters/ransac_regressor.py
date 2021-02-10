# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .._supported_operators import sklearn_operator_name_map
from ..common._apply_operation import apply_identity
from ..common._registration import register_converter


def convert_sklearn_ransac_regressor(scope, operator, container):
    """
    Converter for RANSACRegressor.
    """
    ransac_op = operator.raw_operator
    op_type = sklearn_operator_name_map[type(ransac_op.estimator_)]
    this_operator = scope.declare_local_operator(op_type, ransac_op.estimator_)
    this_operator.inputs = operator.inputs
    label_name = scope.declare_local_variable('label')
    this_operator.outputs.append(label_name)
    apply_identity(scope, label_name.full_name,
                   operator.outputs[0].full_name, container)


register_converter('SklearnRANSACRegressor',
                   convert_sklearn_ransac_regressor)
