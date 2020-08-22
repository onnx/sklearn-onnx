# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType


def calculate_isolation_forest_output_shapes(operator):
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType([N, 1])
    operator.outputs[1].type.shape = [N, 1]


register_shape_calculator(
    'SklearnIsolationForest', calculate_isolation_forest_output_shapes)
