# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType


def calculate_quadratic_discriminant_analysis_shapes(operator):
    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = Int64TensorType([1, N])
    operator.outputs[1].type.shape = [N, ]


register_shape_calculator(
    'SklearnQuadraticDiscriminantAnalysis',
    calculate_quadratic_discriminant_analysis_shapes)
