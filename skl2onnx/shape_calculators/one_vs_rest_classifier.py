# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType
from ..common.shape_calculator import calculate_linear_classifier_output_shapes


def calculate_constant_predictor_output_shapes(operator):
    N = operator.inputs[0].get_first_dimension()
    op = operator.raw_operator
    nc = op.y_.shape[1] if len(op.y_.shape) > 1 else 1
    operator.outputs[0].type = Int64TensorType([N, nc] if nc > 1 else [N])
    operator.outputs[1].type.shape = [N, nc]


register_shape_calculator('Sklearn_ConstantPredictor',
                          calculate_constant_predictor_output_shapes)

register_shape_calculator('SklearnOneVsRestClassifier',
                          calculate_linear_classifier_output_shapes)
