# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator
from ..common.shape_calculator import (
    _calculate_linear_classifier_output_shapes)


def voting_classifier_shape_calculator(operator):
    return _calculate_linear_classifier_output_shapes(
        operator, enable_type_checking=False)


register_shape_calculator(
    'SklearnVotingClassifier', voting_classifier_shape_calculator)
