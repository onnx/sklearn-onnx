# SPDX-License-Identifier: Apache-2.0

import copy
from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers, check_input_and_output_types
from ..common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType


def quantile_transformer_shape_calculator(operator):
    """Shape calculator for QuantileTransformer"""
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    model = operator.raw_operator
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, model.quantiles_.shape[1]]


register_shape_calculator(
    "SklearnQuantileTransformer", quantile_transformer_shape_calculator
)
