# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType
)
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_nearest_neighbours(operator):
    check_input_and_output_numbers(operator, input_count_range=1,
                                   output_count_range=[1, 2])
    check_input_and_output_types(
        operator, good_input_types=[
            FloatTensorType, Int64TensorType, DoubleTensorType])

    N = operator.inputs[0].type.shape[0]
    neighbours = operator.raw_operator.n_neighbors
    operator.outputs[0].type = Int64TensorType([N, neighbours])
    operator.outputs[1].type.shape = [N, neighbours]


register_shape_calculator('SklearnNearestNeighbors',
                          calculate_sklearn_nearest_neighbours)
