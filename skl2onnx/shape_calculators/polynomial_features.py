# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ..common.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType
)
from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_polynomial_features(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[
            FloatTensorType, Int64TensorType, DoubleTensorType])

    N = operator.inputs[0].type.shape[0]
    model = operator.raw_operator
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, model.n_output_features_]


register_shape_calculator('SklearnPolynomialFeatures',
                          calculate_sklearn_polynomial_features)
