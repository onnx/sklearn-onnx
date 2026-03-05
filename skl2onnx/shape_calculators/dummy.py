# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator
from ..common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from ..common.shape_calculator import calculate_linear_classifier_output_shapes


def calculate_dummy_regressor_output_shapes(operator):
    """
    Shape calculator for DummyRegressor.
    Output shape is [N, n_outputs_].
    """
    op = operator.raw_operator
    n_outputs = op.n_outputs_

    inp0 = operator.inputs[0].type
    if isinstance(inp0, (FloatTensorType, DoubleTensorType)):
        cls_type = inp0.__class__
    else:
        cls_type = FloatTensorType

    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = cls_type([N, n_outputs])


def calculate_dummy_classifier_output_shapes(operator):
    """
    Shape calculator for DummyClassifier.
    Delegates to calculate_linear_classifier_output_shapes which handles
    both integer and string class labels.
    """
    calculate_linear_classifier_output_shapes(operator)


register_shape_calculator(
    "SklearnDummyRegressor", calculate_dummy_regressor_output_shapes
)
register_shape_calculator(
    "SklearnDummyClassifier", calculate_dummy_classifier_output_shapes
)
