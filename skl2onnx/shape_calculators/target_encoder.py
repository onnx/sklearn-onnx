# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType
from ..common.data_types import Int64TensorType, StringTensorType
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_target_encoder_output_shapes(operator):
    """
    This function just copy the input shape to the output because target
    encoder only alters input features' values, not their shape.
    """
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, StringTensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    shape = [N, len(operator.raw_operator.categories_)]

    operator.outputs[0].type = FloatTensorType(shape=shape)


register_shape_calculator(
    "SklearnTargetEncoder", calculate_sklearn_target_encoder_output_shapes
)
