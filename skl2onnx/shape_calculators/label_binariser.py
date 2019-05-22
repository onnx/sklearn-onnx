# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType, StringTensorType
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_label_binariser_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[
                                 Int64TensorType, StringTensorType])

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType(
        [N, len(operator.raw_operator.classes_)])


register_shape_calculator('SklearnLabelBinarizer',
                          calculate_sklearn_label_binariser_output_shapes)
