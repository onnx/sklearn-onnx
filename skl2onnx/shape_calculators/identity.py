# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_identity(operator):
    check_input_and_output_numbers(operator, input_count_range=1,
                                   output_count_range=1)
    operator.outputs[0].type = operator.inputs[0].type


register_shape_calculator('SklearnIdentity', calculate_sklearn_identity)
