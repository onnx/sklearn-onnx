# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_zipmap(operator):
    check_input_and_output_numbers(operator, output_count_range=2)
    operator.outputs[0].type = operator.inputs[0].type.__class__(
        operator.inputs[0].type.shape)


def calculate_sklearn_zipmap_columns(operator):
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = operator.inputs[0].type.__class__(
        operator.inputs[0].type.shape)
    for i in range(1, len(operator.outputs)):
        operator.outputs[i].type.shape = [N]


register_shape_calculator('SklearnZipMap', calculate_sklearn_zipmap)
register_shape_calculator(
    'SklearnZipMapColumns', calculate_sklearn_zipmap_columns)
