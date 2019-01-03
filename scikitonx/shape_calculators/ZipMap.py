# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_zipmap(operator):
    check_input_and_output_numbers(operator, output_count_range=2)


register_shape_calculator('SklearnZipMap', calculate_sklearn_zipmap)
