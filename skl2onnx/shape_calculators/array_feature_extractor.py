# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_array_feature_extractor(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    i = operator.inputs[0]
    N = i.type.shape[0]
    C = len(operator.column_indices)
    operator.outputs[0].type = i.type.__class__([N, C])


register_shape_calculator('SklearnArrayFeatureExtractor',
                          calculate_sklearn_array_feature_extractor)
