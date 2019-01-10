# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_concat(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type.shape = [N, 'None']


register_shape_calculator('SklearnConcat', calculate_sklearn_concat)
register_shape_calculator('SklearnGenericUnivariateSelect', calculate_sklearn_concat)
register_shape_calculator('SklearnRFE', calculate_sklearn_concat)
register_shape_calculator('SklearnRFECV', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFdr', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFpr', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFromModel', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFwe', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectKBest', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectPercentile', calculate_sklearn_concat)
register_shape_calculator('SklearnVarianceThreshold', calculate_sklearn_concat)
