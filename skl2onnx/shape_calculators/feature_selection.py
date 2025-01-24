# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_select(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    i = operator.inputs[0]
    N = i.get_first_dimension()
    C = operator.raw_operator.get_support().sum()
    operator.outputs[0].type = i.type.__class__([N, C])


register_shape_calculator("SklearnGenericUnivariateSelect", calculate_sklearn_select)
register_shape_calculator("SklearnRFE", calculate_sklearn_select)
register_shape_calculator("SklearnRFECV", calculate_sklearn_select)
register_shape_calculator("SklearnSelectFdr", calculate_sklearn_select)
register_shape_calculator("SklearnSelectFpr", calculate_sklearn_select)
register_shape_calculator("SklearnSelectFromModel", calculate_sklearn_select)
register_shape_calculator("SklearnSelectFwe", calculate_sklearn_select)
register_shape_calculator("SklearnSelectKBest", calculate_sklearn_select)
register_shape_calculator("SklearnSelectPercentile", calculate_sklearn_select)
register_shape_calculator("SklearnVarianceThreshold", calculate_sklearn_select)
