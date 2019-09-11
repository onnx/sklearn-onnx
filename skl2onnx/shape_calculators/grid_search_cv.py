# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.shape_calculator import calculate_linear_classifier_output_shapes
from ..common.shape_calculator import calculate_linear_regressor_output_shapes
from .._supported_operators import sklearn_classifier_list


def convert_sklearn_grid_search_cv(operator):
    grid_search_op = operator.raw_operator
    best_estimator = grid_search_op.best_estimator_
    if type(best_estimator) in sklearn_classifier_list:
        calculate_linear_classifier_output_shapes(operator)
    else:
        calculate_linear_regressor_output_shapes(operator)


register_shape_calculator('SklearnGridSearchCV',
                          convert_sklearn_grid_search_cv)
