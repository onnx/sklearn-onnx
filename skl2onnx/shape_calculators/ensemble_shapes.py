# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.shape_calculator import calculate_linear_regressor_output_shapes
from ..common.shape_calculator import calculate_linear_classifier_output_shapes


register_shape_calculator('SklearnDecisionTreeRegressor',
                          calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnRandomForestRegressor',
                          calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnExtraTreeRegressor',
                          calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnExtraTreesRegressor',
                          calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnGradientBoostingRegressor',
                          calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnHistGradientBoostingRegressor',
                          calculate_linear_regressor_output_shapes)

register_shape_calculator('SklearnDecisionTreeClassifier',
                          calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnRandomForestClassifier',
                          calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnExtraTreeClassifier',
                          calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnExtraTreesClassifier',
                          calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnGradientBoostingClassifier',
                          calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnHistGradientBoostingClassifier',
                          calculate_linear_classifier_output_shapes)
