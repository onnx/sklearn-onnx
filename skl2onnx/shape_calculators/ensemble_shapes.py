# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.utils import (
    check_input_and_output_numbers, check_input_and_output_types)
from ..common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
    calculate_linear_classifier_output_shapes,
    _calculate_linear_classifier_output_shapes)
from ..common.data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)


def calculate_tree_regressor_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    This operator produces a scalar prediction for every example in a
    batch. If the input batch size is N, the output shape may be
    [N, 1].
    """
    check_input_and_output_numbers(operator, input_count_range=1,
                                   output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[
        BooleanTensorType, DoubleTensorType,
        FloatTensorType, Int64TensorType])

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type.shape = [N, 1]
    if len(operator.outputs) == 2:
        if hasattr(operator.raw_operator, 'estimators_'):
            operator.outputs[1].type.shape = [
                N, len(operator.raw_operator.estimators_)]
        else:
            operator.outputs[1].type.shape = [N, 1]


def calculate_tree_classifier_output_shapes(operator):
    _calculate_linear_classifier_output_shapes(operator, True)
    if len(operator.outputs) == 3:
        N = operator.inputs[0].type.shape[0]
        if hasattr(operator.raw_operator, 'estimators_'):
            operator.outputs[2].type.shape = [
                N, len(operator.raw_operator.estimators_)]
        else:
            operator.outputs[2].type.shape = [N, 1]


register_shape_calculator('SklearnDecisionTreeRegressor',
                          calculate_tree_regressor_output_shapes)
register_shape_calculator('SklearnExtraTreeRegressor',
                          calculate_tree_regressor_output_shapes)
register_shape_calculator('SklearnExtraTreesRegressor',
                          calculate_tree_regressor_output_shapes)
register_shape_calculator('SklearnGradientBoostingRegressor',
                          calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnHistGradientBoostingRegressor',
                          calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnRandomForestRegressor',
                          calculate_tree_regressor_output_shapes)

register_shape_calculator('SklearnDecisionTreeClassifier',
                          calculate_tree_classifier_output_shapes)
register_shape_calculator('SklearnExtraTreeClassifier',
                          calculate_tree_classifier_output_shapes)
register_shape_calculator('SklearnExtraTreesClassifier',
                          calculate_tree_classifier_output_shapes)
register_shape_calculator('SklearnGradientBoostingClassifier',
                          calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnHistGradientBoostingClassifier',
                          calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnRandomForestClassifier',
                          calculate_tree_classifier_output_shapes)
