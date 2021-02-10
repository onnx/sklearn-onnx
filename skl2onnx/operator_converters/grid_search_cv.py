# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from sklearn.base import is_classifier
from ..common._apply_operation import apply_identity
from ..common._registration import register_converter
from .._supported_operators import sklearn_operator_name_map


def convert_sklearn_grid_search_cv(scope, operator, container):
    """
    Converter for scikit-learn's GridSearchCV.
    """
    opts = scope.get_options(operator.raw_operator)
    grid_search_op = operator.raw_operator
    best_estimator = grid_search_op.best_estimator_
    op_type = sklearn_operator_name_map[type(best_estimator)]
    grid_search_operator = scope.declare_local_operator(
        op_type, best_estimator)
    container.add_options(id(best_estimator), opts)
    grid_search_operator.inputs = operator.inputs
    label_name = scope.declare_local_variable('label')
    grid_search_operator.outputs.append(label_name)
    if is_classifier(best_estimator):
        proba_name = scope.declare_local_variable(
            'probability_tensor', operator.inputs[0].type.__class__())
        grid_search_operator.outputs.append(proba_name)
    apply_identity(scope, label_name.full_name,
                   operator.outputs[0].full_name, container)
    if is_classifier(best_estimator):
        apply_identity(scope, proba_name.full_name,
                       operator.outputs[1].full_name, container)


register_converter('SklearnGridSearchCV',
                   convert_sklearn_grid_search_cv,
                   options="passthrough")
