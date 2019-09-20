# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from ..common._apply_operation import apply_cast
from ..common.data_types import Int64TensorType
from ..common._registration import register_converter
from ..common.tree_ensemble import add_tree_to_attribute_pairs
from ..common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from ..common.tree_ensemble import get_default_tree_regressor_attribute_pairs
from ..proto import onnx_proto


def convert_sklearn_gradient_boosting_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'
    if op.loss != 'deviance':
        raise NotImplementedError(
            "Loss '{0}' is not supported yet. You "
            "may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.".format(op.loss))

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)

    transform = 'LOGISTIC' if op.n_classes_ == 2 else 'SOFTMAX'
    if op.init == 'zero':
        base_values = np.zeros(op.loss_.K)
    elif op.init is None:
        x0 = np.zeros((1, op.estimators_[0, 0].n_features_))
        if hasattr(op, '_raw_predict_init'):
            # sklearn >= 0.21
            base_values = op._raw_predict_init(x0).ravel()
        elif hasattr(op, '_init_decision_function'):
            # sklearn >= 0.20 and sklearn < 0.21
            base_values = op._init_decision_function(x0).ravel()
        else:
            raise RuntimeError("scikit-learn < 0.19 is not supported.")
    else:
        raise NotImplementedError(
            'Setting init to an estimator is not supported, you may raise an '
            'issue at https://github.com/onnx/sklearn-onnx/issues.')

    attrs['base_values'] = [float(v) for v in base_values]
    attrs['post_transform'] = transform

    classes = op.classes_
    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        attrs['classlabels_int64s'] = class_labels
    elif all(isinstance(i, (six.string_types, six.text_type))
             for i in classes):
        class_labels = [str(i) for i in classes]
        attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Labels must be all integer or all strings.')

    tree_weight = op.learning_rate
    n_est = (op.n_estimators_ if hasattr(op, 'n_estimators_') else
             op.n_estimators)
    if op.n_classes_ == 2:
        for tree_id in range(n_est):
            tree = op.estimators_[tree_id][0].tree_
            add_tree_to_attribute_pairs(attrs, True, tree, tree_id,
                                        tree_weight, 0, False, True,
                                        dtype=container.dtype)
    else:
        for i in range(n_est):
            for c in range(op.n_classes_):
                tree_id = i * op.n_classes_ + c
                tree = op.estimators_[i][c].tree_
                add_tree_to_attribute_pairs(attrs, True, tree, tree_id,
                                            tree_weight, c, False, True,
                                            dtype=container.dtype)

    container.add_node(
            op_type, operator.input_full_names,
            [operator.outputs[0].full_name, operator.outputs[1].full_name],
            op_domain='ai.onnx.ml', **attrs)


def convert_sklearn_gradient_boosting_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'
    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1

    if op.init == 'zero':
        cst = np.zeros(op.loss_.K)
    elif op.init is None:
        # constant_ was introduced in scikit-learn 0.21.
        if hasattr(op.init_, 'constant_'):
            cst = [float(x) for x in op.init_.constant_]
        elif op.loss == 'ls':
            cst = [op.init_.mean]
        else:
            cst = [op.init_.quantile]
    else:
        raise NotImplementedError(
            'Setting init to an estimator is not supported, you may raise an '
            'issue at https://github.com/onnx/sklearn-onnx/issues.')

    attrs['base_values'] = [float(x) for x in cst]

    tree_weight = op.learning_rate
    n_est = (op.n_estimators_ if hasattr(op, 'n_estimators_') else
             op.n_estimators)
    for i in range(n_est):
        tree = op.estimators_[i][0].tree_
        tree_id = i
        add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight,
                                    0, False, True, dtype=container.dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = cast_input_name

    container.add_node(op_type, input_name,
                       operator.output_full_names, op_domain='ai.onnx.ml',
                       **attrs)


register_converter('SklearnGradientBoostingClassifier',
                   convert_sklearn_gradient_boosting_classifier)
register_converter('SklearnGradientBoostingRegressor',
                   convert_sklearn_gradient_boosting_regressor)
