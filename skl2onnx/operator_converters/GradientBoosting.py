# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from ..common._registration import register_converter
from ..common.tree_ensemble import add_tree_to_attribute_pairs
from ..common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from ..common.tree_ensemble import get_default_tree_regressor_attribute_pairs


def convert_sklearn_gradient_boosting_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'
    if op.loss != 'deviance':
        raise RuntimeError("loss '{}' not supported.".format(op.loss))

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)

    if op.n_classes_ == 2:
        transform = 'LOGISTIC'
        # class_prior_ was introduced in scikit-learn 0.21.
        if hasattr(op.init_, 'class_prior_'):
            base_values = op.init_.class_prior_
            assert base_values.shape == (2, )
        else:
            base_values = [op.init_.prior]
        if op.loss == 'deviance':
            # See https://github.com/scikit-learn/scikit-learn/blob/
            # master/sklearn/ensemble/_gb_losses.py#L666.
            eps = np.finfo(np.float32).eps
            base_values = np.clip(base_values, eps, 1 - eps)
            base_values = np.log(base_values / (1 - base_values))
        else:
            raise NotImplementedError("Other losses are not yet converted.")
    else:
        transform = 'SOFTMAX'
        # class_prior_ was introduced in scikit-learn 0.21.
        if hasattr(op.init_, 'class_prior_'):
            base_values = op.init_.class_prior_
        else:
            base_values = op.init_.priors

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
        raise ValueError('Only string or integer label vector is allowed')

    tree_weight = op.learning_rate
    if op.n_classes_ == 2:
        for tree_id in range(op.n_estimators):
            tree = op.estimators_[tree_id][0].tree_
            add_tree_to_attribute_pairs(attrs, True, tree, tree_id,
                                        tree_weight, 0, False)
    else:
        for i in range(op.n_estimators):
            for c in range(op.n_classes_):
                tree_id = i * op.n_classes_ + c
                tree = op.estimators_[i][c].tree_
                add_tree_to_attribute_pairs(attrs, True, tree, tree_id,
                                            tree_weight, c, False)

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

    # constant_ was introduced in scikit-learn 0.21.
    if hasattr(op.init_, 'constant_'):
        cst = [float(x) for x in op.init_.constant_]
    elif op.loss == 'ls':
        cst = [op.init_.mean]
    else:
        cst = [op.init_.quantile]
    attrs['base_values'] = [float(x) for x in cst]

    tree_weight = op.learning_rate
    for i in range(op.n_estimators):
        tree = op.estimators_[i][0].tree_
        tree_id = i
        add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight,
                                    0, False)

    container.add_node(op_type, operator.input_full_names,
                       operator.output_full_names, op_domain='ai.onnx.ml',
                       **attrs)


register_converter('SklearnGradientBoostingClassifier',
                   convert_sklearn_gradient_boosting_classifier)
register_converter('SklearnGradientBoostingRegressor',
                   convert_sklearn_gradient_boosting_regressor)
