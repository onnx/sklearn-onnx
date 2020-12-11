# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from ..common._apply_operation import (
    apply_cast,
    apply_concat,
    apply_reshape,
    apply_transpose,
)
from ..common.data_types import (
    BooleanTensorType, Int64TensorType, guess_numpy_type)
from ..common._registration import register_converter
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    add_tree_to_attribute_pairs_hist_gradient_boosting,
    get_default_tree_classifier_attribute_pairs,
    get_default_tree_regressor_attribute_pairs
)
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto
from .decision_tree import predict, _build_labels


def _num_estimators(op):
    # don't use op.n_estimators since it may not be the same as
    # len(op.estimators_). At training time n_estimators can be changed by
    # training code:
    #   for j in range(10):
    #       ...
    #       classifier.fit(X_tmp, y_tmp)
    #       classifier.n_estimators += 30
    if hasattr(op, 'estimators_'):
        return len(op.estimators_)
    elif hasattr(op, '_predictors'):
        # HistGradientBoosting*
        return len(op._predictors)
    raise NotImplementedError(
        "Model should have attribute 'estimators_' or '_predictors'.")


def _calculate_labels(scope, container, model, proba):
    predictions = []
    transposed_result_name = scope.get_unique_variable_name(
        'transposed_result')
    apply_transpose(scope, proba, transposed_result_name,
                    container, perm=(1, 2, 0))
    for k in range(model.n_outputs_):
        preds_name = scope.get_unique_variable_name('preds')
        reshaped_preds_name = scope.get_unique_variable_name(
            'reshaped_preds')
        k_name = scope.get_unique_variable_name('k_column')
        out_k_name = scope.get_unique_variable_name('out_k_column')
        argmax_output_name = scope.get_unique_variable_name(
            'argmax_output')
        classes_name = scope.get_unique_variable_name('classes')
        reshaped_result_name = scope.get_unique_variable_name(
            'reshaped_result')

        container.add_initializer(
            k_name, onnx_proto.TensorProto.INT64,
            [], [k])
        container.add_initializer(
            classes_name, onnx_proto.TensorProto.INT64,
            model.classes_[k].shape, model.classes_[k])

        container.add_node(
            'ArrayFeatureExtractor', [transposed_result_name, k_name],
            out_k_name, op_domain='ai.onnx.ml',
            name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
        container.add_node(
            'ArgMax', out_k_name, argmax_output_name,
            name=scope.get_unique_operator_name('ArgMax'), axis=1)
        apply_reshape(scope, argmax_output_name, reshaped_result_name,
                      container, desired_shape=(1, -1))
        container.add_node(
            'ArrayFeatureExtractor', [classes_name, reshaped_result_name],
            preds_name, op_domain='ai.onnx.ml',
            name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
        apply_reshape(scope, preds_name, reshaped_preds_name,
                      container, desired_shape=(-1, 1))
        predictions.append(reshaped_preds_name)
    return predictions


def convert_sklearn_random_forest_classifier(
        scope, operator, container, op_type='TreeEnsembleClassifier',
        op_domain='ai.onnx.ml', op_version=1):
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    op = operator.raw_operator

    if hasattr(op, 'n_outputs_'):
        n_outputs = int(op.n_outputs_)
        options = container.get_options(
            op, dict(raw_scores=False, decision_path=False))
    elif hasattr(op, 'n_trees_per_iteration_'):
        # HistGradientBoostingClassifier
        n_outputs = op.n_trees_per_iteration_
        options = container.get_options(op, dict(raw_scores=False))
    else:
        raise NotImplementedError(
            "Model should have attribute 'n_outputs_' or "
            "'n_trees_per_iteration_'.")

    use_raw_scores = options['raw_scores']

    if n_outputs == 1 or hasattr(op, 'loss_') or hasattr(op, '_loss'):
        classes = get_label_classes(scope, op)

        if all(isinstance(i, np.ndarray) for i in classes):
            classes = np.concatenate(classes)
        attr_pairs = get_default_tree_classifier_attribute_pairs()
        attr_pairs['name'] = scope.get_unique_operator_name(op_type)

        if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
            class_labels = [int(i) for i in classes]
            attr_pairs['classlabels_int64s'] = class_labels
        elif all(isinstance(i, (six.text_type, six.string_types))
                 for i in classes):
            class_labels = [str(i) for i in classes]
            attr_pairs['classlabels_strings'] = class_labels
        else:
            raise ValueError(
                'Only string and integer class labels are allowed.')

        # random forest calculate the final score by averaging over all trees'
        # outcomes, so all trees' weights are identical.
        if hasattr(op, 'estimators_'):
            estimator_count = len(op.estimators_)
            tree_weight = 1. / estimator_count
        elif hasattr(op, '_predictors'):
            # HistGradientBoostingRegressor
            estimator_count = len(op._predictors)
            tree_weight = 1.
        else:
            raise NotImplementedError(
                "Model should have attribute 'estimators_' or '_predictors'.")

        for tree_id in range(estimator_count):

            if hasattr(op, 'estimators_'):
                tree = op.estimators_[tree_id].tree_
                add_tree_to_attribute_pairs(
                    attr_pairs, True, tree, tree_id,
                    tree_weight, 0, True, True,
                    dtype=dtype)
            else:
                # HistGradientBoostClassifier
                if len(op._predictors[tree_id]) == 1:
                    tree = op._predictors[tree_id][0]
                    add_tree_to_attribute_pairs_hist_gradient_boosting(
                        attr_pairs, True, tree, tree_id, tree_weight, 0,
                        False, False, dtype=dtype)
                else:
                    for cl, tree in enumerate(op._predictors[tree_id]):
                        add_tree_to_attribute_pairs_hist_gradient_boosting(
                            attr_pairs, True, tree, tree_id * n_outputs + cl,
                            tree_weight, cl, False, False,
                            dtype=dtype)

        if hasattr(op, '_baseline_prediction'):
            if isinstance(op._baseline_prediction, np.ndarray):
                attr_pairs['base_values'] = list(
                    op._baseline_prediction.ravel())
            else:
                attr_pairs['base_values'] = [op._baseline_prediction]

        if hasattr(op, 'loss_'):
            loss = op.loss_
        elif hasattr(op, '_loss'):
            # scikit-learn >= 0.24
            loss = op._loss
        else:
            loss = None
        if loss is not None:
            if use_raw_scores:
                attr_pairs['post_transform'] = "NONE"
            elif loss.__class__.__name__ == "BinaryCrossEntropy":
                attr_pairs['post_transform'] = "LOGISTIC"
            elif loss.__class__.__name__ == "CategoricalCrossEntropy":
                attr_pairs['post_transform'] = "SOFTMAX"
            else:
                raise NotImplementedError(
                    "There is no corresponding post_transform for "
                    "'{}'.".format(loss.__class__.__name__))
        elif use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}'.".format(type(op)))

        input_name = operator.input_full_names
        if type(operator.inputs[0].type) == BooleanTensorType:
            cast_input_name = scope.get_unique_variable_name('cast_input')

            apply_cast(scope, input_name, cast_input_name,
                       container, to=onnx_proto.TensorProto.FLOAT)
            input_name = cast_input_name

        if dtype is not None:
            for k in attr_pairs:
                if k in ('nodes_values', 'class_weights',
                         'target_weights', 'nodes_hitrates',
                         'base_values'):
                    attr_pairs[k] = np.array(attr_pairs[k], dtype=dtype)

        container.add_node(
            op_type, input_name,
            [operator.outputs[0].full_name, operator.outputs[1].full_name],
            op_domain=op_domain, op_version=op_version, **attr_pairs)

        if not options.get('decision_path', False):
            return

        # decision_path
        tree_paths = []
        for i, tree in enumerate(op.estimators_):

            attrs = get_default_tree_classifier_attribute_pairs()
            attrs['name'] = scope.get_unique_operator_name(
                "%s_%d" % (op_type, i))
            attrs['n_targets'] = int(op.n_outputs_)
            add_tree_to_attribute_pairs(
                attrs, True, tree.tree_, 0, 1., 0, False,
                True, dtype=dtype)

            attrs['n_targets'] = 1
            attrs['post_transform'] = 'NONE'
            attrs['target_ids'] = [0 for _ in attrs['class_ids']]
            attrs['target_weights'] = [
                float(_) for _ in attrs['class_nodeids']]
            attrs['target_nodeids'] = attrs['class_nodeids']
            attrs['target_treeids'] = attrs['class_treeids']
            rem = [k for k in attrs if k.startswith('class')]
            for k in rem:
                del attrs[k]

            if dtype is not None:
                for k in attrs:
                    if k in ('nodes_values', 'class_weights',
                             'target_weights', 'nodes_hitrates',
                             'base_values'):
                        attrs[k] = np.array(attrs[k], dtype=dtype)

            dpath = scope.get_unique_variable_name("dpath%d" % i)
            container.add_node(
                op_type.replace("Classifier", "Regressor"), input_name, dpath,
                op_domain=op_domain, op_version=op_version, **attrs)

            labels = _build_labels(tree.tree_)
            ordered = list(sorted(labels.items()))
            keys = [float(_[0]) for _ in ordered]
            values = [_[1] for _ in ordered]
            name = scope.get_unique_variable_name('spath%d' % i)
            container.add_node(
                'LabelEncoder', dpath, name,
                op_domain=op_domain, op_version=2,
                default_string='0', keys_floats=keys, values_strings=values,
                name=scope.get_unique_operator_name('TreePath'))
            name_shape = scope.get_unique_variable_name('spath%d' % i)
            apply_reshape(
                scope, name, name_shape,
                container, desired_shape=(-1, 1),
                operator_name=scope.get_unique_operator_name('sdpath2d%d' % i))
            tree_paths.append(name_shape)

        # merges everything
        apply_concat(
            scope, tree_paths, operator.outputs[2].full_name, container,
            axis=1, operator_name=scope.get_unique_operator_name('concat'))

    else:
        if use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}'.".format(type(op)))
        concatenated_proba_name = scope.get_unique_variable_name(
            'concatenated_proba')
        proba = []
        for est in op.estimators_:
            reshaped_est_proba_name = scope.get_unique_variable_name(
                'reshaped_est_proba')
            est_proba = predict(
                est, scope, operator, container, op_type, op_domain,
                op_version, is_ensemble=True)
            apply_reshape(
                scope, est_proba, reshaped_est_proba_name, container,
                desired_shape=(
                    1, n_outputs, -1, max([len(x) for x in op.classes_])))
            proba.append(reshaped_est_proba_name)
        apply_concat(scope, proba, concatenated_proba_name,
                     container, axis=0)
        container.add_node('ReduceMean', concatenated_proba_name,
                           operator.outputs[1].full_name,
                           name=scope.get_unique_operator_name('ReduceMean'),
                           axes=[0], keepdims=0)
        predictions = _calculate_labels(
            scope, container, op, operator.outputs[1].full_name)
        apply_concat(scope, predictions, operator.outputs[0].full_name,
                     container, axis=1)

        if options.get('decision_path', False):
            raise RuntimeError(
                "Decision output for multi-outputs is not implemented yet.")


def convert_sklearn_random_forest_regressor_converter(
        scope, operator, container, op_type='TreeEnsembleRegressor',
        op_domain='ai.onnx.ml', op_version=1):
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    op = operator.raw_operator
    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)

    if hasattr(op, 'n_outputs_'):
        attrs['n_targets'] = int(op.n_outputs_)
    elif hasattr(op, 'n_trees_per_iteration_'):
        # HistGradientBoostingRegressor
        attrs['n_targets'] = op.n_trees_per_iteration_
    else:
        raise NotImplementedError(
            "Model should have attribute 'n_outputs_' or "
            "'n_trees_per_iteration_'.")

    if hasattr(op, 'estimators_'):
        estimator_count = len(op.estimators_)
        tree_weight = 1. / estimator_count
    elif hasattr(op, '_predictors'):
        # HistGradientBoostingRegressor
        estimator_count = len(op._predictors)
        tree_weight = 1.
    else:
        raise NotImplementedError(
            "Model should have attribute 'estimators_' or '_predictors'.")

    # random forest calculate the final score by averaging over all trees'
    # outcomes, so all trees' weights are identical.
    for tree_id in range(estimator_count):
        if hasattr(op, 'estimators_'):
            tree = op.estimators_[tree_id].tree_
            add_tree_to_attribute_pairs(attrs, False, tree, tree_id,
                                        tree_weight, 0, False, True,
                                        dtype=dtype)
        else:
            # HistGradientBoostingRegressor
            if len(op._predictors[tree_id]) != 1:
                raise NotImplementedError(
                    "The converter does not work when the number of trees "
                    "is not 1 but {}.".format(len(op._predictors[tree_id])))
            tree = op._predictors[tree_id][0]
            add_tree_to_attribute_pairs_hist_gradient_boosting(
                attrs, False, tree, tree_id, tree_weight, 0, False,
                False, dtype=dtype)

    if hasattr(op, '_baseline_prediction'):
        if isinstance(op._baseline_prediction, np.ndarray):
            attrs['base_values'] = list(op._baseline_prediction)
        else:
            attrs['base_values'] = [op._baseline_prediction]

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = cast_input_name

    if dtype is not None:
        for k in attrs:
            if k in ('nodes_values', 'class_weights',
                     'target_weights', 'nodes_hitrates',
                     'base_values'):
                attrs[k] = np.array(attrs[k], dtype=dtype)

    container.add_node(
        op_type, input_name,
        operator.outputs[0].full_name, op_domain=op_domain,
        op_version=op_version, **attrs)

    if hasattr(op, 'n_trees_per_iteration_'):
        # HistGradientBoostingRegressor does not implement decision_path.
        return
    options = scope.get_options(op, dict(decision_path=False))
    if not options['decision_path']:
        return

    # decision_path
    tree_paths = []
    for i, tree in enumerate(op.estimators_):

        attrs = get_default_tree_regressor_attribute_pairs()
        attrs['name'] = scope.get_unique_operator_name("%s_%d" % (op_type, i))
        attrs['n_targets'] = int(op.n_outputs_)
        add_tree_to_attribute_pairs(attrs, False, tree.tree_, 0, 1., 0, False,
                                    True, dtype=dtype)

        attrs['n_targets'] = 1
        attrs['post_transform'] = 'NONE'
        attrs['target_ids'] = [0 for _ in attrs['target_ids']]
        attrs['target_weights'] = [float(_) for _ in attrs['target_nodeids']]

        if dtype is not None:
            for k in attrs:
                if k in ('nodes_values', 'class_weights',
                         'target_weights', 'nodes_hitrates',
                         'base_values'):
                    attrs[k] = np.array(attrs[k], dtype=dtype)

        dpath = scope.get_unique_variable_name("dpath%d" % i)
        container.add_node(
            op_type, input_name, dpath,
            op_domain=op_domain, op_version=op_version, **attrs)

        labels = _build_labels(tree.tree_)
        ordered = list(sorted(labels.items()))
        keys = [float(_[0]) for _ in ordered]
        values = [_[1] for _ in ordered]
        name = scope.get_unique_variable_name('spath%d' % i)
        container.add_node(
            'LabelEncoder', dpath, name,
            op_domain=op_domain, op_version=2,
            default_string='0', keys_floats=keys, values_strings=values,
            name=scope.get_unique_operator_name('TreePath'))
        name_shape = scope.get_unique_variable_name('spath%d' % i)
        apply_reshape(
            scope, name, name_shape,
            container, desired_shape=(-1, 1),
            operator_name=scope.get_unique_operator_name('sdpath2d%d' % i))
        tree_paths.append(name_shape)

    # merges everything
    apply_concat(
        scope, tree_paths, operator.outputs[1].full_name, container, axis=1,
        operator_name=scope.get_unique_operator_name('concat'))


register_converter('SklearnRandomForestClassifier',
                   convert_sklearn_random_forest_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'raw_scores': [True, False],
                            'nocl': [True, False],
                            'decision_path': [True, False]})
register_converter('SklearnRandomForestRegressor',
                   convert_sklearn_random_forest_regressor_converter,
                   options={'decision_path': [True, False]})
register_converter('SklearnExtraTreesClassifier',
                   convert_sklearn_random_forest_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'raw_scores': [True, False],
                            'nocl': [True, False],
                            'decision_path': [True, False]})
register_converter('SklearnExtraTreesRegressor',
                   convert_sklearn_random_forest_regressor_converter,
                   options={'decision_path': [True, False]})
register_converter('SklearnHistGradientBoostingClassifier',
                   convert_sklearn_random_forest_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'raw_scores': [True, False],
                            'nocl': [True, False]})
register_converter('SklearnHistGradientBoostingRegressor',
                   convert_sklearn_random_forest_regressor_converter,
                   options={'zipmap': [True, False, 'columns'],
                            'raw_scores': [True, False],
                            'nocl': [True, False]})
