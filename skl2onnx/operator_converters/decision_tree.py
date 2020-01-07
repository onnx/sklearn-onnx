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
    apply_mul,
    apply_reshape,
    apply_transpose,
)
from ..common._registration import register_converter
from ..common.data_types import Int64TensorType
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_classifier_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
)
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto


def populate_tree_attributes(model, name):
    """Construct attrs dictionary to be used in predict()
    while adding a node with TreeEnsembleClassifier ONNX op.
    """
    attrs = {}
    attrs['name'] = name
    attrs['post_transform'] = 'NONE'
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_missing_value_tracks_true'] = []
    attrs['nodes_hitrates'] = []
    attrs['class_treeids'] = []
    attrs['class_nodeids'] = []
    attrs['class_ids'] = []
    attrs['class_weights'] = []
    attrs['classlabels_int64s'] = list(range(model.tree_.node_count))

    for i in range(model.tree_.node_count):
        node_id = i
        if (model.tree_.children_left[i] > i and
                model.tree_.children_right[i] > i):
            feat = model.tree_.feature[i]
            thresh = model.tree_.threshold[i]
            left = model.tree_.children_left[i]
            right = model.tree_.children_right[i]
            mode = 'BRANCH_LEQ'
        else:
            feat, thresh, left, right = 0, 0., 0, 0
            mode = 'LEAF'
        attrs['nodes_nodeids'].append(node_id)
        attrs['nodes_treeids'].append(0)
        attrs['nodes_featureids'].append(feat)
        attrs['nodes_modes'].append(mode)
        attrs['nodes_truenodeids'].append(left)
        attrs['nodes_falsenodeids'].append(right)
        attrs['nodes_missing_value_tracks_true'].append(False)
        attrs['nodes_hitrates'].append(1.)
        attrs['nodes_values'].append(thresh)
        if mode == 'LEAF':
            attrs['class_ids'].append(node_id)
            attrs['class_weights'].append(1.)
            attrs['class_treeids'].append(0)
            attrs['class_nodeids'].append(node_id)
    return attrs


def predict(model, scope, operator, container, op_type, is_ensemble=False):
    """Predict target and calculate probability scores."""
    indices_name = scope.get_unique_variable_name('indices')
    dummy_proba_name = scope.get_unique_variable_name('dummy_proba')
    values_name = scope.get_unique_variable_name('values')
    out_values_name = scope.get_unique_variable_name('out_indices')
    transposed_result_name = scope.get_unique_variable_name(
        'transposed_result')
    proba_output_name = scope.get_unique_variable_name('proba_output')
    cast_result_name = scope.get_unique_variable_name('cast_result')
    reshaped_indices_name = scope.get_unique_variable_name('reshaped_indices')
    value = model.tree_.value.transpose(1, 2, 0)
    container.add_initializer(
        values_name, onnx_proto.TensorProto.FLOAT,
        value.shape, value.ravel())

    if model.tree_.node_count > 1:
        attrs = populate_tree_attributes(
            model, scope.get_unique_operator_name(op_type))
        container.add_node(
            op_type, operator.input_full_names,
            [indices_name, dummy_proba_name],
            op_domain='ai.onnx.ml', **attrs)
    else:
        zero_name = scope.get_unique_variable_name('zero')
        zero_matrix_name = scope.get_unique_variable_name('zero_matrix')
        reduced_zero_matrix_name = scope.get_unique_variable_name(
            'reduced_zero_matrix')

        container.add_initializer(
            zero_name, container.proto_dtype, [], [0])
        apply_mul(scope, [operator.inputs[0].full_name, zero_name],
                  zero_matrix_name, container, broadcast=1)
        container.add_node(
            'ReduceSum', zero_matrix_name, reduced_zero_matrix_name, axes=[1],
            name=scope.get_unique_operator_name('ReduceSum'))
        apply_cast(scope, reduced_zero_matrix_name, indices_name,
                   container, to=onnx_proto.TensorProto.INT64)
    apply_reshape(scope, indices_name, reshaped_indices_name,
                  container, desired_shape=[1, -1])
    container.add_node(
        'ArrayFeatureExtractor',
        [values_name, reshaped_indices_name],
        out_values_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_transpose(scope, out_values_name, proba_output_name,
                    container, perm=(0, 2, 1))
    apply_cast(scope, proba_output_name, cast_result_name,
               container, to=onnx_proto.TensorProto.BOOL)
    if is_ensemble:
        proba_result_name = scope.get_unique_variable_name('proba_result')

        apply_cast(scope, cast_result_name, proba_result_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        return proba_result_name
    apply_cast(scope, cast_result_name, operator.outputs[1].full_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    apply_transpose(scope, out_values_name, transposed_result_name,
                    container, perm=(2, 1, 0))
    return transposed_result_name


def convert_sklearn_decision_tree_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'
    if op.n_outputs_ == 1:
        attrs = get_default_tree_classifier_attribute_pairs()
        attrs['name'] = scope.get_unique_operator_name(op_type)
        classes = get_label_classes(scope, op)

        if all(isinstance(i, np.ndarray) for i in classes):
            classes = np.concatenate(classes)
        if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
            class_labels = [int(i) for i in classes]
            attrs['classlabels_int64s'] = class_labels
        elif all(isinstance(i, (six.string_types, six.text_type))
                 for i in classes):
            class_labels = [str(i) for i in classes]
            attrs['classlabels_strings'] = class_labels
        else:
            raise ValueError('Labels must be all integers or all strings.')

        add_tree_to_attribute_pairs(attrs, True, op.tree_, 0, 1., 0, True,
                                    True, dtype=container.dtype)
        container.add_node(
            op_type, operator.input_full_names,
            [operator.outputs[0].full_name, operator.outputs[1].full_name],
            op_domain='ai.onnx.ml', **attrs)
    else:
        transposed_result_name = predict(
            op, scope, operator, container, op_type)
        predictions = []
        for k in range(op.n_outputs_):
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
                op.classes_[k].shape, op.classes_[k])

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
        apply_concat(scope, predictions, operator.outputs[0].full_name,
                     container, axis=1)


def convert_sklearn_decision_tree_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = int(op.n_outputs_)
    add_tree_to_attribute_pairs(attrs, False, op.tree_, 0, 1., 0, False,
                                True, dtype=container.dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = [cast_input_name]

    container.add_node(op_type, input_name,
                       operator.output_full_names, op_domain='ai.onnx.ml',
                       **attrs)


register_converter('SklearnDecisionTreeClassifier',
                   convert_sklearn_decision_tree_classifier)
register_converter('SklearnDecisionTreeRegressor',
                   convert_sklearn_decision_tree_regressor)
register_converter('SklearnExtraTreeClassifier',
                   convert_sklearn_decision_tree_classifier)
register_converter('SklearnExtraTreeRegressor',
                   convert_sklearn_decision_tree_regressor)
