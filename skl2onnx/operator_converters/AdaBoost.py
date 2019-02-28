# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from ..common._apply_operation import (
    apply_add, apply_cast, apply_concat, apply_div, apply_exp, apply_mul,
    apply_reshape, apply_sub)
from ..common._registration import register_converter
from ..common.tree_ensemble import add_tree_to_attribute_pairs
from ..common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from ..proto import onnx_proto


def _samme_proba(scope, container, proba_name, n_classes):
    clipped_proba_name = scope.get_unique_variable_name('clipped_proba')
    log_proba_name = scope.get_unique_variable_name('log_proba')
    reduced_proba_name = scope.get_unique_variable_name('reduced_proba')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')
    inverted_n_classes_name = scope.get_unique_variable_name(
                                                    'inverted_n_classes')
    n_classes_minus_one_name = scope.get_unique_variable_name(
                                                    'n_classes_minus_one')
    prod_result_name = scope.get_unique_variable_name('prod_result')
    sub_result_name = scope.get_unique_variable_name('sub_result')
    samme_proba_name = scope.get_unique_variable_name('samme_proba')

    container.add_initializer(
        inverted_n_classes_name, onnx_proto.TensorProto.FLOAT,
        [], [1. / n_classes])

    container.add_initializer(
        n_classes_minus_one_name, onnx_proto.TensorProto.FLOAT,
        [], [n_classes - 1])

    container.add_node(
        'Clip', proba_name, clipped_proba_name,
        name=scope.get_unique_operator_name('Clip'),
        min=np.finfo(float).eps)  # max=None)
    container.add_node(
        'Log', clipped_proba_name, log_proba_name,
        name=scope.get_unique_operator_name('Log'))
    container.add_node(
        'ReduceSum', log_proba_name, reduced_proba_name, axes=[1],
        name=scope.get_unique_operator_name('ReduceSum'))
    apply_reshape(scope, reduced_proba_name,
                  reshaped_result_name, container,
                  desired_shape=(-1, 1))
    apply_mul(scope, [reshaped_result_name, inverted_n_classes_name],
              prod_result_name, container, broadcast=1)
    apply_sub(scope, [log_proba_name, prod_result_name],
              sub_result_name, container, broadcast=1)
    apply_mul(scope, [sub_result_name, n_classes_minus_one_name],
              samme_proba_name, container, broadcast=1)
    return samme_proba_name


def convert_sklearn_ada_boost_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'
    classes = op.classes_

    est_weights_sum_name = scope.get_unique_variable_name('est_weights_sum')
    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    array_feature_extractor_result_name = scope.get_unique_variable_name(
                                            'array_feature_extractor_result')
    mul_operand_name = scope.get_unique_variable_name('mul_operand')
    concatenated_prob_name = scope.get_unique_variable_name(
                                                    'concatenated_prob')
    reduced_prob_name = scope.get_unique_variable_name('reduced_prob')
    div_result_name = scope.get_unique_variable_name('div_result')
    exp_operand_name = scope.get_unique_variable_name('exp_operand')
    exp_result_name = scope.get_unique_variable_name('exp_result')
    reduced_exp_result_name = scope.get_unique_variable_name(
                                                'reduced_exp_result')
    normaliser_name = scope.get_unique_variable_name('normaliser')
    zero_scaler_name = scope.get_unique_variable_name('zero_scaler')
    comparison_result_name = scope.get_unique_variable_name(
                                                'comparison_result')
    cast_output_name = scope.get_unique_variable_name('cast_output')
    zero_filtered_normaliser_name = scope.get_unique_variable_name(
                                                'zero_filtered_normaliser')
    classes_name = scope.get_unique_variable_name('classes')
    cast_normaliser_name = scope.get_unique_variable_name('cast_normaliser')

    container.add_initializer(
        est_weights_sum_name, onnx_proto.TensorProto.FLOAT,
        [], [op.estimator_weights_.sum()])
    container.add_initializer(
        mul_operand_name, onnx_proto.TensorProto.FLOAT,
        [], [1. / (op.n_classes_ - 1)])
    container.add_initializer(zero_scaler_name,
                              onnx_proto.TensorProto.INT32, [1], [0])

    attrs = get_default_tree_classifier_attribute_pairs()

    class_type = onnx_proto.TensorProto.STRING
    if all(isinstance(i, np.ndarray) for i in classes):
        classes = np.concatenate(classes)
    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        attrs['classlabels_int64s'] = class_labels
        class_type = onnx_proto.TensorProto.INT32
    elif all(isinstance(i, (six.string_types, six.text_type))
             for i in classes):
        class_labels = [str(i) for i in classes]
        attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Only support pure string or integer class labels')

    container.add_initializer(classes_name, class_type, classes.shape, classes)

    proba_names_list = []
    for tree_id in range(op.n_estimators):
        label_name = scope.get_unique_variable_name('label')
        proba_name = scope.get_unique_variable_name('proba')
        attrs['name'] = scope.get_unique_operator_name(op_type)

        add_tree_to_attribute_pairs(attrs, True, op.estimators_[tree_id].tree_,
                                    tree_id, op.learning_rate, 0, True)
        container.add_node(
            op_type, operator.input_full_names,
            [label_name, proba_name],
            op_domain='ai.onnx.ml', **attrs)
        if op.algorithm == 'SAMME.R':
            cur_proba_name = _samme_proba(scope, container, proba_name,
                                          op.n_classes_)
        # else:  # SAMME
        proba_names_list.append(cur_proba_name)

    apply_concat(scope, [p for p in proba_names_list], concatenated_prob_name,
                 container, axis=0)
    container.add_node('ReduceSum', concatenated_prob_name,
                       reduced_prob_name, axes=[0],
                       name=scope.get_unique_operator_name('ReduceSum'))
    apply_div(scope, [reduced_prob_name, est_weights_sum_name],
              div_result_name, container, broadcast=1)
    apply_mul(scope, [div_result_name, mul_operand_name],
              exp_operand_name, container, broadcast=1)
    apply_exp(scope, exp_operand_name, exp_result_name, container)
    container.add_node(
        'ReduceSum', exp_result_name, reduced_exp_result_name, axes=[1],
        name=scope.get_unique_operator_name('ReduceSum'))
    apply_reshape(scope, reduced_exp_result_name,
                  normaliser_name, container,
                  desired_shape=(-1, 1))
    apply_cast(scope, normaliser_name, cast_normaliser_name,
               container, to=onnx_proto.TensorProto.INT32)
    container.add_node('Equal', [cast_normaliser_name, zero_scaler_name],
                       comparison_result_name,
                       name=scope.get_unique_operator_name('Equal'))
    apply_cast(scope, comparison_result_name, cast_output_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    apply_add(scope, [normaliser_name, cast_output_name],
              zero_filtered_normaliser_name,
              container, broadcast=0)
    apply_div(scope, [exp_result_name, zero_filtered_normaliser_name],
              operator.outputs[1].full_name, container, broadcast=1)
    class_prob_name = operator.outputs[1].full_name
    container.add_node('ArgMax', class_prob_name,
                       argmax_output_name,
                       name=scope.get_unique_operator_name('ArgMax'), axis=1)
    container.add_node(
        'ArrayFeatureExtractor', [classes_name, argmax_output_name],
        array_feature_extractor_result_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))

    if class_type == onnx_proto.TensorProto.INT32:
        reshaped_result_name = scope.get_unique_variable_name(
                                                'reshaped_result')

        apply_reshape(scope, array_feature_extractor_result_name,
                      reshaped_result_name, container,
                      desired_shape=(-1,))
        apply_cast(scope, reshaped_result_name, operator.outputs[0].full_name,
                   container, to=onnx_proto.TensorProto.INT64)
    else:
        apply_reshape(scope, array_feature_extractor_result_name,
                      operator.outputs[0].full_name, container,
                      desired_shape=(-1,))


register_converter('SklearnAdaBoostClassifier',
                   convert_sklearn_ada_boost_classifier)
