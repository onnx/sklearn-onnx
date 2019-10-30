# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from onnx.helper import make_tensor
from ..common._apply_operation import (
    apply_add, apply_cast, apply_clip, apply_concat, apply_div, apply_exp,
    apply_mul, apply_reshape, apply_sub, apply_topk, apply_transpose
)
from ..common.data_types import FloatTensorType
from ..common._registration import register_converter
from ..proto import onnx_proto
from .._supported_operators import sklearn_operator_name_map


def _samme_r_proba(scope, container, proba_name, n_classes):
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

    apply_clip(
        scope, proba_name, clipped_proba_name, container,
        operator_name=scope.get_unique_operator_name('Clip'),
        min=np.finfo(float).eps)
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


def _normalise_probability(scope, container, operator, proba_names_list,
                           model):
    est_weights_sum_name = scope.get_unique_variable_name('est_weights_sum')
    summation_prob_name = scope.get_unique_variable_name('summation_prob')
    div_result_name = scope.get_unique_variable_name('div_result')
    exp_operand_name = scope.get_unique_variable_name('exp_operand')
    exp_result_name = scope.get_unique_variable_name('exp_result')
    reduced_exp_result_name = scope.get_unique_variable_name(
        'reduced_exp_result')
    normaliser_name = scope.get_unique_variable_name('normaliser')
    zero_scalar_name = scope.get_unique_variable_name('zero_scalar')
    comparison_result_name = scope.get_unique_variable_name(
        'comparison_result')
    cast_output_name = scope.get_unique_variable_name('cast_output')
    zero_filtered_normaliser_name = scope.get_unique_variable_name(
        'zero_filtered_normaliser')
    mul_operand_name = scope.get_unique_variable_name('mul_operand')
    cast_normaliser_name = scope.get_unique_variable_name('cast_normaliser')

    container.add_initializer(
        est_weights_sum_name, onnx_proto.TensorProto.FLOAT,
        [], [model.estimator_weights_.sum()])
    container.add_initializer(
        mul_operand_name, onnx_proto.TensorProto.FLOAT,
        [], [1. / (model.n_classes_ - 1)])
    container.add_initializer(zero_scalar_name,
                              onnx_proto.TensorProto.INT32, [], [0])

    container.add_node('Sum', proba_names_list,
                       summation_prob_name,
                       name=scope.get_unique_operator_name('Sum'))
    apply_div(scope, [summation_prob_name, est_weights_sum_name],
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
    container.add_node('Equal', [cast_normaliser_name, zero_scalar_name],
                       comparison_result_name,
                       name=scope.get_unique_operator_name('Equal'))
    apply_cast(scope, comparison_result_name, cast_output_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    apply_add(scope, [normaliser_name, cast_output_name],
              zero_filtered_normaliser_name,
              container, broadcast=0)
    apply_div(scope, [exp_result_name, zero_filtered_normaliser_name],
              operator.outputs[1].full_name, container, broadcast=1)
    return operator.outputs[1].full_name


def _samme_proba(scope, container, proba_name, n_classes, weight,
                 zero_name, classes_ind_name, one_name):
    weight_name = scope.get_unique_variable_name('weight')
    container.add_initializer(
        weight_name, onnx_proto.TensorProto.FLOAT, [], [weight])

    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    container.add_node('ArgMax', proba_name,
                       argmax_output_name,
                       name=scope.get_unique_operator_name('ArgMax'),
                       axis=1)

    equal_name = scope.get_unique_variable_name('equal')
    container.add_node('Equal', [argmax_output_name, classes_ind_name],
                       equal_name,
                       name=scope.get_unique_operator_name('Equal'))

    max_proba_name = scope.get_unique_variable_name('probsmax')
    container.add_node('Where', [equal_name, one_name, zero_name],
                       max_proba_name,
                       name=scope.get_unique_operator_name('Where'))

    samme_proba_name = scope.get_unique_variable_name('samme_proba')
    apply_mul(scope, [max_proba_name, weight_name],
              samme_proba_name, container, broadcast=1)
    return samme_proba_name


def convert_sklearn_ada_boost_classifier(scope, operator, container):
    """
    Converter for AdaBoost classifier.
    This function goes through the list of estimators and uses
    TreeEnsembleClassifer op to calculate class probabilities
    for each estimator. Then it calculates the weighted sum
    across all the estimators depending on the algorithm
    picked during trainging (SAMME.R or SAMME) and normalises
    the probability score for the final result. Label is
    calculated by simply doing an argmax of the probability scores.
    """
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'
    classes = op.classes_
    class_type = onnx_proto.TensorProto.STRING
    if np.issubdtype(classes.dtype, np.floating):
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype('int')
    elif np.issubdtype(classes.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode('utf-8') for s in classes])

    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        'array_feature_extractor_result')

    classes_name = scope.get_unique_variable_name('classes')
    container.add_initializer(classes_name, class_type, classes.shape, classes)

    proba_names_list = []
    zero_name = None
    classes_ind_name = None

    for i_est, estimator in enumerate(op.estimators_):
        label_name = scope.declare_local_variable('elab_name_%d' % i_est)
        proba_name = scope.declare_local_variable('eprob_name_%d' % i_est)

        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type)
        this_operator.raw_operator = estimator
        this_operator.inputs = operator.inputs
        this_operator.outputs.extend([label_name, proba_name])

        if op.algorithm == 'SAMME.R':
            cur_proba_name = _samme_r_proba(
                scope, container, proba_name.onnx_name, op.n_classes_)
        else:  # SAMME

            if classes_ind_name is None:
                classes_ind_name = scope.get_unique_variable_name(
                    'classes_ind3')
                container.add_initializer(
                    classes_ind_name, onnx_proto.TensorProto.INT64,
                    (1, len(classes)), list(range(len(classes))))

            if zero_name is None:
                shape_name = scope.get_unique_variable_name('shape')
                container.add_node(
                    'Shape', proba_name.onnx_name, shape_name,
                    name=scope.get_unique_operator_name('Shape'))

                zero_name = scope.get_unique_variable_name('zero')
                container.add_node(
                    'ConstantOfShape', shape_name, zero_name,
                    name=scope.get_unique_operator_name('CoSA'),
                    value=make_tensor("value", onnx_proto.TensorProto.FLOAT,
                                      (1, ), [0]))

                one_name = scope.get_unique_variable_name('one')
                container.add_node(
                    'ConstantOfShape', shape_name, one_name,
                    name=scope.get_unique_operator_name('CoSB'),
                    value=make_tensor("value", onnx_proto.TensorProto.FLOAT,
                                      (1, ), [1.]))

            cur_proba_name = _samme_proba(
                scope, container, proba_name.onnx_name, op.n_classes_,
                op.estimator_weights_[i_est], zero_name, classes_ind_name,
                one_name)

        proba_names_list.append(cur_proba_name)

    class_prob_name = _normalise_probability(scope, container, operator,
                                             proba_names_list, op)
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


def _get_estimators_label(scope, operator, container, model):
    """
    This function computes labels for each estimator and returns
    a tensor produced by concatenating the labels.
    """
    op_type = 'TreeEnsembleRegressor'
    concatenated_labels_name = scope.get_unique_variable_name(
        'concatenated_labels')

    input_name = operator.inputs
    estimators_results_list = []
    for i, estimator in enumerate(model.estimators_):
        estimator_label_name = scope.declare_local_variable(
            'est_label_%d' % i, FloatTensorType([None, 1]))

        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type)
        this_operator.raw_operator = estimator
        this_operator.inputs = input_name
        this_operator.outputs.append(estimator_label_name)

        estimators_results_list.append(estimator_label_name.onnx_name)

    apply_concat(scope, estimators_results_list, concatenated_labels_name,
                 container, axis=1)
    return concatenated_labels_name


def cum_sum(scope, container, rnn_input_name, sequence_length):
    opv = container.target_opset
    if opv < 11:
        transposed_input_name = scope.get_unique_variable_name(
            'transposed_input')
        reshaped_result_name = scope.get_unique_variable_name(
            'reshaped_result')
        weights_name = scope.get_unique_variable_name('weights')
        rec_weights_name = scope.get_unique_variable_name('rec_weights')
        rnn_output_name = scope.get_unique_variable_name('rnn_output')
        permuted_rnn_y_name = scope.get_unique_variable_name('permuted_rnn_y')
        weights_cdf_name = scope.get_unique_variable_name('weights_cdf')

        container.add_initializer(weights_name,
                                  onnx_proto.TensorProto.FLOAT, [1, 1, 1], [1])
        container.add_initializer(rec_weights_name,
                                  onnx_proto.TensorProto.FLOAT, [1, 1, 1], [1])

        apply_transpose(scope, rnn_input_name, transposed_input_name,
                        container, perm=(1, 0))
        apply_reshape(scope, transposed_input_name, reshaped_result_name,
                      container, desired_shape=(sequence_length, -1, 1))
        container.add_node(
            'RNN', inputs=[reshaped_result_name,
                           weights_name, rec_weights_name],
            outputs=[rnn_output_name], activations=['Affine'],
            name=scope.get_unique_operator_name('RNN'),
            activation_alpha=[1.0], activation_beta=[0.0], hidden_size=1)
        apply_transpose(scope, rnn_output_name, permuted_rnn_y_name, container,
                        perm=(2, 0, 1, 3))
        apply_reshape(
            scope, permuted_rnn_y_name, weights_cdf_name, container,
            desired_shape=(-1, sequence_length))
        return weights_cdf_name
    else:
        axis_name = scope.get_unique_variable_name('axis_name')
        container.add_initializer(axis_name, onnx_proto.TensorProto.INT32,
                                  [], [1])
        weights_cdf_name = scope.get_unique_variable_name('weights_cdf')
        container.add_node(
            'CumSum', [rnn_input_name, axis_name], [weights_cdf_name],
            name=scope.get_unique_operator_name('CumSum'),
            op_version=11)
        return weights_cdf_name


def _apply_gather_elements(scope, container, inputs, output, axis,
                           dim, zero_type, suffix):
    if container.target_opset >= 11:
        container.add_node(
            'GatherElements', inputs, output, op_version=11, axis=axis,
            name=scope.get_unique_operator_name('GatEls' + suffix))
    else:
        classes_ind_name = scope.get_unique_variable_name('classes_ind2')
        container.add_initializer(
            classes_ind_name, onnx_proto.TensorProto.INT64,
            (1, dim), list(range(dim)))

        shape_name = scope.get_unique_variable_name('shape')
        container.add_node(
            'Shape', inputs[0], shape_name,
            name=scope.get_unique_operator_name('Shape'))
        zero_name = scope.get_unique_variable_name('zero')
        zero_val = (0 if zero_type == onnx_proto.TensorProto.INT64
                    else 0.)
        container.add_node(
            'ConstantOfShape', shape_name, zero_name,
            name=scope.get_unique_operator_name('CoSA'),
            value=make_tensor("value", zero_type,
                              (1, ), [zero_val]), op_version=9)

        equal_name = scope.get_unique_variable_name('equal')
        container.add_node('Equal', [inputs[1], classes_ind_name],
                           equal_name,
                           name=scope.get_unique_operator_name('Equal'))

        selected = scope.get_unique_variable_name('selected')
        container.add_node('Where', [equal_name, inputs[0], zero_name],
                           selected,
                           name=scope.get_unique_operator_name('Where'))
        container.add_node('ReduceSum', selected, output, axes=[1],
                           name=scope.get_unique_operator_name('Where'))


def convert_sklearn_ada_boost_regressor(scope, operator, container):
    """
    Converter for AdaBoost regressor.
    This function first calls _get_estimators_label() which returns a
    tensor of concatenated labels predicted by each estimator. Then,
    median is calculated and returned as the final output.

    Note: This function creates an ONNX model which can predict on only
    one instance at a time because ArrayFeatureExtractor can only
    extract based on the last axis, so we can't fetch different columns
    for different rows.
    """
    op = operator.raw_operator

    negate_name = scope.get_unique_variable_name('negate')
    estimators_weights_name = scope.get_unique_variable_name(
        'estimators_weights')
    half_scalar_name = scope.get_unique_variable_name('half_scalar')
    last_index_name = scope.get_unique_variable_name('last_index')
    negated_labels_name = scope.get_unique_variable_name('negated_labels')
    sorted_values_name = scope.get_unique_variable_name('sorted_values')
    sorted_indices_name = scope.get_unique_variable_name('sorted_indices')
    array_feat_extractor_output_name = scope.get_unique_variable_name(
        'array_feat_extractor_output')
    median_value_name = scope.get_unique_variable_name('median_value')
    comp_value_name = scope.get_unique_variable_name('comp_value')
    median_or_above_name = scope.get_unique_variable_name('median_or_above')
    median_idx_name = scope.get_unique_variable_name('median_idx')
    cast_result_name = scope.get_unique_variable_name('cast_result')
    reshaped_weights_name = scope.get_unique_variable_name('reshaped_weights')
    median_estimators_name = scope.get_unique_variable_name(
        'median_estimators')

    container.add_initializer(negate_name, onnx_proto.TensorProto.FLOAT,
                              [], [-1])
    container.add_initializer(estimators_weights_name,
                              onnx_proto.TensorProto.FLOAT,
                              [len(op.estimator_weights_)],
                              op.estimator_weights_)
    container.add_initializer(half_scalar_name, onnx_proto.TensorProto.FLOAT,
                              [], [0.5])
    container.add_initializer(last_index_name, onnx_proto.TensorProto.INT64,
                              [], [len(op.estimators_) - 1])

    concatenated_labels = _get_estimators_label(scope, operator,
                                                container, op)
    apply_mul(scope, [concatenated_labels, negate_name],
              negated_labels_name, container, broadcast=1)
    apply_topk(scope, negated_labels_name,
               [sorted_values_name, sorted_indices_name],
               container, k=len(op.estimators_))
    container.add_node(
        'ArrayFeatureExtractor',
        [estimators_weights_name, sorted_indices_name],
        array_feat_extractor_output_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_reshape(
        scope, array_feat_extractor_output_name, reshaped_weights_name,
        container, desired_shape=(-1, len(op.estimators_)))
    weights_cdf_name = cum_sum(
        scope, container, reshaped_weights_name,
        len(op.estimators_))
    container.add_node(
        'ArrayFeatureExtractor', [weights_cdf_name, last_index_name],
        median_value_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_mul(scope, [median_value_name, half_scalar_name],
              comp_value_name, container, broadcast=1)
    container.add_node(
        'Less', [weights_cdf_name, comp_value_name],
        median_or_above_name,
        name=scope.get_unique_operator_name('Less'))
    apply_cast(scope, median_or_above_name, cast_result_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    container.add_node('ArgMin', cast_result_name,
                       median_idx_name,
                       name=scope.get_unique_operator_name('ArgMin'), axis=1)
    _apply_gather_elements(
        scope, container, [sorted_indices_name, median_idx_name],
        median_estimators_name, axis=1, dim=len(op.estimators_),
        zero_type=onnx_proto.TensorProto.INT64, suffix="A")
    output_name = operator.output_full_names[0]
    _apply_gather_elements(
        scope, container, [concatenated_labels, median_estimators_name],
        output_name, axis=1, dim=len(op.estimators_),
        zero_type=onnx_proto.TensorProto.FLOAT, suffix="B")


register_converter('SklearnAdaBoostClassifier',
                   convert_sklearn_ada_boost_classifier)
register_converter('SklearnAdaBoostRegressor',
                   convert_sklearn_ada_boost_regressor)
