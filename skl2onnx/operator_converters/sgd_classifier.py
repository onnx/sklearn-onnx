# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._apply_operation import (
    apply_add, apply_cast, apply_clip, apply_concat, apply_div, apply_exp,
    apply_identity, apply_mul, apply_reciprocal, apply_reshape, apply_sub)
from ..common.data_types import (
    BooleanTensorType, Int64TensorType, guess_numpy_type)
from ..common._registration import register_converter
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto


def _decision_function(scope, operator, container, model):
    """Predict for linear model.
    score = X * coefficient + intercept
    """
    coef_name = scope.get_unique_variable_name('coef')
    intercept_name = scope.get_unique_variable_name('intercept')
    matmul_result_name = scope.get_unique_variable_name(
        'matmul_result')
    score_name = scope.get_unique_variable_name('score')
    coef = model.coef_.T

    container.add_initializer(coef_name, onnx_proto.TensorProto.FLOAT,
                              coef.shape, coef.ravel())
    container.add_initializer(intercept_name, onnx_proto.TensorProto.FLOAT,
                              model.intercept_.shape, model.intercept_)

    input_name = operator.inputs[0].full_name
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = cast_input_name

    container.add_node(
        'MatMul', [input_name, coef_name],
        matmul_result_name,
        name=scope.get_unique_operator_name('MatMul'))
    apply_add(scope, [matmul_result_name, intercept_name],
              score_name, container, broadcast=0)
    return score_name


def _handle_zeros(scope, container, proba, reduced_proba, num_classes):
    """Handle cases where reduced_proba values are zeros to avoid NaNs in
    class probability scores because of divide by 0 when we calculate
    proba / reduced_proba in _normalise_proba().
    This is done by replacing reduced_proba values of 0s with
    num_classes and corresponding proba values with 1.
    """
    num_classes_name = scope.get_unique_variable_name('num_classes')
    bool_reduced_proba_name = scope.get_unique_variable_name(
        'bool_reduced_proba')
    bool_not_reduced_proba_name = scope.get_unique_variable_name(
        'bool_not_reduced_proba')
    not_reduced_proba_name = scope.get_unique_variable_name(
        'not_reduced_proba')
    proba_updated_name = scope.get_unique_variable_name('proba_updated')
    mask_name = scope.get_unique_variable_name('mask')
    reduced_proba_updated_name = scope.get_unique_variable_name(
        'reduced_proba_updated')

    container.add_initializer(num_classes_name, onnx_proto.TensorProto.FLOAT,
                              [], [num_classes])

    apply_cast(scope, reduced_proba, bool_reduced_proba_name, container,
               to=onnx_proto.TensorProto.BOOL)
    container.add_node('Not', bool_reduced_proba_name,
                       bool_not_reduced_proba_name,
                       name=scope.get_unique_operator_name('Not'))
    apply_cast(scope, bool_not_reduced_proba_name, not_reduced_proba_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    apply_add(scope, [proba, not_reduced_proba_name],
              proba_updated_name, container, broadcast=1)
    apply_mul(scope, [not_reduced_proba_name, num_classes_name],
              mask_name, container, broadcast=1)
    apply_add(scope, [reduced_proba, mask_name],
              reduced_proba_updated_name, container, broadcast=0)
    return proba_updated_name, reduced_proba_updated_name


def _normalise_proba(scope, operator, container, proba, num_classes,
                     unity_name):
    reduced_proba_name = scope.get_unique_variable_name('reduced_proba')
    sub_result_name = scope.get_unique_variable_name('sub_result')

    if num_classes == 2:
        apply_sub(scope, [unity_name, proba],
                  sub_result_name, container, broadcast=1)
        apply_concat(scope, [sub_result_name, proba],
                     operator.outputs[1].full_name, container, axis=1)
    else:
        if container.target_opset < 13:
            container.add_node(
                'ReduceSum', proba, reduced_proba_name, axes=[1],
                name=scope.get_unique_operator_name('ReduceSum'))
        else:
            axis_name = scope.get_unique_variable_name('axis')
            container.add_initializer(
                axis_name, onnx_proto.TensorProto.INT64, [1], [1])
            container.add_node(
                'ReduceSum', [proba, axis_name], reduced_proba_name,
                name=scope.get_unique_operator_name('ReduceSum'))
        proba_updated, reduced_proba_updated = _handle_zeros(
            scope, container, proba, reduced_proba_name, num_classes)
        apply_div(scope, [proba_updated, reduced_proba_updated],
                  operator.outputs[1].full_name, container, broadcast=1)
    return operator.outputs[1].full_name


def _predict_proba_log(scope, operator, container, scores, num_classes):
    """Probability estimation for SGDClassifier with loss=log and
    Logistic Regression.
    Positive class probabilities are computed as
        1. / (1. + exp(-scores))
        multiclass is handled by normalising that over all classes.
    """
    negate_name = scope.get_unique_variable_name('negate')
    negated_scores_name = scope.get_unique_variable_name('negated_scores')
    exp_result_name = scope.get_unique_variable_name('exp_result')
    unity_name = scope.get_unique_variable_name('unity')
    add_result_name = scope.get_unique_variable_name('add_result')
    proba_name = scope.get_unique_variable_name('proba')

    container.add_initializer(negate_name, onnx_proto.TensorProto.FLOAT,
                              [], [-1])
    container.add_initializer(unity_name, onnx_proto.TensorProto.FLOAT,
                              [], [1])

    apply_mul(scope, [scores, negate_name],
              negated_scores_name, container, broadcast=1)
    apply_exp(scope, negated_scores_name, exp_result_name, container)
    apply_add(scope, [exp_result_name, unity_name],
              add_result_name, container, broadcast=1)
    apply_reciprocal(scope, add_result_name, proba_name, container)
    return _normalise_proba(scope, operator, container, proba_name,
                            num_classes, unity_name)


def _predict_proba_modified_huber(scope, operator, container,
                                  scores, num_classes):
    """Probability estimation for SGDClassifier with
    loss=modified_huber.
    Multiclass probability estimates are derived from binary
    estimates by normalisation.
    Binary probability estimates are given by
    (clip(scores, -1, 1) + 1) / 2.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    unity_name = scope.get_unique_variable_name('unity')
    constant_name = scope.get_unique_variable_name('constant')
    add_result_name = scope.get_unique_variable_name('add_result')
    proba_name = scope.get_unique_variable_name('proba')
    clipped_scores_name = scope.get_unique_variable_name('clipped_scores')

    container.add_initializer(unity_name, onnx_proto.TensorProto.FLOAT,
                              [], [1])
    container.add_initializer(constant_name, onnx_proto.TensorProto.FLOAT,
                              [], [2])

    apply_clip(scope, scores, clipped_scores_name, container,
               max=np.array(1, dtype=dtype),
               min=np.array(-1, dtype=dtype))
    apply_add(scope, [clipped_scores_name, unity_name],
              add_result_name, container, broadcast=1)
    apply_div(scope, [add_result_name, constant_name],
              proba_name, container, broadcast=1)
    return _normalise_proba(scope, operator, container, proba_name,
                            num_classes, unity_name)


def convert_sklearn_sgd_classifier(scope, operator, container):
    """Converter for SGDClassifier."""
    sgd_op = operator.raw_operator
    classes = get_label_classes(scope, sgd_op)
    class_type = onnx_proto.TensorProto.STRING

    if np.issubdtype(classes.dtype, np.floating):
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype(np.int32)
    elif np.issubdtype(classes.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode('utf-8') for s in classes])

    classes_name = scope.get_unique_variable_name('classes')
    predicted_label_name = scope.get_unique_variable_name(
        'predicted_label')
    final_label_name = scope.get_unique_variable_name('final_label')

    container.add_initializer(classes_name, class_type,
                              classes.shape, classes)

    scores = _decision_function(scope, operator, container, sgd_op)
    options = container.get_options(sgd_op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']
    if sgd_op.loss == 'log' and not use_raw_scores:
        proba = _predict_proba_log(scope, operator, container, scores,
                                   len(classes))
    elif sgd_op.loss == 'modified_huber' and not use_raw_scores:
        proba = _predict_proba_modified_huber(
            scope, operator, container, scores, len(classes))
    else:
        if len(classes) == 2:
            negate_name = scope.get_unique_variable_name('negate')
            negated_scores_name = scope.get_unique_variable_name(
                'negated_scores')

            container.add_initializer(
                negate_name, onnx_proto.TensorProto.FLOAT, [], [-1])

            apply_mul(scope, [scores, negate_name],
                      negated_scores_name, container, broadcast=1)
            apply_concat(scope, [negated_scores_name, scores],
                         operator.outputs[1].full_name, container, axis=1)
        else:
            apply_identity(scope, scores,
                           operator.outputs[1].full_name, container)
        proba = operator.outputs[1].full_name

    container.add_node('ArgMax', proba,
                       predicted_label_name,
                       name=scope.get_unique_operator_name('ArgMax'), axis=1)
    container.add_node(
        'ArrayFeatureExtractor', [classes_name, predicted_label_name],
        final_label_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    if class_type == onnx_proto.TensorProto.INT32:
        reshaped_final_label_name = scope.get_unique_variable_name(
            'reshaped_final_label')

        apply_reshape(scope, final_label_name, reshaped_final_label_name,
                      container, desired_shape=(-1,))
        apply_cast(scope, reshaped_final_label_name,
                   operator.outputs[0].full_name, container,
                   to=onnx_proto.TensorProto.INT64)
    else:
        apply_reshape(scope, final_label_name,
                      operator.outputs[0].full_name, container,
                      desired_shape=(-1,))


register_converter('SklearnSGDClassifier',
                   convert_sklearn_sgd_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'raw_scores': [True, False]})
