# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._apply_operation import apply_cast, apply_concat
from ..common._apply_operation import apply_mul, apply_reshape
from ..common._registration import register_converter
from ..proto import onnx_proto
from .sgd_classifier import _decision_function, _predict_proba_log


def convert_sklearn_logistic_regression(scope, operator, container):
    """Converter for LogisticRegression."""
    lr_op = operator.raw_operator
    classes = lr_op.classes_
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

    scores = _decision_function(scope, operator, container, lr_op)
    ovr = (lr_op.multi_class in ['ovr', 'warn'] or
           (lr_op.multi_class == 'auto' and
            (classes.size <= 2 or lr_op.solver == 'liblinear')))

    if ovr:
        proba = _predict_proba_log(scope, operator, container, scores,
                                   len(classes))
    else:
        if len(classes) == 2:
            negate_name = scope.get_unique_variable_name('negate')
            negated_scores_name = scope.get_unique_variable_name(
                'negated_scores')
            decision_name = scope.get_unique_variable_name('decision')

            container.add_initializer(
                negate_name, onnx_proto.TensorProto.FLOAT, [], [-1])

            apply_mul(scope, [scores, negate_name],
                      negated_scores_name, container, broadcast=1)
            apply_concat(scope, [negated_scores_name, scores],
                         decision_name, container, axis=1)
            scores = decision_name
        container.add_node(
            'Softmax', scores, operator.outputs[1].full_name,
            name=scope.get_unique_operator_name('Softmax'))
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


register_converter('SklearnLogisticRegression',
                   convert_sklearn_logistic_regression)
