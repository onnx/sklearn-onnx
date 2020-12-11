# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    RidgeClassifierCV,
)
from sklearn.svm import LinearSVC
from ..common._apply_operation import apply_cast
from ..common._registration import register_converter
from ..common.data_types import BooleanTensorType
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto


def convert_sklearn_linear_classifier(scope, operator, container):
    op = operator.raw_operator
    coefficients = op.coef_.flatten().astype(float).tolist()
    classes = get_label_classes(scope, op)
    number_of_classes = len(classes)

    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']

    if isinstance(op.intercept_, (float, np.float32)) and op.intercept_ == 0:
        # fit_intercept = False
        intercepts = ([0.0] * number_of_classes if number_of_classes != 2 else
                      [0.0])
    else:
        intercepts = op.intercept_.tolist()

    if number_of_classes == 2:
        coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
        intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

    multi_class = 0
    if hasattr(op, 'multi_class'):
        if op.multi_class == 'ovr':
            multi_class = 1
        else:
            multi_class = 2

    classifier_type = 'LinearClassifier'
    classifier_attrs = {
        'name': scope.get_unique_operator_name(classifier_type)
    }

    classifier_attrs['coefficients'] = coefficients
    classifier_attrs['intercepts'] = intercepts
    classifier_attrs['multi_class'] = 1 if multi_class == 2 else 0
    if (use_raw_scores or
            isinstance(op, (LinearSVC, RidgeClassifier, RidgeClassifierCV))):
        classifier_attrs['post_transform'] = 'NONE'
    elif isinstance(op, LogisticRegression):
        ovr = (op.multi_class in ["ovr", "warn"] or
               (op.multi_class == 'auto' and (op.classes_.size <= 2 or
                                              op.solver == 'liblinear')))
        classifier_attrs['post_transform'] = (
            'LOGISTIC' if ovr else 'SOFTMAX')
    else:
        classifier_attrs['post_transform'] = (
            'LOGISTIC' if multi_class > 2 else 'SOFTMAX')

    if all(isinstance(i, (six.string_types, six.text_type)) for i in classes):
        class_labels = [str(i) for i in classes]
        classifier_attrs['classlabels_strings'] = class_labels
    elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        classifier_attrs['classlabels_ints'] = class_labels
    else:
        raise RuntimeError('Label vector must be a string or a integer '
                           'tensor.')

    label_name = operator.outputs[0].full_name
    input_name = operator.inputs[0].full_name
    if type(operator.inputs[0].type) == BooleanTensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, input_name, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = cast_input_name

    if use_raw_scores:
        container.add_node(classifier_type, input_name,
                           [label_name, operator.outputs[1].full_name],
                           op_domain='ai.onnx.ml', **classifier_attrs)
    elif (isinstance(op, (LinearSVC, RidgeClassifier, RidgeClassifierCV))
            and op.classes_.shape[0] <= 2):
        raw_scores_tensor_name = scope.get_unique_variable_name(
            'raw_scores_tensor')
        positive_class_index_name = scope.get_unique_variable_name(
            'positive_class_index')

        container.add_initializer(positive_class_index_name,
                                  onnx_proto.TensorProto.INT64, [], [1])

        if (hasattr(op, '_label_binarizer') and
                op._label_binarizer.y_type_ == 'multilabel-indicator'):
            y_pred_name = scope.get_unique_variable_name('y_pred')
            binarised_label_name = scope.get_unique_variable_name(
                'binarised_label')

            container.add_node(classifier_type, input_name,
                               [y_pred_name, raw_scores_tensor_name],
                               op_domain='ai.onnx.ml', **classifier_attrs)
            container.add_node(
                'Binarizer', raw_scores_tensor_name, binarised_label_name,
                op_domain='ai.onnx.ml')
            apply_cast(
                scope, binarised_label_name, label_name,
                container, to=onnx_proto.TensorProto.INT64)
        else:
            container.add_node(classifier_type, input_name,
                               [label_name, raw_scores_tensor_name],
                               op_domain='ai.onnx.ml', **classifier_attrs)
        container.add_node(
            'ArrayFeatureExtractor',
            [raw_scores_tensor_name, positive_class_index_name],
            operator.outputs[1].full_name, op_domain='ai.onnx.ml',
            name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    else:
        # Make sure the probability sum is 1 over all classes
        if multi_class > 0 and not isinstance(
                op, (LinearSVC, RidgeClassifier, RidgeClassifierCV)):
            probability_tensor_name = scope.get_unique_variable_name(
                'probability_tensor')
            container.add_node(classifier_type, input_name,
                               [label_name, probability_tensor_name],
                               op_domain='ai.onnx.ml', **classifier_attrs)
            normalizer_type = 'Normalizer'
            normalizer_attrs = {
                'name': scope.get_unique_operator_name(normalizer_type),
                'norm': 'L1'
            }
            container.add_node(normalizer_type, probability_tensor_name,
                               operator.outputs[1].full_name,
                               op_domain='ai.onnx.ml', **normalizer_attrs)
        elif (hasattr(op, '_label_binarizer') and
              op._label_binarizer.y_type_ == 'multilabel-indicator'):
            y_pred_name = scope.get_unique_variable_name('y_pred')
            binarised_label_name = scope.get_unique_variable_name(
                'binarised_label')

            container.add_node(
                classifier_type, input_name,
                [y_pred_name, operator.outputs[1].full_name],
                op_domain='ai.onnx.ml', **classifier_attrs)
            container.add_node(
                'Binarizer', operator.outputs[1].full_name,
                binarised_label_name, op_domain='ai.onnx.ml')
            apply_cast(
                scope, binarised_label_name, label_name,
                container, to=onnx_proto.TensorProto.INT64)
        else:
            container.add_node(classifier_type, input_name,
                               [label_name, operator.outputs[1].full_name],
                               op_domain='ai.onnx.ml', **classifier_attrs)


register_converter('SklearnLinearClassifier',
                   convert_sklearn_linear_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'raw_scores': [True, False]})
register_converter('SklearnLinearSVC', convert_sklearn_linear_classifier,
                   options={'nocl': [True, False],
                            'raw_scores': [True, False]})
