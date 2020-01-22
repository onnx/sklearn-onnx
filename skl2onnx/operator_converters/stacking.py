# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np

from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_cast,
    apply_concat,
    apply_reshape,
)
from ..common._topology import FloatTensorType
from ..common._registration import register_converter
from .._supported_operators import sklearn_operator_name_map


def _fetch_scores(scope, container, model, inputs, output_proba=None,
                  raw_scores=False):
    op_type = sklearn_operator_name_map[type(model)]
    this_operator = scope.declare_local_operator(op_type)
    this_operator.raw_operator = model
    container.add_options(id(model), {'raw_scores': raw_scores})
    this_operator.inputs.append(inputs)
    label_name = scope.declare_local_variable('label')
    if output_proba is None:
        output_proba = scope.declare_local_variable(
            'probability_tensor', FloatTensorType())
    this_operator.outputs.append(label_name)
    this_operator.outputs.append(output_proba)
    return output_proba.full_name


def _transform(scope, operator, container, model, raw_scores=False):
    merged_prob_tensor = scope.declare_local_variable(
        'merged_probability_tensor', FloatTensorType())

    predictions = [
        _fetch_scores(scope, container, est, operator.inputs[0],
                      raw_scores=meth == 'decision_function')
        for est, meth in zip(model.estimators_, model.stack_method_)
        if est != 'drop'
    ]
    apply_concat(
        scope, predictions, merged_prob_tensor.full_name, container, axis=1)
    return merged_prob_tensor


def convert_sklearn_stacking_classifier(scope, operator, container):
    stacking_op = operator.raw_operator
    classes = stacking_op.classes_
    options = container.get_options(stacking_op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']
    class_type = onnx_proto.TensorProto.STRING
    if np.issubdtype(stacking_op.classes_.dtype, np.floating):
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype(np.int32)
    elif np.issubdtype(stacking_op.classes_.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode('utf-8') for s in classes])

    classes_name = scope.get_unique_variable_name('classes')
    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        'array_feature_extractor_result')

    container.add_initializer(classes_name, class_type, classes.shape, classes)

    merged_proba_tensor = _transform(
        scope, operator, container, stacking_op,
        raw_scores=use_raw_scores)
    prob = _fetch_scores(
        scope, container, stacking_op.final_estimator_, merged_proba_tensor,
        output_proba=operator.outputs[1], raw_scores=use_raw_scores)
    container.add_node('ArgMax', prob,
                       argmax_output_name,
                       name=scope.get_unique_operator_name('ArgMax'), axis=1)
    container.add_node(
        'ArrayFeatureExtractor', [classes_name, argmax_output_name],
        array_feature_extractor_result_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))

    if class_type == onnx_proto.TensorProto.INT32:
        apply_reshape(scope, array_feature_extractor_result_name,
                      reshaped_result_name, container,
                      desired_shape=(-1,))
        apply_cast(scope, reshaped_result_name, operator.outputs[0].full_name,
                   container, to=onnx_proto.TensorProto.INT64)
    else:
        apply_reshape(scope, array_feature_extractor_result_name,
                      operator.outputs[0].full_name, container,
                      desired_shape=(-1,))


register_converter('SklearnStackingClassifier',
                   convert_sklearn_stacking_classifier)
