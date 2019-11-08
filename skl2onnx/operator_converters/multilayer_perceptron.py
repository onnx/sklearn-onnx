# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._apply_operation import (
    apply_add, apply_cast, apply_concat, apply_identity,
    apply_reshape, apply_sub)
from ..common._registration import register_converter
from ..proto import onnx_proto


def _forward_pass(scope, container, model, activations):
    """
    Perform a forward pass on the network by computing the values of
    the neurons in the hidden layers and the output layer.
    """
    activations_map = {
        'identity': 'Identity', 'tanh': 'Tanh', 'logistic': 'Sigmoid',
        'relu': 'Relu', 'softmax': 'Softmax'
    }

    out_activation_result_name = scope.get_unique_variable_name(
        'out_activations_result')

    # Iterate over the hidden layers
    for i in range(model.n_layers_ - 1):
        coefficient_name = scope.get_unique_variable_name('coefficient')
        intercepts_name = scope.get_unique_variable_name('intercepts')
        mul_result_name = scope.get_unique_variable_name('mul_result')
        add_result_name = scope.get_unique_variable_name('add_result')

        container.add_initializer(
            coefficient_name, container.proto_dtype,
            model.coefs_[i].shape, model.coefs_[i].ravel())
        container.add_initializer(
            intercepts_name, container.proto_dtype,
            [1, len(model.intercepts_[i])], model.intercepts_[i])

        container.add_node(
            'MatMul', [activations[i], coefficient_name],
            mul_result_name, name=scope.get_unique_operator_name('MatMul'))
        apply_add(scope, [mul_result_name, intercepts_name],
                  add_result_name, container, broadcast=1)

        # For the hidden layers
        if (i + 1) != (model.n_layers_ - 1):
            activations_result_name = scope.get_unique_variable_name(
                'next_activations')

            container.add_node(
                activations_map[model.activation], add_result_name,
                activations_result_name,
                name=scope.get_unique_operator_name(
                    activations_map[model.activation]))
            activations.append(activations_result_name)

    # For the last layer
    container.add_node(
        activations_map[model.out_activation_], add_result_name,
        out_activation_result_name,
        name=scope.get_unique_operator_name(activations_map[model.activation]))
    activations.append(out_activation_result_name)

    return activations


def _predict(scope, input_name, container, model):
    """
    This function initialises the input layer, calls _forward_pass()
    and returns the final layer.
    """
    cast_input_name = scope.get_unique_variable_name('cast_input')

    apply_cast(scope, input_name, cast_input_name,
               container, to=container.proto_dtype)

    # forward propagate
    activations = _forward_pass(scope, container, model, [cast_input_name])
    return activations[-1]


def convert_sklearn_mlp_classifier(scope, operator, container):
    """
    Converter for MLPClassifier.
    This function calls _predict() which returns the probability scores
    of the positive class in case of binary labels and class
    probabilities in case of multi-class. It then calculates probability
    scores for the negative class in case of binary labels. It
    calculates the class labels and sets the output.
    """
    mlp_op = operator.raw_operator
    classes = mlp_op.classes_
    class_type = onnx_proto.TensorProto.STRING

    classes_name = scope.get_unique_variable_name('classes')
    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        'array_feature_extractor_result')

    y_pred = _predict(scope, operator.inputs[0].full_name, container, mlp_op)

    if np.issubdtype(mlp_op.classes_.dtype, np.floating):
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype(np.int32)
    elif np.issubdtype(mlp_op.classes_.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode('utf-8') for s in classes])

    container.add_initializer(classes_name, class_type,
                              classes.shape, classes)

    if len(classes) == 2:
        unity_name = scope.get_unique_variable_name('unity')
        negative_class_proba_name = scope.get_unique_variable_name(
            'negative_class_proba')
        container.add_initializer(unity_name, container.proto_dtype,
                                  [], [1])

        apply_sub(scope, [unity_name, y_pred],
                  negative_class_proba_name, container, broadcast=1)
        apply_concat(scope, [negative_class_proba_name, y_pred],
                     operator.outputs[1].full_name, container, axis=1)
    else:
        apply_identity(scope, y_pred,
                       operator.outputs[1].full_name, container)

    if mlp_op._label_binarizer.y_type_ == 'multilabel-indicator':
        container.add_node('Binarizer', y_pred, operator.outputs[0].full_name,
                           threshold=0.5, op_domain='ai.onnx.ml')
    else:
        container.add_node('ArgMax', operator.outputs[1].full_name,
                           argmax_output_name, axis=1,
                           name=scope.get_unique_operator_name('ArgMax'))
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
            apply_cast(
                scope, reshaped_result_name, operator.outputs[0].full_name,
                container, to=onnx_proto.TensorProto.INT64)
        else:
            apply_reshape(scope, array_feature_extractor_result_name,
                          operator.outputs[0].full_name, container,
                          desired_shape=(-1,))


def convert_sklearn_mlp_regressor(scope, operator, container):
    """
    Converter for MLPRegressor.
    This function calls _predict() which returns the scores.
    """
    mlp_op = operator.raw_operator

    y_pred = _predict(scope, operator.inputs[0].full_name, container, mlp_op)
    apply_reshape(scope, y_pred, operator.output_full_names,
                  container, desired_shape=(-1, 1))


register_converter('SklearnMLPClassifier',
                   convert_sklearn_mlp_classifier)
register_converter('SklearnMLPRegressor',
                   convert_sklearn_mlp_regressor)
