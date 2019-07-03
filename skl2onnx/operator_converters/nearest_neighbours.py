# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np

from ..common._apply_operation import apply_abs, apply_cast, apply_mul
from ..common._apply_operation import apply_add, apply_div
from ..common._apply_operation import apply_reshape, apply_sub, apply_topk
from ..common._apply_operation import apply_pow, apply_concat, apply_transpose
from ..common.data_types import Int64TensorType
from ..common._registration import register_converter
from ..proto import onnx_proto


def _calculate_weights(scope, container, unity, distance):
    """
    weights = 1 / distance
    Handle divide by 0.
    """
    weights_name = scope.get_unique_variable_name('weights')
    ceil_result_name = scope.get_unique_variable_name('ceil_result')
    floor_result_name = scope.get_unique_variable_name('floor_result')
    mask_sum_name = scope.get_unique_variable_name('mask_sum')
    bool_mask_sum_name = scope.get_unique_variable_name('bool_mask_sum')
    ceil_floor_sum_name = scope.get_unique_variable_name('ceil_floor_sum')
    distance_without_zero_name = scope.get_unique_variable_name(
        'distance_without_zero')
    not_ceil_floor_sum_name = scope.get_unique_variable_name(
        'not_ceil_floor_sum')
    bool_ceil_floor_sum_name = scope.get_unique_variable_name(
        'bool_ceil_floor_sum')
    bool_not_ceil_floor_sum_name = scope.get_unique_variable_name(
        'bool_not_ceil_floor_sum')
    mask_sum_complement_name = scope.get_unique_variable_name(
        'mask_sum_complement')
    mask_sum_complement_float_name = scope.get_unique_variable_name(
        'mask_sum_complement_float')
    masked_weights_name = scope.get_unique_variable_name('masked_weights')
    final_weights_name = scope.get_unique_variable_name('final_weights')

    container.add_node('Ceil', distance, ceil_result_name,
                       name=scope.get_unique_operator_name('Ceil'))
    container.add_node('Floor', distance, floor_result_name,
                       name=scope.get_unique_operator_name('Floor'))
    apply_add(scope, [ceil_result_name, floor_result_name],
              ceil_floor_sum_name, container, broadcast=0)
    apply_cast(scope, ceil_floor_sum_name, bool_ceil_floor_sum_name, container,
               to=onnx_proto.TensorProto.BOOL)
    container.add_node('Not', bool_ceil_floor_sum_name,
                       bool_not_ceil_floor_sum_name,
                       name=scope.get_unique_operator_name('Not'))
    apply_cast(scope, bool_not_ceil_floor_sum_name, not_ceil_floor_sum_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    apply_add(scope, [distance, not_ceil_floor_sum_name],
              distance_without_zero_name, container, broadcast=0)
    apply_div(scope, [unity, distance_without_zero_name],
              weights_name, container, broadcast=1)
    container.add_node('ReduceSum', not_ceil_floor_sum_name,
                       mask_sum_name, axes=[1],
                       name=scope.get_unique_operator_name('ReduceSum'))
    apply_cast(scope, mask_sum_name, bool_mask_sum_name, container,
               to=onnx_proto.TensorProto.BOOL)
    container.add_node('Not', bool_mask_sum_name,
                       mask_sum_complement_name,
                       name=scope.get_unique_operator_name('Not'))
    apply_cast(scope, mask_sum_complement_name, mask_sum_complement_float_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    apply_mul(scope, [weights_name, mask_sum_complement_float_name],
              masked_weights_name, container, broadcast=1)
    apply_add(scope, [masked_weights_name, not_ceil_floor_sum_name],
              final_weights_name, container, broadcast=0)
    return final_weights_name


def _get_weights(scope, container, topk_values_name, distance_power):
    """
    Get the weights from an array of distances.
    """
    unity_name = scope.get_unique_variable_name('unity')
    root_power_name = scope.get_unique_variable_name('root_power')
    nearest_distance_name = scope.get_unique_variable_name(
        'nearest_distance')
    actual_distance_name = scope.get_unique_variable_name(
        'actual_distance')

    container.add_initializer(unity_name, onnx_proto.TensorProto.FLOAT,
                              [], [1])
    container.add_initializer(root_power_name,
                              onnx_proto.TensorProto.FLOAT,
                              [], [1 / distance_power])

    apply_abs(scope, topk_values_name, nearest_distance_name,
              container)
    apply_pow(scope, [nearest_distance_name, root_power_name],
              actual_distance_name, container)
    weights_name = _calculate_weights(scope, container, unity_name,
                                      actual_distance_name)
    return weights_name


def _get_probability_score(scope, container, operator, weights,
                           topk_values_name, distance_power, topk_labels_name,
                           classes):
    """
    Calculate the class probability scores, update the second output of
    KNeighboursClassifier converter with the probability scores and
    return it.
    """
    labels_name = [None] * len(classes)
    output_label_name = [None] * len(classes)
    output_cast_label_name = [None] * len(classes)
    output_label_reduced_name = [None] * len(classes)

    for i in range(len(classes)):
        labels_name[i] = scope.get_unique_variable_name(
            'class_labels_{}'.format(i))
        container.add_initializer(labels_name[i],
                                  onnx_proto.TensorProto.INT32, [], [i])
        output_label_name[i] = scope.get_unique_variable_name(
            'output_label_{}'.format(i))
        output_cast_label_name[i] = scope.get_unique_variable_name(
            'output_cast_label_{}'.format(i))
        output_label_reduced_name[i] = scope.get_unique_variable_name(
            'output_label_reduced_{}'.format(i))
    if weights == 'distance':
        weights_val = _get_weights(
            scope, container, topk_values_name, distance_power)
        for i in range(len(classes)):
            weighted_distance_name = scope.get_unique_variable_name(
                'weighted_distance')

            container.add_node('Equal', [labels_name[i], topk_labels_name],
                               output_label_name[i],
                               name=scope.get_unique_operator_name('Equal'))
            apply_cast(scope, output_label_name[i], output_cast_label_name[i],
                       container, to=onnx_proto.TensorProto.FLOAT)
            apply_mul(scope, [output_cast_label_name[i], weights_val],
                      weighted_distance_name, container, broadcast=0)
            container.add_node('ReduceSum', weighted_distance_name,
                               output_label_reduced_name[i], axes=[1],
                               name=scope.get_unique_operator_name(
                                'ReduceSum'))
    else:
        for i in range(len(classes)):
            container.add_node('Equal', [labels_name[i], topk_labels_name],
                               output_label_name[i],
                               name=scope.get_unique_operator_name('Equal'))
            apply_cast(scope, output_label_name[i], output_cast_label_name[i],
                       container, to=onnx_proto.TensorProto.INT32)
            container.add_node('ReduceSum', output_cast_label_name[i],
                               output_label_reduced_name[i], axes=[1],
                               name=scope.get_unique_operator_name(
                                'ReduceSum'))

    concat_labels_name = scope.get_unique_variable_name('concat_labels')
    cast_concat_labels_name = scope.get_unique_variable_name(
        'cast_concat_labels')
    normaliser_name = scope.get_unique_variable_name('normaliser')

    apply_concat(scope, output_label_reduced_name,
                 concat_labels_name, container, axis=1)
    apply_cast(scope, concat_labels_name, cast_concat_labels_name,
               container, to=onnx_proto.TensorProto.FLOAT)
    container.add_node('ReduceSum', cast_concat_labels_name,
                       normaliser_name, axes=[1],
                       name=scope.get_unique_operator_name('ReduceSum'))
    apply_div(scope, [cast_concat_labels_name, normaliser_name],
              operator.outputs[1].full_name, container, broadcast=1)
    return operator.outputs[1].full_name


def _convert_k_neighbours_classifier(scope, container, operator, classes,
                                     class_type, training_labels,
                                     topk_values_name, topk_indices_name,
                                     distance_power, weights):
    """
    Convert KNeighboursClassifier model to onnx format.
    """
    classes_name = scope.get_unique_variable_name('classes')
    predicted_label_name = scope.get_unique_variable_name(
        'predicted_label')
    final_label_name = scope.get_unique_variable_name('final_label')
    training_labels_name = scope.get_unique_variable_name(
        'training_labels')
    topk_labels_name = scope.get_unique_variable_name('topk_labels')

    container.add_initializer(classes_name, class_type,
                              classes.shape, classes)
    container.add_initializer(
        training_labels_name, onnx_proto.TensorProto.INT32,
        training_labels.shape, training_labels.ravel())

    container.add_node(
        'ArrayFeatureExtractor', [training_labels_name, topk_indices_name],
        topk_labels_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    proba = _get_probability_score(scope, container, operator,
                                   weights, topk_values_name, distance_power,
                                   topk_labels_name, classes)
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


def _convert_k_neighbours_regressor(scope, container, new_training_labels,
                                    new_training_labels_shape,
                                    topk_values_name, topk_indices_name,
                                    distance_power, weights):
    """
    Convert KNeighboursRegressor model to onnx format.
    """
    training_labels_name = scope.get_unique_variable_name(
        'training_labels')
    topk_labels_name = scope.get_unique_variable_name('topk_labels')

    container.add_initializer(
        training_labels_name, onnx_proto.TensorProto.FLOAT,
        new_training_labels_shape,
        new_training_labels.ravel().astype(float))

    container.add_node(
        'ArrayFeatureExtractor', [training_labels_name, topk_indices_name],
        topk_labels_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    weighted_labels = topk_labels_name
    final_op_type = 'ReduceMean'
    if weights == 'distance':
        weighted_distance_name = scope.get_unique_variable_name(
            'weighted_distance')
        reduced_weights_name = scope.get_unique_variable_name(
            'reduced_weights')
        weighted_labels_name = scope.get_unique_variable_name(
            'weighted_labels')

        weights_val = _get_weights(
            scope, container, topk_values_name, distance_power)
        apply_mul(scope, [topk_labels_name, weights_val],
                  weighted_distance_name, container, broadcast=0)
        container.add_node(
            'ReduceSum', weights_val, reduced_weights_name,
            name=scope.get_unique_operator_name('ReduceSum'), axes=[1])
        apply_div(scope, [weighted_distance_name, reduced_weights_name],
                  weighted_labels_name, container, broadcast=1)
        weighted_labels = weighted_labels_name
        final_op_type = 'ReduceSum'
    return final_op_type, weighted_labels


def convert_sklearn_knn(scope, operator, container):
    """
    Converter for KNN models to onnx format.
    """
    # Computational graph:
    #
    # In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names
    # from the official ONNX spec:
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    # Note that KNN regressor and classifier share the same computation graphs
    # until the top-k nearest examples' labels (aka `topk_labels` in the graph
    # below) are found.
    #
    # Symbols:
    # M: Number of training set instances
    # N: Number of features
    # C: Number of classes
    # input: input
    # output: output
    # output_prob (for KNN Classifier): class probabilities
    #
    # Graph:
    #
    #   input [1, N] --> SUB <---- training_examples [M, N]
    #                     |
    #                     V
    #           sub_results [M, N] ----> POW <---- distance_power [1]
    #                                     |
    #                                     V
    #  reduced_sum [M] <-- REDUCESUM <-- distance [M, N]
    #            |
    #            V
    # length -> RESHAPE -> reshaped_result [1, M]
    #                       |
    #                       V
    # n_neighbors [1] ----> TOPK
    #                       |
    #                      / \
    #                     /   \
    #                     |    |
    #                     V    V
    #       topk_indices [K]   topk_values [K]
    #               |
    #               V
    #   ARRAYFEATUREEXTRACTOR <- training_labels [M]
    #           |
    #           V                   (KNN Regressor)
    #          topk_labels [K] ---------------------> REDUCEMEAN --> output [1]
    #                    |
    #                    |
    #                    | (KNN Classifier)
    #                    |
    #                    |------------------------------------------------.
    #                   /|\                (probability calculation)      |
    #                  / | \                                              |
    #                 /  |  \ (label prediction)                          V
    #                /   |   \                                          CAST
    #               /    |    \__                                         |
    #               |    |       |                                        V
    #               V    V       V                       cast_pred_label [K, 1]
    # label0 -> EQUAL  EQUAL ... EQUAL <- label(C-1)                      |
    #            |       |          |                                     |
    #            V       V          V                                     |
    # output_label_0 [C] ...       output_label_(C-1) [C]                 |
    #            |       |          |                                     V
    #            V       V          V           pred_label_shape [2] -> RESHAPE
    #          CAST    CAST    ... CAST                                   |
    #            |       |          |                                     V
    #            V       V          V                reshaped_pred_label [K, 1]
    # output_cast_label_0 [C] ...  output_cast_label_(C-1) [C]            |
    #            |       |          |                                     |
    #            V       V          V                                     |
    #      REDUCESUM  REDUCESUM ... REDUCESUM                             |
    #            |       |          |                                     |
    #            V       V          V                                     |
    # output_label_reduced_0 [1] ... output_label_reduced_(C-1) [1]       |
    #           \        |           /                                    |
    #            \____   |      ____/                                     |
    #                 \  |  ___/                                          |
    #                  \ | /                                              |
    #                   \|/                                               |
    #                    V                                                |
    #                 CONCAT --> concat_labels [C]                        |
    #                               |                                     |
    #                               V                                     |
    #                           ARGMAX --> predicted_label [1]            |
    #                                       |                             |
    #                                       V                             |
    #            output [1] <--- ARRAYFEATUREEXTRACTOR <- classes [C]     |
    #                                                                     |
    #                                                                     |
    #                                                                     |
    #   ohe_model --> ONEHOTENCODER <-------------------------------------'
    #                   |
    #                   V
    #  ohe_result [n_neighbors, C] -> REDUCEMEAN -> reduced_prob [1, C]
    #                                                |
    #                                                V
    #               output_probability [1, C]  <-  ZipMap

    knn = operator.raw_operator
    training_examples = knn._fit_X.astype(float)
    distance_power = knn.p if knn.metric == 'minkowski' else (
        2 if knn.metric == 'euclidean' or knn.metric == 'l2' else 1)

    if operator.type != 'SklearnNearestNeighbors':
        training_labels = knn._y

    training_examples_name = scope.get_unique_variable_name(
        'training_examples')
    sub_results_name = scope.get_unique_variable_name('sub_results')
    abs_results_name = scope.get_unique_variable_name('abs_results')
    distance_name = scope.get_unique_variable_name('distance')
    distance_power_name = scope.get_unique_variable_name('distance_power')
    reduced_sum_name = scope.get_unique_variable_name('reduced_sum')
    topk_values_name = scope.get_unique_variable_name('topk_values')
    topk_indices_name = scope.get_unique_variable_name('topk_indices')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')
    negate_name = scope.get_unique_variable_name('negate')
    negated_reshaped_result_name = scope.get_unique_variable_name(
        'negated_reshaped_result')

    container.add_initializer(
        training_examples_name, onnx_proto.TensorProto.FLOAT,
        training_examples.shape, training_examples.flatten())
    container.add_initializer(distance_power_name,
                              onnx_proto.TensorProto.FLOAT,
                              [], [distance_power])
    container.add_initializer(negate_name, onnx_proto.TensorProto.FLOAT,
                              [], [-1])

    input_name = operator.inputs[0].full_name
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.inputs[0].full_name, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = cast_input_name

    apply_sub(scope, [input_name, training_examples_name],
              sub_results_name, container, broadcast=1)
    apply_abs(scope, sub_results_name, abs_results_name, container)
    apply_pow(scope, [abs_results_name, distance_power_name], distance_name,
              container)
    container.add_node('ReduceSum', distance_name, reduced_sum_name,
                       name=scope.get_unique_operator_name('ReduceSum'),
                       axes=[1])
    apply_reshape(scope, reduced_sum_name, reshaped_result_name, container,
                  desired_shape=[1, -1])
    apply_mul(scope, [reshaped_result_name, negate_name],
              negated_reshaped_result_name, container, broadcast=1)
    apply_topk(scope, negated_reshaped_result_name,
               [topk_values_name, topk_indices_name], container,
               k=knn.n_neighbors)

    if operator.type == 'SklearnKNeighborsClassifier':
        classes = knn.classes_
        class_type = onnx_proto.TensorProto.STRING

        if np.issubdtype(knn.classes_.dtype, np.floating):
            class_type = onnx_proto.TensorProto.INT32
            classes = classes.astype(np.int32)
        elif np.issubdtype(knn.classes_.dtype, np.signedinteger):
            class_type = onnx_proto.TensorProto.INT32
        else:
            classes = np.array([s.encode('utf-8') for s in classes])

        _convert_k_neighbours_classifier(
            scope, container, operator, classes, class_type, training_labels,
            topk_values_name, topk_indices_name, distance_power,
            knn.weights)
    elif operator.type == 'SklearnKNeighborsRegressor':
        multi_reg = (len(training_labels.shape) > 1 and
                     (len(training_labels.shape) > 2 or
                      training_labels.shape[1] > 1))
        if multi_reg:
            shape = training_labels.shape
            irange = tuple(range(len(shape)))
            new_shape = (shape[-1],) + shape[:-1]
            perm = irange[-1:] + irange[:-1]
            new_training_labels = training_labels.transpose(perm)
            perm = irange[1:] + (0,)
            shape = new_shape
        else:
            shape = training_labels.shape
            new_training_labels = training_labels

        final_op_type, weighted_labels = _convert_k_neighbours_regressor(
            scope, container, new_training_labels, shape,
            topk_values_name, topk_indices_name, distance_power, knn.weights)
        if multi_reg:
            means_name = scope.get_unique_variable_name('means')
            container.add_node(
                final_op_type, weighted_labels, means_name,
                name=scope.get_unique_operator_name(final_op_type), axes=[1])
            apply_transpose(scope, means_name, operator.output_full_names,
                            container, perm=perm)
        else:
            container.add_node(
                final_op_type, weighted_labels, operator.output_full_names,
                name=scope.get_unique_operator_name(final_op_type), axes=[1])
    elif operator.type == 'SklearnNearestNeighbors':
        container.add_node(
            'Identity', topk_indices_name, operator.outputs[0].full_name,
            name=scope.get_unique_operator_name('Identity'))
        apply_abs(scope, topk_values_name, operator.outputs[1].full_name,
                  container)


register_converter('SklearnKNeighborsClassifier', convert_sklearn_knn)
register_converter('SklearnKNeighborsRegressor', convert_sklearn_knn)
register_converter('SklearnNearestNeighbors', convert_sklearn_knn)
