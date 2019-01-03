# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_add, apply_cast, apply_div, apply_exp, apply_mul, apply_reshape
from ..common._topology import FloatTensorType
from ..common._registration import get_converter, register_converter
from .._parse import sklearn_operator_name_map
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import numpy as np


def convert_calibrated_classifier_base_estimator(scope, operator, container, model):
    # Computational graph:
    #
    # In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names
    # from the official ONNX spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of instances
    # N: Number of features
    # C: Number of classes
    # CLASSIFIERCONVERTER: classifier converter corresponding to the op_type
    # a: slope in sigmoid model
    # b: intercept in sigmoid model
    # k: variable in the range [0, C)
    # input: input
    # class_prob_tensor: tensor with class probabilities (output of this function)
    #
    # Graph:
    #
    #   input [M, N] -> CLASSIFIERCONVERTER -> label [M]
    #                          |
    #                          V
    #                    probability_tensor [M, C]
    #                          |
    #         .----------------'---------.
    #         |                          |
    #         V                          V
    # ARRAYFEATUREEXTRACTOR <- k [1] -> ARRAYFEATUREEXTRACTOR
    #         |                          |
    #         V                          V
    #  transposed_df_col[M, 1]        transposed_df_col[M, 1]
    #         |--------------------------|---------------------------.---------------------------.
    #         |                          |                           |                           |
    #         |if model.method='sigmoid' |                           | if model.method='isotonic'|
    #         |                          |                           |                           |
    #         V                          V                           | if out_of_bounds='clip'   |
    #        MUL <-------- a -------->  MUL                          V                           V
    #         |                          |                          CLIP        ...             CLIP
    #         V                          V                           |                           |
    #   a_df_prod [M, 1]  ...    a_df_prod [M, 1]                    V                           V
    #         |                          |                        clipped_df [M, 1]  ...  clipped_df [M, 1]
    #         V                          V                           |                           |
    #       ADD <--------- b ---------> ADD                          '----------.----------------'
    #         |                          |                                      |
    #         V                          V                                      |
    #  exp_parameter [M, 1] ...   exp_parameter [M, 1]                          |
    #         |                          |                                      |
    #         V                          V                                      |
    #        EXP        ...             EXP                                     |
    #         |                          |                                      |
    #         V                          V                                      |
    #  exp_result [M, 1]  ...    exp_result [M, 1]                              |
    #         |                          |                                      |
    #         V                          V                                      |
    #       ADD <------- unity -------> ADD                                     |
    #         |                          |                                      |
    #         V                          V                                      |
    #  denominator [M, 1]  ...   denominator [M, 1]                             |
    #         |                          |                                      |
    #         V                          V                                      |
    #        DIV <------- unity ------> DIV                                     |
    #         |                          |                                      |
    #         V                          V                                      |
    # sigmoid_predict_result [M, 1] ... sigmoid_predict_result [M, 1]           |
    #         |                          |                                      |
    #         '-----.--------------------'                                      |
    #               |-----------------------------------------------------------'
    #               |
    #               V
    #            CONCAT -> concatenated_prob [M, C]
    #                          |
    #        if  C = 2         |  if C != 2
    #      .-------------------'-----------------------------------------.-------------------.
    #      |                                                             |                   |
    #      V                                                             |                   V
    # ARRAYFEATUREEXTRACTOR <- col_number [1]                            |             REDUCESUM
    #                   |                                                |                   |
    #                   '--------------------------------.               V                   V
    # unit_float_tensor [1] -> SUB <- first_col [M, 1] <-'              DIV <----------- reduced_prob [M]
    #                           |                                        |
    #                           V                                        |
    #                         CONCAT                                     |
    #                           |                                        |
    #                           V                                        |
    #                        class_prob_tensor [M, C] <------------------'

    base_model = model.base_estimator
    op_type = sklearn_operator_name_map[type(base_model)]
    n_classes = len(model.classes_)
    prob_name = [None] * n_classes

    this_operator = scope.declare_local_operator(op_type)
    this_operator.raw_operator = base_model
    this_operator.inputs = operator.inputs
    label_name = scope.declare_local_variable('label')
    df_name = scope.declare_local_variable('probability_tensor', FloatTensorType())
    this_operator.outputs.append(label_name)
    this_operator.outputs.append(df_name)

    concatenated_prob_name = scope.get_unique_variable_name('concatenated_prob')

    for k in range(n_classes):
        if n_classes == 2:
            k += 1

        k_name = scope.get_unique_variable_name('k')
        df_col_name = scope.get_unique_variable_name('transposed_df_col')
        prob_name[k] = scope.get_unique_variable_name('prob_{}'.format(k))

        container.add_initializer(k_name, onnx_proto.TensorProto.INT64, [], [k])

        container.add_node('ArrayFeatureExtractor', [df_name.full_name, k_name],
                           df_col_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                           op_domain='ai.onnx.ml')
        T = df_col_name
        if model.method == 'sigmoid':
            a_name = scope.get_unique_variable_name('a')
            b_name = scope.get_unique_variable_name('b')
            a_df_prod_name = scope.get_unique_variable_name('a_df_prod')
            exp_parameter_name = scope.get_unique_variable_name('exp_parameter')
            exp_result_name = scope.get_unique_variable_name('exp_result')
            unity_name = scope.get_unique_variable_name('unity')
            denominator_name = scope.get_unique_variable_name('denominator')
            sigmoid_predict_result_name = scope.get_unique_variable_name('sigmoid_predict_result')
            
            container.add_initializer(a_name, onnx_proto.TensorProto.FLOAT, [], [model.calibrators_[k].a_])
            container.add_initializer(b_name, onnx_proto.TensorProto.FLOAT, [], [model.calibrators_[k].b_])
            container.add_initializer(unity_name, onnx_proto.TensorProto.FLOAT, [], [1])

            apply_mul(scope, [a_name, df_col_name], a_df_prod_name, container, broadcast=0)
            apply_add(scope, [a_df_prod_name, b_name], exp_parameter_name, container, broadcast=0)
            apply_exp(scope, exp_parameter_name, exp_result_name, container)
            apply_add(scope, [unity_name, exp_result_name], denominator_name, container, broadcast=0)
            apply_div(scope, [unity_name, denominator_name], sigmoid_predict_result_name, container, broadcast=0)
            T = sigmoid_predict_result_name
        else: # isotonic method
            if model.calibrators_[k].out_of_bounds == 'clip':
                clipped_df_name = scope.get_unique_variable_name('clipped_df')

                container.add_node('Clip', df_col_name,
                                    clipped_df_name, name=scope.get_unique_operator_name('Clip'),
                                    min=model.calibrators_[k].X_min_, max=model.calibrators_[k].X_max_)
                T = clipped_df_name

        prob_name[k] = T

    container.add_node('Concat', [p for p in prob_name],
                       concatenated_prob_name, name=scope.get_unique_operator_name('Concat'), axis=1)
    if n_classes == 2:
        col_index_name = scope.get_unique_variable_name('col_index')
        zeroth_col_name = scope.get_unique_variable_name('zeroth_col')
        first_col_name = scope.get_unique_variable_name('first_col')
        merged_prob_name = scope.get_unique_variable_name('merged_prob')
        unit_float_tensor_name = scope.get_unique_variable_name('unit_float_tensor')

        container.add_initializer(col_index_name, onnx_proto.TensorProto.INT32, [], [1])
        container.add_initializer(unit_float_tensor_name, onnx_proto.TensorProto.FLOAT, [], [1.0])

        container.add_node('ArrayFeatureExtractor', [concatenated_prob_name, col_index_name],
                           first_col_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                           op_domain='ai.onnx.ml')
        apply_sub(scope, [unit_float_tensor_name, first_col_name], zeroth_col_name, container, broadcast=1)
        container.add_node('Concat', [zeroth_col_name, first_col_name],
                           merged_prob_name, name=scope.get_unique_operator_name('Concat'), axis=1)
        class_prob_tensor_name = merged_prob_name
    else:
        reduced_prob_name = scope.get_unique_variable_name('reduced_prob')
        calc_prob_name = scope.get_unique_variable_name('calc_prob')

        container.add_node('ReduceSum', concatenated_prob_name,
                   reduced_prob_name, name=scope.get_unique_operator_name('ReduceSum'), axes=[1])
        apply_div(scope, [concatenated_prob_name, reduced_prob_name], calc_prob_name, container, broadcast=1)
        class_prob_tensor_name = calc_prob_name
    return class_prob_tensor_name


def convert_sklearn_calibrated_classifier_cv(scope, operator, container):
    # Computational graph:
    #
    # In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names
    # from the official ONNX spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of instances
    # N: Number of features
    # C: Number of classes
    # CONVERT_BASE_ESTIMATOR: base estimator convert function defined above
    # clf_length: number of calibrated classifiers
    # input: input
    # output: output
    # class_prob: class probabilities
    #
    # Graph:
    #
    #                         input [M, N]
    #                               |
    #           .-------------------|--------------------------.
    #           |                   |                          |
    #           V                   V                          V
    # CONVERT_BASE_ESTIMATOR  CONVERT_BASE_ESTIMATOR ... CONVERT_BASE_ESTIMATOR
    #           |                   |                          |
    #           V                   V                          V
    #  prob_scores_0 [M, C]    prob_scores_1 [M, C] ... prob_scores_(clf_length-1) [M, C]
    #           |                   |                          |
    #           '-------------------|--------------------------'
    #                               V
    #       add_result [M, C] <--- SUM
    #           |
    #           '--> DIV <- clf_length [1]
    #                 |
    #                 V
    #            class_prob [M, C] -> ARGMAX -> argmax_output [M, 1]
    #                                                   |
    #             classes -> ARRAYFEATUREEXTRACTOR  <---'
    #                               |
    #                               V
    #                            output [1]

    op = operator.raw_operator
    classes = op.classes_
    output_shape = [-1,]

    if np.issubdtype(op.classes_.dtype, np.floating):
        class_type = onnx_proto.TensorProto.INT32
        classes = np.array(list(map(lambda x: int(x), classes)))
    elif np.issubdtype(op.classes_.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode('utf-8') for s in classes])

    zero_constant = np.zeros((1, len(op.classes_)))
    clf_length = len(op.calibrated_classifiers_)
    prob_scores_name = []

    clf_length_name = scope.get_unique_variable_name('clf_length')
    classes_name = scope.get_unique_variable_name('classes')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')
    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    array_feature_extractor_result_name = scope.get_unique_variable_name('array_feature_extractor_result')
    add_result_name = scope.get_unique_variable_name('add_result')

    container.add_initializer(classes_name, class_type, classes.shape, classes)
    container.add_initializer(clf_length_name, onnx_proto.TensorProto.FLOAT,
                              [], [clf_length])

    for clf in op.calibrated_classifiers_:
        prob_scores_name.append(convert_calibrated_classifier_base_estimator(scope, operator, container, clf))

    container.add_node('Sum', [s for s in prob_scores_name],
                       add_result_name, name=scope.get_unique_operator_name('Sum'), op_version=7)
    apply_div(scope, [add_result_name, clf_length_name], operator.outputs[1].full_name, container, broadcast=1)
    class_prob_name = operator.outputs[1].full_name
    container.add_node('ArgMax', class_prob_name,
                       argmax_output_name, name=scope.get_unique_operator_name('ArgMax'), axis=1)
    container.add_node('ArrayFeatureExtractor', [classes_name, argmax_output_name],
                       array_feature_extractor_result_name,
                       name=scope.get_unique_operator_name('ArrayFeatureExtractor'), op_domain='ai.onnx.ml')

    if class_type == onnx_proto.TensorProto.INT32: # int labels
        apply_reshape(scope, array_feature_extractor_result_name, reshaped_result_name, container,
                      desired_shape=output_shape)
        apply_cast(scope, reshaped_result_name, operator.outputs[0].full_name, container,
                   to=onnx_proto.TensorProto.INT64)
    else: # string labels
        apply_reshape(scope, array_feature_extractor_result_name, operator.outputs[0].full_name, container,
                      desired_shape=output_shape)


register_converter('SklearnCalibratedClassifierCV', convert_sklearn_calibrated_classifier_cv)
