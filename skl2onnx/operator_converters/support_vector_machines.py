# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from sklearn.svm import SVC, NuSVC, OneClassSVM
from ..common._apply_operation import (
    apply_add,
    apply_cast,
    apply_exp,
    apply_mul,
    apply_pow,
    apply_reshape,
    apply_sub,
    apply_tanh,
)
from ..common.data_types import BooleanTensorType, Int64TensorType
from ..common._registration import register_converter
from ..proto import onnx_proto


def _kernel_linear(scope, container, model, input_name):
    support_vectors_name = scope.get_unique_variable_name('support_vectors')
    matmul_result_name = scope.get_unique_variable_name('matmul_result')

    container.add_initializer(
        support_vectors_name, container.proto_dtype,
        model.support_vectors_.T.shape, model.support_vectors_.T.ravel())

    container.add_node(
        'MatMul', [input_name, support_vectors_name],
        matmul_result_name, name=scope.get_unique_operator_name('MatMul'))
    return matmul_result_name


def _kernel_poly_sigmoid(scope, container, model, input_name):
    gamma_name = scope.get_unique_variable_name('gamma')
    coef0_name = scope.get_unique_variable_name('coef0')
    degree_name = scope.get_unique_variable_name('degree')
    prod_result_name = scope.get_unique_variable_name('product_result')
    prod_coef0_sum_name = scope.get_unique_variable_name('prod_coef0_sum')
    kernel_result_name = scope.get_unique_variable_name('kernel_result')
    support_vectors_name = scope.get_unique_variable_name('support_vectors')
    matmul_result_name = scope.get_unique_variable_name('matmul_result')

    container.add_initializer(
        support_vectors_name, container.proto_dtype,
        model.support_vectors_.T.shape, model.support_vectors_.T.ravel())
    container.add_initializer(
        gamma_name, container.proto_dtype,
        [], [model._gamma])
    container.add_initializer(
        coef0_name, container.proto_dtype,
        [], [model.coef0])
    container.add_initializer(
        degree_name, container.proto_dtype,
        [], [model.degree])

    container.add_node(
        'MatMul', [input_name, support_vectors_name],
        matmul_result_name, name=scope.get_unique_operator_name('MatMul'))
    apply_mul(scope, [matmul_result_name, gamma_name],
              prod_result_name, container, broadcast=1)
    apply_add(scope, [prod_result_name, coef0_name],
              prod_coef0_sum_name, container, broadcast=1)
    if model.kernel == 'poly':
        apply_pow(scope, [prod_coef0_sum_name, degree_name],
                  kernel_result_name, container, broadcast=1)
    else:
        apply_tanh(scope, prod_coef0_sum_name, kernel_result_name,
                   container)
    return kernel_result_name


def _kernel_rbf(scope, container, model, input_name):
    gamma_name = scope.get_unique_variable_name('gamma')
    support_vectors_name = scope.get_unique_variable_name('support_vectors')
    sub_result_name = scope.get_unique_variable_name('sub_result')
    squared_result_name = scope.get_unique_variable_name('squared_result')
    reshaped_input_name = scope.get_unique_variable_name('reshaped_result')
    exp_result_name = scope.get_unique_variable_name('exp_result')
    reduced_square_sum_result_name = scope.get_unique_variable_name(
        'reduced_square_sum_result')
    gamma_reduced_sum_result_name = scope.get_unique_variable_name(
        'gamma_reduced_sum_result')

    container.add_initializer(
        gamma_name, container.proto_dtype,
        [], [-model._gamma])
    container.add_initializer(
        support_vectors_name, container.proto_dtype,
        (1, ) + model.support_vectors_.shape,
        model.support_vectors_.ravel())

    apply_reshape(
        scope, input_name, reshaped_input_name, container,
        desired_shape=(-1, 1, model.support_vectors_.shape[1]))
    apply_sub(scope, [reshaped_input_name, support_vectors_name],
              sub_result_name, container, broadcast=1)
    apply_mul(scope, [sub_result_name, sub_result_name],
              squared_result_name, container, broadcast=1)
    container.add_node(
        'ReduceSum', squared_result_name, reduced_square_sum_result_name,
        axes=[2], name=scope.get_unique_operator_name('ReduceSum'),
        keepdims=0)
    apply_mul(scope, [reduced_square_sum_result_name, gamma_name],
              gamma_reduced_sum_result_name, container, broadcast=1)
    apply_exp(scope, gamma_reduced_sum_result_name, exp_result_name,
              container)
    return exp_result_name


def convert_sklearn_svr(scope, operator, container):
    """
    Converter for NuSVR and SVR. Here, we call the respective kernel
    functions whose return value is mutipled to dual_coef, reduced
    along axis 1 and added to the intercept value.
    """
    model = operator.raw_operator
    dual_coef_name = scope.get_unique_variable_name('dual_coef')
    intercept_name = scope.get_unique_variable_name('intercept')
    mul_result_name = scope.get_unique_variable_name('mul_result')
    reduce_sum_result_name = scope.get_unique_variable_name(
        'reduce_sum_result')
    kernel_mapper = {
        'linear': _kernel_linear,
        'poly': _kernel_poly_sigmoid,
        'rbf': _kernel_rbf,
        'sigmoid': _kernel_poly_sigmoid,
    }
    if model.kernel not in kernel_mapper:
        raise NotImplementedError(
            "kernel {} not supported yet. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues".format(model.kernel))

    container.add_initializer(
        dual_coef_name, container.proto_dtype,
        model.dual_coef_.shape, model.dual_coef_.ravel())
    container.add_initializer(
        intercept_name, container.proto_dtype,
        model.intercept_.shape, model.intercept_)

    input_name = operator.inputs[0].full_name
    if type(operator.inputs[0].type) in (
            BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.inputs[0].full_name, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = cast_input_name
    kernel_result = kernel_mapper[model.kernel](
        scope, container, model, input_name)
    apply_mul(scope, [kernel_result, dual_coef_name],
              mul_result_name, container, broadcast=1)
    container.add_node(
        'ReduceSum', mul_result_name, reduce_sum_result_name,
        axes=[1], name=scope.get_unique_operator_name('ReduceSum'))
    apply_add(scope, [reduce_sum_result_name, intercept_name],
              operator.outputs[0].full_name, container, broadcast=1)


def convert_sklearn_svm(scope, operator, container):
    """
    Converter for model
    `SVC <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.SVC.html>`_,
    `SVR <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.SVR.html>`_,
    `NuSVC <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.NuSVC.html>`_,
    `NuSVR <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.NuSVR.html>`_,
    `OneClassSVM <https://scikit-learn.org/stable/
    modules/generated/sklearn.svm.OneClassSVM.html>`_.
    The converted model in ONNX produces the same results as the
    original model except when probability=False:
    *onnxruntime* and *scikit-learn* do not return the same raw
    scores. *scikit-learn* returns aggregated scores
    as a *matrix[N, C]* coming from `_ovr_decision_function
    <https://github.com/scikit-learn/scikit-learn/blob/master/
    sklearn/utils/multiclass.py#L402>`_. *onnxruntime* returns
    the raw score from *svm* algorithm as a *matrix[N, (C(C-1)/2]*.
    """
    svm_attrs = {'name': scope.get_unique_operator_name('SVM')}
    op = operator.raw_operator
    if isinstance(op.dual_coef_, np.ndarray):
        coef = op.dual_coef_.ravel().tolist()
    else:
        coef = op.dual_coef_
    intercept = op.intercept_
    if isinstance(op.support_vectors_, np.ndarray):
        support_vectors = op.support_vectors_.ravel().tolist()
    else:
        support_vectors = op.support_vectors_

    svm_attrs['kernel_type'] = op.kernel.upper()
    svm_attrs['kernel_params'] = [float(_) for _ in
                                  [op._gamma, op.coef0, op.degree]]
    svm_attrs['support_vectors'] = support_vectors

    if (operator.type in ['SklearnSVC', 'SklearnNuSVC'] or isinstance(
            op, (SVC, NuSVC))) and len(op.classes_) == 2:
        svm_attrs['coefficients'] = [-v for v in coef]
        svm_attrs['rho'] = [-v for v in intercept]
    else:
        svm_attrs['coefficients'] = coef
        svm_attrs['rho'] = intercept

    if operator.type in ['SklearnSVC', 'SklearnNuSVC'] or isinstance(
            op, (SVC, NuSVC)):
        op_type = 'SVMClassifier'

        if len(op.probA_) > 0:
            svm_attrs['prob_a'] = op.probA_
        elif op.decision_function_shape == 'ovr':
            raise RuntimeError(
                "decision_function_shape == 'ovr' is not supported. "
                "Please raise an issue if you need to be implemented.")
        if len(op.probB_) > 0:
            svm_attrs['prob_b'] = op.probB_

        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['vectors_per_class'] = op.n_support_.tolist()

        label_name = operator.outputs[0].full_name
        probability_tensor_name = operator.outputs[1].full_name

        if all(isinstance(i, (numbers.Real, bool, np.bool_))
                for i in op.classes_):
            labels = [int(i) for i in op.classes_]
            svm_attrs['classlabels_ints'] = labels
        elif all(isinstance(i, (six.text_type, six.string_types))
                 for i in op.classes_):
            labels = [str(i) for i in op.classes_]
            svm_attrs['classlabels_strings'] = labels
        else:
            raise RuntimeError("Invalid class label type '%s'." % op.classes_)

        container.add_node(op_type, operator.inputs[0].full_name,
                           [label_name, probability_tensor_name],
                           op_domain='ai.onnx.ml', **svm_attrs)

    elif (operator.type in ['SklearnOneClassSVM'] or
          isinstance(op, OneClassSVM)):
        op_type = 'SVMRegressor'
        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['n_supports'] = len(op.support_)

        input_name = operator.input_full_names
        if type(operator.inputs[0].type) in (
                BooleanTensorType, Int64TensorType):
            cast_input_name = scope.get_unique_variable_name('cast_input')

            apply_cast(scope, operator.input_full_names, cast_input_name,
                       container, to=onnx_proto.TensorProto.FLOAT)
            input_name = cast_input_name

        svm_out = operator.output_full_names[1]
        container.add_node(op_type, input_name, svm_out,
                           op_domain='ai.onnx.ml', **svm_attrs)

        pred = scope.get_unique_variable_name('float_prediction')
        container.add_node('Sign', svm_out, pred, op_version=9)
        apply_cast(scope, pred, operator.output_full_names[0],
                   container, to=onnx_proto.TensorProto.INT64)
    else:
        raise ValueError("Unknown support vector machine model type found "
                         "'{0}'.".format(operator.type))


register_converter('SklearnOneClassSVM', convert_sklearn_svm)
register_converter('SklearnSVC', convert_sklearn_svm,
                   options={'zipmap': [True, False],
                            'nocl': [True, False]})
register_converter('SklearnSVR', convert_sklearn_svr)
