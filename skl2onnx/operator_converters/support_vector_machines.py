# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM
from ..common._apply_operation import apply_cast
from ..common.data_types import Int64TensorType
from ..common._registration import register_converter
from ..proto import onnx_proto


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
        if len(op.probB_) > 0:
            svm_attrs['prob_b'] = op.probB_

        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['vectors_per_class'] = op.n_support_.tolist()

        label_name = operator.outputs[0].full_name
        probability_tensor_name = operator.outputs[1].full_name

        zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
        if all(isinstance(i, (numbers.Real, bool, np.bool_))
                for i in op.classes_):
            labels = [int(i) for i in op.classes_]
            svm_attrs['classlabels_ints'] = labels
            zipmap_attrs['classlabels_int64s'] = labels
        elif all(isinstance(i, (six.text_type, six.string_types))
                 for i in op.classes_):
            labels = [str(i) for i in op.classes_]
            svm_attrs['classlabels_strings'] = labels
            zipmap_attrs['classlabels_strings'] = labels
        else:
            raise RuntimeError("Invalid class label type '%s'." % op.classes_)

        container.add_node(op_type, operator.inputs[0].full_name,
                           [label_name, probability_tensor_name],
                           op_domain='ai.onnx.ml', **svm_attrs)

    elif operator.type in ['SklearnSVR', 'SklearnNuSVR'] or isinstance(
            op, (SVR, NuSVR)):
        op_type = 'SVMRegressor'
        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['n_supports'] = len(op.support_)

        input_name = operator.input_full_names
        if type(operator.inputs[0].type) == Int64TensorType:
            cast_input_name = scope.get_unique_variable_name('cast_input')

            apply_cast(scope, operator.input_full_names, cast_input_name,
                       container, to=onnx_proto.TensorProto.FLOAT)
            input_name = cast_input_name

        container.add_node(op_type, input_name,
                           operator.output_full_names,
                           op_domain='ai.onnx.ml', **svm_attrs)
    elif (operator.type in ['SklearnOneClassSVM'] or
          isinstance(op, OneClassSVM)):
        op_type = 'SVMRegressor'
        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['n_supports'] = len(op.support_)

        input_name = operator.input_full_names
        if type(operator.inputs[0].type) == Int64TensorType:
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
register_converter('SklearnSVC', convert_sklearn_svm)
register_converter('SklearnSVR', convert_sklearn_svm)
