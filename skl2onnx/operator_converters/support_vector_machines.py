# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM
from ..common._apply_operation import (
    apply_cast, apply_concat, apply_abs,
    apply_add, apply_mul, apply_div)
try:
    from ..common._apply_operation import apply_less
except ImportError:
    # onnxconverter-common is too old
    apply_less = None
from ..common.data_types import BooleanTensorType, Int64TensorType
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

    handles_ovr = False

    if operator.type in ['SklearnSVC', 'SklearnNuSVC'] or isinstance(
            op, (SVC, NuSVC)):
        op_type = 'SVMClassifier'

        if len(op.probA_) > 0:
            svm_attrs['prob_a'] = op.probA_
        else:
            handles_ovr = True
        if len(op.probB_) > 0:
            svm_attrs['prob_b'] = op.probB_

        if (hasattr(op, 'decision_function_shape') and
                op.decision_function_shape == 'ovr') and handles_ovr:
            output_name = scope.get_unique_variable_name('before_ovr')
        else:
            output_name = operator.outputs[1].full_name

        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['vectors_per_class'] = op.n_support_.tolist()

        label_name = operator.outputs[0].full_name
        probability_tensor_name = output_name

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

    elif operator.type in ['SklearnSVR', 'SklearnNuSVR'] or isinstance(
            op, (SVR, NuSVR)):
        op_type = 'SVMRegressor'
        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['n_supports'] = len(op.support_)

        input_name = operator.input_full_names
        if type(operator.inputs[0].type) in (
                BooleanTensorType, Int64TensorType):
            cast_input_name = scope.get_unique_variable_name('cast_input')

            apply_cast(scope, operator.input_full_names, cast_input_name,
                       container, to=container.proto_dtype)
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
        if type(operator.inputs[0].type) in (
                BooleanTensorType, Int64TensorType):
            cast_input_name = scope.get_unique_variable_name('cast_input')

            apply_cast(scope, operator.input_full_names, cast_input_name,
                       container, to=container.proto_dtype)
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

    if (hasattr(op, 'decision_function_shape') and
            op.decision_function_shape == 'ovr' and handles_ovr):
        # Applies _ovr_decision_function.
        # See https://github.com/scikit-learn/scikit-learn/blob/
        # master/sklearn/utils/multiclass.py#L407:
        # ::
        #     _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        #
        #     ...
        #     def _ovr_decision_function(predictions, confidences, n_classes):
        #
        #     n_samples = predictions.shape[0]
        #     votes = np.zeros((n_samples, n_classes))
        #     sum_of_confidences = np.zeros((n_samples, n_classes))
        #     k = 0
        #     for i in range(n_classes):
        #         for j in range(i + 1, n_classes):
        #             sum_of_confidences[:, i] -= confidences[:, k]
        #             sum_of_confidences[:, j] += confidences[:, k]
        #             votes[predictions[:, k] == 0, i] += 1
        #             votes[predictions[:, k] == 1, j] += 1
        #             k += 1
        #     transformed_confidences = (
        #         sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1)))
        #     return votes + transformed_confidences
        if list(op.classes_) != list(range(len(op.classes_))):
            raise RuntimeError(
                "Classes different from first n integers are not supported "
                "in SVC converter.")

        cst3 = scope.get_unique_variable_name('cst3')
        container.add_initializer(cst3, container.proto_dtype, [], [3])
        cst1 = scope.get_unique_variable_name('cst1')
        container.add_initializer(cst1, container.proto_dtype, [], [1])
        cst0 = scope.get_unique_variable_name('cst0')
        container.add_initializer(cst0, container.proto_dtype, [], [0])

        prediction = scope.get_unique_variable_name('prediction')
        if apply_less is None:
            raise RuntimeError(
                "Function apply_less is missing. "
                "onnxconverter-common is too old.")
        apply_less(scope, [output_name, cst0], prediction, container)
        iprediction = scope.get_unique_variable_name('iprediction')
        apply_cast(scope, prediction, iprediction, container,
                   to=container.proto_dtype)

        n_classes = len(op.classes_)
        sumc_name = [scope.get_unique_variable_name('svcsumc_%d' % i)
                     for i in range(n_classes)]
        vote_name = [scope.get_unique_variable_name('svcvote_%d' % i)
                     for i in range(n_classes)]
        sumc_add = {n: [] for n in sumc_name}
        vote_add = {n: [] for n in vote_name}
        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                name = scope.get_unique_operator_name(
                    'ArrayFeatureExtractor')
                ext = scope.get_unique_variable_name('Csvc_%d' % k)
                ind = scope.get_unique_variable_name('Cind_%d' % k)
                container.add_initializer(
                    ind, onnx_proto.TensorProto.INT64, [], [k])
                container.add_node(
                    'ArrayFeatureExtractor', [output_name, ind],
                    ext, op_domain='ai.onnx.ml', name=name)
                sumc_add[sumc_name[i]].append(ext)

                neg = scope.get_unique_variable_name('Cneg_%d' % k)
                name = scope.get_unique_operator_name('Neg')
                container.add_node(
                    'Neg', ext, neg, op_domain='', name=name,
                    op_version=6)
                sumc_add[sumc_name[j]].append(neg)

                # votes
                name = scope.get_unique_operator_name(
                    'ArrayFeatureExtractor')
                ext = scope.get_unique_variable_name('Vsvcv_%d' % k)
                container.add_node(
                    'ArrayFeatureExtractor', [iprediction, ind],
                    ext, op_domain='ai.onnx.ml', name=name)
                vote_add[vote_name[j]].append(ext)
                neg = scope.get_unique_variable_name('Vnegv_%d' % k)
                name = scope.get_unique_operator_name('Neg')
                container.add_node(
                    'Neg', ext, neg, op_domain='', name=name,
                    op_version=6)
                neg1 = scope.get_unique_variable_name('Vnegv1_%d' % k)
                apply_add(scope, [neg, cst1], neg1, container, broadcast=1)
                vote_add[vote_name[i]].append(neg1)

                # next
                k += 1

        for k, v in sumc_add.items():
            name = scope.get_unique_operator_name('Sum')
            container.add_node(
                'Sum', v, k, op_domain='', name=name, op_version=8)
        for k, v in vote_add.items():
            name = scope.get_unique_operator_name('Sum')
            container.add_node(
                'Sum', v, k, op_domain='', name=name, op_version=8)

        conc = scope.get_unique_variable_name('Csvcconc')
        apply_concat(scope, sumc_name, conc, container, axis=1)
        conc_vote = scope.get_unique_variable_name('Vsvcconcv')
        apply_concat(scope, vote_name, conc_vote, container, axis=1)

        conc_abs = scope.get_unique_variable_name('Cabs')
        apply_abs(scope, conc, conc_abs, container)

        conc_abs1 = scope.get_unique_variable_name('Cconc_abs1')
        apply_add(scope, [conc_abs, cst1], conc_abs1, container, broadcast=1)
        conc_abs3 = scope.get_unique_variable_name('Cconc_abs3')
        apply_mul(scope, [conc_abs1, cst3], conc_abs3, container, broadcast=1)

        final = scope.get_unique_variable_name('Csvcfinal')
        apply_div(
            scope, [conc, conc_abs3], final, container, broadcast=0)

        output_name = operator.outputs[1].full_name
        apply_add(
            scope, [conc_vote, final], output_name, container, broadcast=0)


register_converter('SklearnOneClassSVM', convert_sklearn_svm)
register_converter('SklearnSVC', convert_sklearn_svm,
                   options={'zipmap': [True, False],
                            'nocl': [True, False]})
register_converter('SklearnSVR', convert_sklearn_svm)
