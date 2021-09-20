# SPDX-License-Identifier: Apache-2.0


import numbers
import numpy as np
from scipy.sparse import isspmatrix
from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM
from ..common._apply_operation import (
    apply_cast, apply_concat, apply_abs,
    apply_add, apply_mul, apply_div)
try:
    from ..common._apply_operation import apply_less
except ImportError:
    # onnxconverter-common is too old
    apply_less = None
from ..common.data_types import (
    BooleanTensorType, Int64TensorType, guess_proto_type)
from ..common._registration import register_converter
from ..proto import onnx_proto


def convert_sklearn_svm_regressor(
        scope, operator, container,
        op_type='SVMRegressor', op_domain='ai.onnx.ml', op_version=1):
    """
    Converter for model
    `SVR <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.SVR.html>`_,
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
        coef = op.dual_coef_.ravel()
    else:
        coef = op.dual_coef_
    intercept = op.intercept_
    if isinstance(op.support_vectors_, np.ndarray):
        support_vectors = op.support_vectors_.ravel()
    else:
        support_vectors = op.support_vectors_

    svm_attrs['kernel_type'] = op.kernel.upper()
    svm_attrs['kernel_params'] = [np.float32(_) for _ in
                                  [op._gamma, op.coef0, op.degree]]
    if isspmatrix(support_vectors):
        svm_attrs['support_vectors'] = support_vectors.toarray().ravel()
    else:
        svm_attrs['support_vectors'] = support_vectors
    if isspmatrix(coef):
        svm_attrs['coefficients'] = coef.toarray().ravel()
    else:
        svm_attrs['coefficients'] = coef
    svm_attrs['rho'] = intercept.astype(np.float32)
    svm_attrs['coefficients'] = svm_attrs['coefficients'].astype(np.float32)
    svm_attrs['support_vectors'] = svm_attrs['support_vectors'].astype(
        np.float32)

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    if operator.type in ['SklearnSVR', 'SklearnNuSVR'] or isinstance(
            op, (SVR, NuSVR)):
        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['n_supports'] = len(op.support_)

        input_name = operator.input_full_names
        if type(operator.inputs[0].type) in (
                BooleanTensorType, Int64TensorType):
            cast_input_name = scope.get_unique_variable_name('cast_input')
            apply_cast(scope, operator.input_full_names, cast_input_name,
                       container, to=proto_dtype)
            input_name = cast_input_name

        svm_out = scope.get_unique_variable_name('SVM03')
        container.add_node(
            op_type, input_name, svm_out,
            op_domain=op_domain, op_version=op_version, **svm_attrs)
        apply_cast(scope, svm_out, operator.output_full_names,
                   container, to=proto_dtype)
    elif (operator.type in ['SklearnOneClassSVM'] or
          isinstance(op, OneClassSVM)):
        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['n_supports'] = len(op.support_)

        input_name = operator.input_full_names
        if type(operator.inputs[0].type) in (
                BooleanTensorType, Int64TensorType):
            cast_input_name = scope.get_unique_variable_name('cast_input')
            apply_cast(scope, operator.input_full_names, cast_input_name,
                       container, to=proto_dtype)
            input_name = cast_input_name

        svm_out0 = scope.get_unique_variable_name('SVMO1')
        container.add_node(
            op_type, input_name, svm_out0,
            op_domain=op_domain, op_version=op_version, **svm_attrs)

        svm_out = operator.output_full_names[1]
        apply_cast(scope, svm_out0, svm_out, container, to=proto_dtype)

        pred = scope.get_unique_variable_name('float_prediction')
        container.add_node('Sign', svm_out, pred, op_version=9)
        apply_cast(scope, pred, operator.output_full_names[0],
                   container, to=onnx_proto.TensorProto.INT64)
    else:
        raise ValueError("Unknown support vector machine model type found "
                         "'{0}'.".format(operator.type))


def convert_sklearn_svm_classifier(
        scope, operator, container,
        op_type='SVMClassifier', op_domain='ai.onnx.ml', op_version=1):
    """
    Converter for model
    `SVC <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.SVC.html>`_,
    `NuSVC <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.NuSVC.html>`_.
    The converted model in ONNX produces the same results as the
    original model except when probability=False:
    *onnxruntime* and *scikit-learn* do not return the same raw
    scores. *scikit-learn* returns aggregated scores
    as a *matrix[N, C]* coming from `_ovr_decision_function
    <https://github.com/scikit-learn/scikit-learn/blob/master/
    sklearn/utils/multiclass.py#L402>`_. *onnxruntime* returns
    the raw score from *svm* algorithm as a *matrix[N, (C(C-1)/2]*.
    """
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    svm_attrs = {'name': scope.get_unique_operator_name('SVMc')}
    op = operator.raw_operator
    if isinstance(op.dual_coef_, np.ndarray):
        coef = op.dual_coef_.ravel()
    else:
        coef = op.dual_coef_
    intercept = op.intercept_
    if isinstance(op.support_vectors_, np.ndarray):
        support_vectors = op.support_vectors_.ravel()
    elif isspmatrix(op.support_vectors_):
        support_vectors = op.support_vectors_.toarray().ravel()
    else:
        support_vectors = op.support_vectors_

    svm_attrs['kernel_type'] = op.kernel.upper()
    svm_attrs['kernel_params'] = [float(_) for _ in
                                  [op._gamma, op.coef0, op.degree]]
    svm_attrs['support_vectors'] = support_vectors

    if (operator.type in ['SklearnSVC', 'SklearnNuSVC'] or isinstance(
            op, (SVC, NuSVC))) and len(op.classes_) == 2:
        if isspmatrix(coef):
            coef_dense = coef.toarray().ravel()
            svm_attrs['coefficients'] = -coef_dense
        else:
            svm_attrs['coefficients'] = -coef
        svm_attrs['rho'] = -intercept
    else:
        if isspmatrix(coef):
            svm_attrs['coefficients'] = coef.todense()
        else:
            svm_attrs['coefficients'] = coef
        svm_attrs['rho'] = intercept

    handles_ovr = False
    svm_attrs['coefficients'] = svm_attrs['coefficients'].astype(np.float32)
    svm_attrs['support_vectors'] = svm_attrs['support_vectors'].astype(
        np.float32)
    svm_attrs['rho'] = svm_attrs['rho'].astype(np.float32)

    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']

    if operator.type in ['SklearnSVC', 'SklearnNuSVC'] or isinstance(
            op, (SVC, NuSVC)):
        if len(op.probA_) > 0:
            svm_attrs['prob_a'] = op.probA_.astype(np.float32)
        else:
            handles_ovr = True
        if len(op.probB_) > 0:
            svm_attrs['prob_b'] = op.probB_.astype(np.float32)

        if (hasattr(op, 'decision_function_shape') and
                op.decision_function_shape == 'ovr' and handles_ovr and
                len(op.classes_) > 2):
            output_name = scope.get_unique_variable_name('before_ovr')
        elif len(op.classes_) == 2 and use_raw_scores:
            output_name = scope.get_unique_variable_name('raw_scores')
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
        elif all(isinstance(i, str) for i in op.classes_):
            labels = [str(i) for i in op.classes_]
            svm_attrs['classlabels_strings'] = labels
        else:
            raise RuntimeError("Invalid class label type '%s'." % op.classes_)

        svm_out = scope.get_unique_variable_name('SVM02')
        container.add_node(
            op_type, operator.inputs[0].full_name,
            [label_name, svm_out],
            op_domain=op_domain, op_version=op_version, **svm_attrs)
        apply_cast(scope, svm_out, probability_tensor_name,
                   container, to=proto_dtype)
        if len(op.classes_) == 2 and use_raw_scores:
            minus_one = scope.get_unique_variable_name('minus_one')
            container.add_initializer(minus_one, proto_dtype, [], [-1])
            container.add_node(
                'Mul', [output_name, minus_one], operator.outputs[1].full_name,
                name=scope.get_unique_operator_name('MulRawScores'))
    else:
        raise ValueError("Unknown support vector machine model type found "
                         "'{0}'.".format(operator.type))

    if (hasattr(op, 'decision_function_shape') and
            op.decision_function_shape == 'ovr' and handles_ovr and
            len(op.classes_) > 2):
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

        cst3 = scope.get_unique_variable_name('cst3')
        container.add_initializer(cst3, proto_dtype, [], [3])
        cst1 = scope.get_unique_variable_name('cst1')
        container.add_initializer(cst1, proto_dtype, [], [1])
        cst0 = scope.get_unique_variable_name('cst0')
        container.add_initializer(cst0, proto_dtype, [], [0])

        prediction = scope.get_unique_variable_name('prediction')
        if apply_less is None:
            raise RuntimeError(
                "Function apply_less is missing. "
                "onnxconverter-common is too old.")
        proto_dtype = guess_proto_type(operator.inputs[0].type)
        if proto_dtype != onnx_proto.TensorProto.DOUBLE:
            proto_dtype = onnx_proto.TensorProto.FLOAT
        apply_less(scope, [output_name, cst0], prediction, container)
        iprediction = scope.get_unique_variable_name('iprediction')
        apply_cast(scope, prediction, iprediction, container,
                   to=proto_dtype)

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
                apply_add(scope, [neg, cst1], neg1, container, broadcast=1,
                          operator_name='AddCl_%d_%d' % (i, j))
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
        apply_add(scope, [conc_abs, cst1], conc_abs1, container, broadcast=1,
                  operator_name='AddF0')
        conc_abs3 = scope.get_unique_variable_name('Cconc_abs3')
        apply_mul(scope, [conc_abs1, cst3], conc_abs3, container, broadcast=1)

        final = scope.get_unique_variable_name('Csvcfinal')
        apply_div(
            scope, [conc, conc_abs3], final, container, broadcast=0)

        output_name = operator.outputs[1].full_name
        apply_add(
            scope, [conc_vote, final], output_name, container, broadcast=0,
            operator_name='AddF1')


register_converter('SklearnOneClassSVM', convert_sklearn_svm_regressor)
register_converter('SklearnSVC', convert_sklearn_svm_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'raw_scores': [True, False]})
register_converter('SklearnSVR', convert_sklearn_svm_regressor)
