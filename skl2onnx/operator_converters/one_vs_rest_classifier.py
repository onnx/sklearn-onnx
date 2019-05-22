# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from sklearn.base import is_regressor
from ..proto import onnx_proto
from ..common._apply_operation import apply_concat
from ..common._topology import FloatTensorType
from ..common._registration import register_converter
from ..common._apply_operation import apply_normalization
from ..common._apply_operation import apply_slice, apply_sub, apply_clip
from ..common.utils_classifier import _finalize_converter_classes
from .._supported_operators import sklearn_operator_name_map


def convert_one_vs_rest_classifier(scope, operator, container):
    """
    Converts a *OneVsRestClassifier* into *ONNX* format.
    """
    op = operator.raw_operator
    probs_names = []
    for i, estimator in enumerate(op.estimators_):
        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type)
        this_operator.raw_operator = estimator
        this_operator.inputs = operator.inputs

        if is_regressor(estimator):
            score_name = scope.declare_local_variable('score_%d' % i,
                                                      FloatTensorType())
            this_operator.outputs.append(score_name)

            if len(estimator.coef_.shape) == 2:
                raise RuntimeError("OneVsRestClassifier accepts "
                                   "regressor with only one target.")
            p1 = score_name.raw_name
        else:
            label_name = scope.declare_local_variable('label_%d' % i)
            prob_name = scope.declare_local_variable('proba_%d' % i,
                                                     FloatTensorType())
            this_operator.outputs.append(label_name)
            this_operator.outputs.append(prob_name)

            # gets the probability for the class 1
            p1 = scope.get_unique_variable_name('probY_%d' % i)
            apply_slice(scope, prob_name.raw_name, p1, container, starts=[1],
                        ends=[2], axes=[1],
                        operator_name=scope.get_unique_operator_name('Slice'))

        probs_names.append(p1)

    if op.multilabel_:
        # concatenates outputs
        conc_name = operator.outputs[1].full_name
        apply_concat(scope, probs_names, conc_name, container, axis=1)

        # builds the labels (matrix with integer)
        # scikit-learn may use probabilities or raw score
        # but ONNX converters only uses probabilities.
        # https://github.com/scikit-learn/scikit-learn/sklearn/
        # multiclass.py#L290
        # Raw score would mean: scores = conc_name.
        thresh_name = scope.get_unique_variable_name('thresh')
        container.add_initializer(
            thresh_name, onnx_proto.TensorProto.FLOAT,
            [1, len(op.classes_)], [.5] * len(op.classes_))
        scores = scope.get_unique_variable_name('threshed')
        apply_sub(scope, [conc_name, thresh_name], scores, container)

        # sign
        signed_input = scope.get_unique_variable_name('signed')
        container.add_node('Sign', [scores], [signed_input],
                           name=scope.get_unique_operator_name('Sign'))
        # clip
        apply_clip(scope, signed_input, operator.outputs[0].full_name,
                   container, operator_name=None, max=None, min=0)
    else:
        # concatenates outputs
        conc_name = scope.get_unique_variable_name('concatenated')
        apply_concat(scope, probs_names, conc_name, container, axis=1)

        # normalizes the outputs
        apply_normalization(scope, conc_name, operator.outputs[1].full_name,
                            container, axis=1, p=1)

        # extracts the labels
        label_name = scope.get_unique_variable_name('label_name')
        container.add_node('ArgMax', conc_name, label_name,
                           name=scope.get_unique_operator_name('ArgMax'),
                           axis=1)

        _finalize_converter_classes(scope, label_name,
                                    operator.outputs[0].full_name, container,
                                    op.classes_)


register_converter('SklearnOneVsRestClassifier',
                   convert_one_vs_rest_classifier)
