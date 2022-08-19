# SPDX-License-Identifier: Apache-2.0

from sklearn.base import is_regressor
from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_concat, apply_identity, apply_mul, apply_reshape)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_normalization
from ..common._apply_operation import (
    apply_slice, apply_sub, apply_cast, apply_abs, apply_add, apply_div)
from ..common.utils_classifier import _finalize_converter_classes
from ..common.data_types import guess_proto_type, Int64TensorType
from .._supported_operators import sklearn_operator_name_map


def convert_one_vs_one_classifier(scope: Scope, operator: Operator,
                                   container: ModelComponentContainer):
    """
    Converts a *OneVsOneClassifier* into *ONNX* format.
    """
    if scope.get_options(operator.raw_operator, dict(nocl=False))['nocl']:
        raise RuntimeError(
            "Option 'nocl' is not implemented for operator '{}'.".format(
                operator.raw_operator.__class__.__name__))
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator
    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']
    probs_names = []
    for i, estimator in enumerate(op.estimators_):
        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(
            op_type, raw_model=estimator)
        this_operator.inputs = operator.inputs

        if is_regressor(estimator):
            score_name = scope.declare_local_variable(
                'score_%d' % i, operator.inputs[0].type.__class__())
            this_operator.outputs.append(score_name)

            if hasattr(estimator, 'coef_') and len(estimator.coef_.shape) == 2:
                raise RuntimeError("OneVsRestClassifier accepts "
                                   "regressor with only one target.")
            p1 = score_name.onnx_name
        else:
            label_name = scope.declare_local_variable(
                'label_%d' % i, Int64TensorType())
            prob_name = scope.declare_local_variable(
                'proba_%d' % i, operator.inputs[0].type.__class__())
            this_operator.outputs.append(label_name)
            this_operator.outputs.append(prob_name)

            # gets the probability for the class 1
            p1 = scope.get_unique_variable_name('probY_%d' % i)
            apply_slice(scope, prob_name.onnx_name, p1, container, starts=[1],
                        ends=[2], axes=[1],
                        operator_name=scope.get_unique_operator_name('Slice'))

        probs_names.append(p1)


    conc_name = scope.get_unique_variable_name('concatenated')
    apply_concat(scope, probs_names, conc_name, container, axis=1)
    if len(op.estimators_) == 1:
        zeroth_col_name = scope.get_unique_variable_name('zeroth_col')
        merged_prob_name = scope.get_unique_variable_name('merged_prob')
        unit_float_tensor_name = scope.get_unique_variable_name(
            'unit_float_tensor')
        if use_raw_scores:
            container.add_initializer(
                unit_float_tensor_name, proto_dtype, [], [-1.0])
            apply_mul(scope, [unit_float_tensor_name, conc_name],
                        zeroth_col_name, container, broadcast=1)
        else:
            container.add_initializer(
                unit_float_tensor_name, proto_dtype, [], [1.0])
            apply_sub(scope, [unit_float_tensor_name, conc_name],
                        zeroth_col_name, container, broadcast=1)
        apply_concat(scope, [zeroth_col_name, conc_name],
                        merged_prob_name, container, axis=1)
        conc_name = merged_prob_name

    if use_raw_scores:
        apply_identity(scope, conc_name,
                        operator.outputs[1].full_name, container)
    else:
        # normalizes the outputs
        apply_normalization(
            scope, conc_name, operator.outputs[1].full_name,
            container, axis=1, p=1)

    # extracts the labels
    label_name = scope.get_unique_variable_name('label_name')
    container.add_node('ArgMax', conc_name, label_name,
                        name=scope.get_unique_operator_name('ArgMax'),
                        axis=1)

    _finalize_converter_classes(scope, label_name,
                                operator.outputs[0].full_name, container,
                                op.classes_, proto_dtype)


register_converter('SklearnOneVsOneClassifier',
                   convert_one_vs_one_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'output_class_labels': [False, True],
                            'raw_scores': [True, False]})
