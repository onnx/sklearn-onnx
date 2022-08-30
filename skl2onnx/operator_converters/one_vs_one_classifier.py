# SPDX-License-Identifier: Apache-2.0

from sklearn.base import is_regressor
from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_concat, apply_identity, apply_mul, apply_reshape, apply_transpose)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_normalization
from ..common._apply_operation import (
    apply_slice, apply_sub, apply_cast, apply_abs, apply_add, apply_div)
from ..common.utils_classifier import _finalize_converter_classes
from ..common.data_types import guess_proto_type, Int64TensorType, FloatTensorType
from .._supported_operators import sklearn_operator_name_map


def convert_one_vs_one_classifier(scope: Scope, operator: Operator,
                                   container: ModelComponentContainer):

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
            if container.has_options(estimator, 'raw_scores'):
                container.add_options(
                    id(estimator), {'raw_scores': use_raw_scores})
                scope.add_options(
                    id(estimator), {'raw_scores': use_raw_scores})
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


    this_operator = scope.declare_local_operator("SklearnOVRDecisionFunction", op)
    this_operator.inputs.append(probs_names)
    ovr_name = scope.declare_local_variable(
        'ovr_output', operator.inputs[0].type.__class__())
    this_operator.outputs.append(ovr_name)

    output_name = operator.outputs[1].full_name
    container.add_node('Identity', [ovr_name.onnx_name], [output_name])

register_converter('SklearnOneVsOneClassifier',
                   convert_one_vs_one_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'output_class_labels': [False, True],
                            'raw_scores': [True, False]})



