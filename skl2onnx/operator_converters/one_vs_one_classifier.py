# SPDX-License-Identifier: Apache-2.0

from sklearn.base import is_regressor
from sklearn.svm import LinearSVC
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
from ..common.data_types import (
    guess_proto_type, Int64TensorType, FloatTensorType)
from .._supported_operators import sklearn_operator_name_map


def convert_one_vs_one_classifier(scope: Scope, operator: Operator,
                                  container: ModelComponentContainer):

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator
    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']

    # shape to use to reshape score
    cst0 = scope.get_unique_variable_name('cst0')
    container.add_initializer(cst0, onnx_proto.TensorProto.INT64, [1], [0])
    cst1 = scope.get_unique_variable_name('cst1')
    container.add_initializer(cst1, onnx_proto.TensorProto.INT64, [1], [1])
    cst2 = scope.get_unique_variable_name('cst2')
    container.add_initializer(cst2, onnx_proto.TensorProto.INT64, [1], [2])
    shape = scope.get_unique_variable_name('shape')
    container.add_node('Shape', [operator.inputs[0].full_name], [shape])
    first_dim = scope.get_unique_variable_name('dim')
    container.add_node('Gather', [shape, cst0], [first_dim])
    cst_1 = scope.get_unique_variable_name('cst_1')
    container.add_initializer(cst_1, onnx_proto.TensorProto.INT64, [1], [-1])
    prob_shape = scope.get_unique_variable_name('shape')
    apply_concat(scope, [first_dim, cst_1], prob_shape, container, axis=0)
  
    label_names = []
    prob_names = []
    for i, estimator in enumerate(op.estimators_):
        if is_regressor(estimator):
            raise NotImplementedError(
                f"Conversion of OneVersusOneClassifier for regressors "
                f"{type(estimator)} is not implemented yet.")
        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(
            op_type, raw_model=estimator)
        this_operator.inputs = operator.inputs

        if container.has_options(estimator, 'raw_scores'):
            options = {'raw_scores': True}
        elif container.has_options(estimator, 'zipmap'):
            options = {'zipmap': False}
        else:
            options = None
        if options is not None:
            container.add_options(id(estimator), options)
            scope.add_options(id(estimator), options)

        label_name = scope.declare_local_variable(
            'label_%d' % i, Int64TensorType())
        prob_name = scope.declare_local_variable(
            'proba_%d' % i, operator.inputs[0].type.__class__())
        this_operator.outputs.append(label_name)
        this_operator.outputs.append(prob_name)

        # gets the label for the class 1
        label = scope.get_unique_variable_name('lab_%d' % i)
        apply_reshape(scope, label_name.onnx_name, label, container,
                      desired_shape=(-1, 1))
        cast_label = scope.get_unique_variable_name('cast_lab_%d' % i)
        apply_cast(scope, label, cast_label, container,
                   to=proto_dtype)
        label_names.append(cast_label)

        # get the probability for the class 1
        prob_reshaped = scope.get_unique_variable_name('prob_%d' % i)
        container.add_node('Reshape', [prob_name.onnx_name, prob_shape],
                           [prob_reshaped])
        prob1 = scope.get_unique_variable_name('prob1_%d' % i)
        sliced = container.add_node(
            'Slice', [prob_reshaped, cst1, cst2, cst1], prob1)
        prob_names.append(prob1)

    conc_lab_name = scope.get_unique_variable_name('concat_out_ovo_label')
    apply_concat(scope, label_names, conc_lab_name, container, axis=1)
    conc_prob_name = scope.get_unique_variable_name('concat_out_ovo_prob')
    apply_concat(scope, prob_names, conc_prob_name, container, axis=1)

    # calls _ovr_decision_function
    this_operator = scope.declare_local_operator(
        "SklearnOVRDecisionFunction", op)

    cl_type = operator.inputs[0].type.__class__
    label = scope.declare_local_variable("label", cl_type())
    container.add_node('Identity', [conc_lab_name], [label.onnx_name])
    prob_score = scope.declare_local_variable("prob_score", cl_type())
    container.add_node('Identity', [conc_prob_name], [prob_score.onnx_name])

    this_operator.inputs.append(label)
    this_operator.inputs.append(prob_score)

    ovr_name = scope.declare_local_variable('ovr_output', cl_type())
    this_operator.outputs.append(ovr_name)

    output_name = operator.outputs[1].full_name
    container.add_node('Identity', [ovr_name.onnx_name], [output_name])

    container.add_node('ArgMax', 'ovr_output', 
        operator.outputs[0].full_name, axis=1)


register_converter('SklearnOneVsOneClassifier',
                   convert_one_vs_one_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'output_class_labels': [False, True],
                            'raw_scores': [True, False]})
