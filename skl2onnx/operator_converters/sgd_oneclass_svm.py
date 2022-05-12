# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..common._apply_operation import (
    apply_add, apply_cast, apply_clip, apply_concat, apply_div, apply_exp,
    apply_identity, apply_mul, apply_reciprocal, apply_reshape, apply_sub)
from ..common.data_types import (
    BooleanTensorType, Int64TensorType, guess_numpy_type,
    guess_proto_type)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto
from scipy.sparse import isspmatrix


def convert_sklearn_sgd_oneclass_svm(scope: Scope, operator: Operator,
                                   container: ModelComponentContainer):
  
    op = operator.raw_operator
    if isinstance(op.coef_, np.ndarray):
        coef = op.coef_.ravel()
    else:
        coef = op.coef_
    intercept = op.offset_



    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT



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



register_converter('SklearnSGDOneClassSVM', convert_sklearn_sgd_oneclass_svm)
