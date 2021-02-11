# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from onnx import onnx_pb as onnx_proto
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from ..common._apply_operation import apply_identity
from ..common.data_types import (
    FloatTensorType, DoubleTensorType, guess_proto_type)
from ..common._registration import register_converter
from .._supported_operators import sklearn_operator_name_map


def convert_sklearn_tfidf_vectoriser(scope, operator, container):
    """
    Converter for scikit-learn's TfidfVectoriser.
    """
    tfidf_op = operator.raw_operator

    op_type = sklearn_operator_name_map[CountVectorizer]
    cv_operator = scope.declare_local_operator(op_type, tfidf_op)
    cv_operator.inputs = operator.inputs
    cv_output_name = scope.declare_local_variable('count_vec_output')
    columns = max(operator.raw_operator.vocabulary_.values()) + 1
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    if proto_dtype == onnx_proto.TensorProto.FLOAT:
        clr = FloatTensorType
    elif proto_dtype == onnx_proto.TensorProto.DOUBLE:
        clr = DoubleTensorType
    else:
        raise RuntimeError(
            "Unexpected dtype '{}'. Float or double expected.".format(
                proto_dtype))
    cv_output_name.type = clr([None, columns])
    cv_operator.outputs.append(cv_output_name)

    op_type = sklearn_operator_name_map[TfidfTransformer]
    tfidf_operator = scope.declare_local_operator(op_type, tfidf_op)
    tfidf_operator.inputs.append(cv_output_name)
    tfidf_output_name = scope.declare_local_variable('tfidf_output')
    tfidf_operator.outputs.append(tfidf_output_name)

    apply_identity(scope, tfidf_output_name.full_name,
                   operator.outputs[0].full_name, container)


register_converter('SklearnTfidfVectorizer', convert_sklearn_tfidf_vectoriser,
                   options={'tokenexp': None, 'separators': None})
