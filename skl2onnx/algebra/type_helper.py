# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from ..proto import TensorProto, ValueInfoProto, onnx_proto
from ..common._topology import Variable
from ..common.data_types import FloatTensorType, Int64TensorType
from ..common.data_types import StringTensorType
from ..common.data_types import Int32TensorType, DoubleTensorType
from ..common.data_types import BoolTensorType


def _guess_type_proto(data_type, dims):
    if data_type == onnx_proto.TensorProto.FLOAT:
        return FloatTensorType(dims)
    elif data_type == onnx_proto.TensorProto.DOUBLE:
        return DoubleTensorType(dims)
    elif data_type == onnx_proto.TensorProto.STRING:
        return StringTensorType(dims)
    elif data_type == onnx_proto.TensorProto.INT64:
        return Int64TensorType(dims)
    elif data_type == onnx_proto.TensorProto.INT32:
        return Int32TensorType(dims)
    elif data_type == onnx_proto.TensorProto.BOOL:
        return BoolTensorType(dims)
    else:
        raise NotImplementedError("Unsupported type '{}' "
                                  "data_type={}".format(
                                      type(data_type),
                                      dims))


def _guess_type(given_type):
    """
    Returns the proper type of an input.
    """
    if isinstance(given_type, np.ndarray):
        if given_type.dtype == np.float32:
            return FloatTensorType(given_type.shape)
        elif given_type.dtype == np.int64:
            return Int64TensorType(given_type.shape)
        elif given_type.dtype == np.str:
            return StringTensorType(given_type.shape)
        else:
            raise NotImplementedError(
                "Unsupported type '{}'. Double should "
                "be converted into floats.".format(given_type.dtype))
    elif isinstance(given_type, (FloatTensorType, Int64TensorType,
                                 StringTensorType)):
        return given_type
    elif isinstance(given_type, Variable):
        return given_type.type
    elif isinstance(given_type, TensorProto):
        return _guess_type_proto(given_type.data_type,
                                 given_type.dims)
    elif isinstance(given_type, ValueInfoProto):
        ttype = given_type.type.tensor_type
        dims = [ttype.shape.dim[i] for i in range(len(ttype.shape.dim))]
        return _guess_type_proto(ttype.elem_type, dims)
    else:
        raise NotImplementedError(
            "Unsupported type '{}'".format(type(given_type)))


def guess_initial_types(X, initial_types):
    if X is None:
        raise NotImplementedError("Initial types must be specified.")
    else:
        if isinstance(X, np.ndarray):
            X = X[:1]
        gt = _guess_type(X)
        initial_types = [('X', gt)]
    return initial_types
