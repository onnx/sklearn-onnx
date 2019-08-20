# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from ..proto import TensorProto, ValueInfoProto
from ..common._topology import Variable
from ..common.data_types import (
    _guess_numpy_type,
    _guess_type_proto,
    BooleanTensorType,
    DataType,
    DoubleTensorType,
    FloatTensorType,
    Int64Type,
    Int64TensorType, Int32TensorType,
    StringTensorType
)


def _guess_type(given_type):
    """
    Returns the proper type of an input.
    """
    if isinstance(given_type, np.ndarray):
        shape = list(given_type.shape)
        shape[0] = None
        return _guess_numpy_type(given_type.dtype, shape)
    elif isinstance(given_type, (FloatTensorType, Int64TensorType,
                                 Int32TensorType, StringTensorType,
                                 BooleanTensorType, DoubleTensorType)):
        return given_type
    elif isinstance(given_type, Variable):
        return given_type.type
    elif isinstance(given_type, DataType):
        return given_type
    elif isinstance(given_type, TensorProto):
        return _guess_type_proto(given_type.data_type,
                                 given_type.dims)
    elif isinstance(given_type, ValueInfoProto):
        ttype = given_type.type.tensor_type
        dims = [ttype.shape.dim[i].dim_value
                for i in range(len(ttype.shape.dim))]
        return _guess_type_proto(ttype.elem_type, dims)
    elif isinstance(given_type, np.int64):
        return Int64Type()
    else:
        raise NotImplementedError(
            "Unsupported type '{}'. You may raise an issue "
            "at https://github.com/onnx/sklearn-onnx/issues."
            "".format(type(given_type)))


def guess_initial_types(X, initial_types):
    if X is None and initial_types is None:
        raise NotImplementedError("Initial types must be specified.")
    elif initial_types is None:
        if isinstance(X, np.ndarray):
            X = X[:1]
        gt = _guess_type(X)
        initial_types = [('X', gt)]
    return initial_types
