# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from scipy.sparse import coo_matrix
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
    def _guess_dim(value):
        if value == 0:
            return None
        return value

    if isinstance(given_type, (np.ndarray, coo_matrix)):
        shape = list(given_type.shape)
        if len(shape) == 0:
            # a number
            return _guess_numpy_type(given_type.dtype, tuple())
        shape[0] = None
        return _guess_numpy_type(given_type.dtype, shape)
    if isinstance(given_type, (FloatTensorType, Int64TensorType,
                               Int32TensorType, StringTensorType,
                               BooleanTensorType, DoubleTensorType)):
        return given_type
    if isinstance(given_type, Variable):
        return given_type.type
    if isinstance(given_type, DataType):
        return given_type
    if isinstance(given_type, TensorProto):
        return _guess_type_proto(given_type.data_type,
                                 given_type.dims)
    if isinstance(given_type, ValueInfoProto):
        ttype = given_type.type.tensor_type
        dims = [_guess_dim(ttype.shape.dim[i].dim_value)
                for i in range(len(ttype.shape.dim))]
        return _guess_type_proto(ttype.elem_type, dims)
    if isinstance(given_type, np.int64):
        return Int64Type()
    if given_type.__class__.__name__.endswith("Categorical"):
        # pandas Categorical without important pandas
        return Int64TensorType()
    raise NotImplementedError(
        "Unsupported type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(type(given_type)))


def guess_initial_types(X, initial_types):
    if X is None and initial_types is None:
        raise NotImplementedError("Initial types must be specified.")
    if initial_types is None:
        if isinstance(X, np.ndarray):
            X = X[:1]
        gt = _guess_type(X)
        initial_types = [('X', gt)]
    return initial_types
