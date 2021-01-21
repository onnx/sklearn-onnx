# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from ..proto import TensorProto, onnx_proto
from onnxconverter_common.data_types import DataType, Int64Type, FloatType  # noqa
from onnxconverter_common.data_types import StringType, TensorType  # noqa
from onnxconverter_common.data_types import (  # noqa
    Int64TensorType, Int32TensorType, BooleanTensorType,
    FloatTensorType, StringTensorType, DoubleTensorType,
    DictionaryType, SequenceType)
from onnxconverter_common.data_types import find_type_conversion, onnx_built_with_ml  # noqa


def _guess_type_proto(data_type, dims):
    # This could be moved to onnxconverter_common.
    for d in dims:
        if d == 0:
            raise RuntimeError(
                "Dimension should not be null: {}.".format(
                    list(dims)))
    if data_type == onnx_proto.TensorProto.FLOAT:
        return FloatTensorType(dims)
    if data_type == onnx_proto.TensorProto.DOUBLE:
        return DoubleTensorType(dims)
    if data_type == onnx_proto.TensorProto.STRING:
        return StringTensorType(dims)
    if data_type == onnx_proto.TensorProto.INT64:
        return Int64TensorType(dims)
    if data_type == onnx_proto.TensorProto.INT32:
        return Int32TensorType(dims)
    if data_type == onnx_proto.TensorProto.BOOL:
        return BooleanTensorType(dims)
    raise NotImplementedError(
        "Unsupported data_type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(data_type))


def _guess_type_proto_str(data_type, dims):
    # This could be moved to onnxconverter_common.
    if data_type == "tensor(float)":
        return FloatTensorType(dims)
    if data_type == "tensor(double)":
        return DoubleTensorType(dims)
    if data_type == "tensor(string)":
        return StringTensorType(dims)
    if data_type == "tensor(int64)":
        return Int64TensorType(dims)
    if data_type == "tensor(int32)":
        return Int32TensorType(dims)
    if data_type == "tensor(bool)":
        return BooleanTensorType(dims)
    raise NotImplementedError(
        "Unsupported data_type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(data_type))


def _guess_numpy_type(data_type, dims):
    # This could be moved to onnxconverter_common.
    if data_type == np.float32:
        return FloatTensorType(dims)
    if data_type == np.float64:
        return DoubleTensorType(dims)
    if data_type in (np.str, str, object) or str(data_type) in ('<U1', ) or (
            hasattr(data_type, 'type') and data_type.type is np.str_): # noqa
        return StringTensorType(dims)
    if data_type in (np.int64, np.uint64) or str(data_type) == '<U6':
        return Int64TensorType(dims)
    if data_type in (np.int32, np.uint32) or str(
        data_type) in ('<U4', ): # noqa
        return Int32TensorType(dims)
    if data_type == np.bool:
        return BooleanTensorType(dims)
    raise NotImplementedError(
        "Unsupported data_type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(data_type))


def guess_data_type(type_, shape=None):
    """
    Guess the datatype given the type type_
    """
    if isinstance(type_, TensorProto):
        return _guess_type_proto(type, shape)
    if isinstance(type_, str):
        return _guess_type_proto_str(type_, shape)
    if hasattr(type_, 'columns') and hasattr(type_, 'dtypes'):
        # DataFrame
        return [(name, _guess_numpy_type(dt, [None, 1]))
                for name, dt in zip(type_.columns, type_.dtypes)]
    if hasattr(type_, 'name') and hasattr(type_, 'dtype'):
        # Series
        return [(type_.name, _guess_numpy_type(type_.dtype, [None, 1]))]
    if hasattr(type_, 'shape') and hasattr(type_, 'dtype'):
        # array
        return [('input', _guess_numpy_type(type_.dtype, type_.shape))]
    raise TypeError("Type {} cannot be converted into a "
                    "DataType. You may raise an issue at "
                    "https://github.com/onnx/sklearn-onnx/issues."
                    "".format(type(type_)))


def guess_numpy_type(data_type):
    """
    Guess the corresponding numpy type based on data_type.
    """
    if data_type in (np.float64, np.float32, np.int8, np.uint8,
                     np.str, np.bool, np.int32, np.int64):
        return data_type
    if isinstance(data_type, FloatTensorType):
        return np.float32
    if isinstance(data_type, DoubleTensorType):
        return np.float64
    if isinstance(data_type, Int32TensorType):
        return np.int32
    if isinstance(data_type, Int64TensorType):
        return np.int64
    if isinstance(data_type, StringTensorType):
        return np.str
    if isinstance(data_type, BooleanTensorType):
        return np.bool
    raise NotImplementedError(
        "Unsupported data_type '{}'.".format(data_type))


def guess_proto_type(data_type):
    """
    Guess the corresponding proto type based on data_type.
    """
    if isinstance(data_type, FloatTensorType):
        return onnx_proto.TensorProto.FLOAT
    if isinstance(data_type, DoubleTensorType):
        return onnx_proto.TensorProto.DOUBLE
    if isinstance(data_type, Int32TensorType):
        return onnx_proto.TensorProto.INT32
    if isinstance(data_type, Int64TensorType):
        return onnx_proto.TensorProto.INT64
    if isinstance(data_type, StringTensorType):
        return onnx_proto.TensorProto.STRING
    if isinstance(data_type, BooleanTensorType):
        return onnx_proto.TensorProto.BOOL
    raise NotImplementedError(
        "Unsupported data_type '{}'.".format(data_type))


def guess_tensor_type(data_type):
    """
    Guess the corresponding variable type based on input type.
    """
    if isinstance(data_type, DoubleTensorType):
        return DoubleTensorType()
    if isinstance(data_type, DictionaryType):
        return guess_tensor_type(data_type.value_type)
    if not isinstance(data_type, (
            Int64TensorType, Int32TensorType, BooleanTensorType,
            FloatTensorType, StringTensorType, DoubleTensorType)):
        raise TypeError(
            "data_type is not a tensor type but '{}'.".format(
                type(data_type)))
    return FloatTensorType()
