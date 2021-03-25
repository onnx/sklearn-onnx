# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from onnxconverter_common.data_types import (  # noqa
    DataType, Int64Type, FloatType,  # noqa
    StringType, TensorType,  # noqa
    Int64TensorType, Int32TensorType, BooleanTensorType,  # noqa
    FloatTensorType, StringTensorType, DoubleTensorType,  # noqa
    DictionaryType, SequenceType)  # noqa
from onnxconverter_common.data_types import find_type_conversion, onnx_built_with_ml  # noqa


try:
    from onnxconverter_common.data_types import DoubleType
except ImportError:

    class DoubleType(DataType):
        def __init__(self, doc_string=''):
            super(DoubleType, self).__init__([1, 1], doc_string)

        def to_onnx_type(self):
            onnx_type = onnx_proto.TypeProto()
            onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.DOUBLE
            s = onnx_type.tensor_type.shape.dim.add()
            s.dim_value = 1
            return onnx_type

        def __repr__(self):
            return "{}()".format(self.__class__.__name__)


try:
    from onnxconverter_common.data_types import UInt8TensorType
except ImportError:

    class UInt8TensorType(TensorType):
        def __init__(self, shape=None, doc_string=''):
            super(UInt8TensorType, self).__init__(shape, doc_string)

        def _get_element_onnx_type(self):
            return onnx_proto.TensorProto.UINT8


try:
    from onnxconverter_common.data_types import UInt8Type
except ImportError:

    class UInt8Type(DataType):
        def __init__(self, doc_string=''):
            super(UInt8Type, self).__init__([1, 1], doc_string)

        def to_onnx_type(self):
            onnx_type = onnx_proto.TypeProto()
            onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.UINT8
            s = onnx_type.tensor_type.shape.dim.add()
            s.dim_value = 1
            return onnx_type

        def __repr__(self):
            return "{}()".format(self.__class__.__name__)


from ..proto import TensorProto, onnx_proto


def copy_type(vtype, empty=True):
    if isinstance(vtype, SequenceType):
        return vtype.__class__(copy_type(vtype.element_type))
    if isinstance(vtype, DictionaryType):
        return vtype.__class__(copy_type(vtype.key_type),
                               copy_type(vtype.value_type))
    return vtype.__class__()


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


def _guess_type_proto_str_inv(data_type):
    # This could be moved to onnxconverter_common.
    if isinstance(data_type, FloatTensorType):
        return "tensor(float)"
    if isinstance(data_type, DoubleTensorType):
        return "tensor(double)"
    if isinstance(data_type, StringTensorType):
        return "tensor(string)"
    if isinstance(data_type, Int64TensorType):
        return "tensor(int64)"
    if isinstance(data_type, Int32TensorType):
        return "tensor(int32)"
    if isinstance(data_type, BooleanTensorType):
        return "tensor(bool)"
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
    if data_type in (np.int64, ) or str(data_type) == '<U6':
        return Int64TensorType(dims)
    if data_type in (np.int32, ) or str(
        data_type) in ('<U4', ): # noqa
        return Int32TensorType(dims)
    if data_type == np.uint8:
        return UInt8TensorType(dims)
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
    if isinstance(data_type, UInt8TensorType):
        return onnx_proto.TensorProto.UINT8
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
