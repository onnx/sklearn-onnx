# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from skl2onnx.proto import onnx_proto
from skl2onnx.algebra.type_helper import _guess_type
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType,
    Int32TensorType, StringTensorType,
    BooleanTensorType, DoubleTensorType,
    Int8TensorType, UInt8TensorType,
    guess_data_type, guess_numpy_type, _guess_numpy_type,
    guess_proto_type, guess_tensor_type, _guess_type_proto)
try:
    from skl2onnx.common.data_types import (
        Complex64TensorType, Complex128TensorType)
except ImportError:
    Complex64TensorType = None
    Complex128TensorType = None


class TestAlgebraTestHelper(unittest.TestCase):

    def test_guess_type(self):
        dtypes = [
            (np.int32, Int32TensorType),
            (np.int64, Int64TensorType),
            (np.float32, FloatTensorType),
            (np.str_, StringTensorType),
            (np.bool_, BooleanTensorType),
            (np.int8, Int8TensorType),
            (np.uint8, UInt8TensorType)
        ]
        if Complex64TensorType is not None:
            dtypes.append((np.complex64, Complex64TensorType))
        if Complex128TensorType is not None:
            dtypes.append((np.complex128, Complex128TensorType))
        for dtype, exp in dtypes:
            if dtype == np.str_:
                mat = np.empty((3, 3), dtype=dtype)
                mat[:, :] = ""
            else:
                mat = np.zeros((3, 3), dtype=dtype)
            res = _guess_type(mat)
            assert isinstance(res, exp)

        dtypes = [np.float64]
        for dtype in dtypes:
            mat = np.zeros((3, 3), dtype=dtype)
            _guess_type(mat, )

    def test_guess_data_type(self):
        ty = guess_data_type(np.array([3, 5], dtype=np.int32))
        self.assertEqual(len(ty), 1)
        self.assertEqual(ty[0][0], 'input')
        assert isinstance(ty[0][1], Int32TensorType)

        ty = guess_data_type("tensor(int32)", shape=[3, 5])
        assert isinstance(ty, Int32TensorType)

        ty = guess_data_type("tensor(bool)")
        assert isinstance(ty, BooleanTensorType)

        ty = guess_data_type("tensor(int64)")
        assert isinstance(ty, Int64TensorType)

        ty = guess_data_type("tensor(float)")
        assert isinstance(ty, FloatTensorType)

        ty = guess_data_type("tensor(double)")
        assert isinstance(ty, DoubleTensorType)

        ty = guess_data_type("tensor(int8)")
        assert isinstance(ty, Int8TensorType)

        ty = guess_data_type("tensor(uint8)")
        assert isinstance(ty, UInt8TensorType)

        ty = guess_data_type("tensor(string)")
        assert isinstance(ty, StringTensorType)

        try:
            guess_data_type(None)
        except TypeError as e:
            assert "cannot be converted into a DataType" in str(e)

    def test_guess_numpy_type(self):
        dtypes = [
            (np.int32, Int32TensorType),
            (np.int64, Int64TensorType),
            (np.float32, FloatTensorType),
            (np.float64, DoubleTensorType),
            (np.str_, StringTensorType),
            (np.bool_, BooleanTensorType),
            (np.int8, Int8TensorType),
            (np.uint8, UInt8TensorType)
        ]
        if Complex64TensorType is not None:
            dtypes.append((np.complex64, Complex64TensorType))
        if Complex128TensorType is not None:
            dtypes.append((np.complex128, Complex128TensorType))
        for dtype, exp in dtypes:
            nt1 = _guess_numpy_type(dtype, [None, 1])
            nt2 = guess_numpy_type(dtype)
            self.assertEqual(nt1.__class__, exp)
            self.assertEqual(nt2, dtype)
            nt2 = guess_numpy_type(dtype)
            self.assertEqual(nt2, dtype)

    def test_proto_type(self):
        dtypes = [
            (np.int32, Int32TensorType, onnx_proto.TensorProto.INT32),
            (np.int64, Int64TensorType, onnx_proto.TensorProto.INT64),
            (np.float32, FloatTensorType, onnx_proto.TensorProto.FLOAT),
            (np.float64, DoubleTensorType, onnx_proto.TensorProto.DOUBLE),
            (np.str_, StringTensorType, onnx_proto.TensorProto.STRING),
            (np.bool_, BooleanTensorType, onnx_proto.TensorProto.BOOL),
            (np.int8, Int8TensorType, onnx_proto.TensorProto.INT8),
            (np.uint8, UInt8TensorType, onnx_proto.TensorProto.UINT8)
        ]
        if Complex64TensorType is not None:
            dtypes.append((np.complex64, Complex64TensorType,
                           onnx_proto.TensorProto.COMPLEX64))
        if Complex128TensorType is not None:
            dtypes.append((np.complex128, Complex128TensorType,
                           onnx_proto.TensorProto.COMPLEX128))
        for dtype, exp, pt in dtypes:
            nt2 = guess_proto_type(exp([None, 1]))
            self.assertEqual(nt2, pt)
            nt1 = _guess_type_proto(pt, [None, 1])
            self.assertEqual(nt1.__class__, exp)

    def test_tensor_type(self):
        dtypes = [
            (np.int32, FloatTensorType, onnx_proto.TensorProto.INT32),
            (np.int64, FloatTensorType, onnx_proto.TensorProto.INT64),
            (np.float32, FloatTensorType, onnx_proto.TensorProto.FLOAT),
            (np.float64, DoubleTensorType, onnx_proto.TensorProto.DOUBLE),
            (np.int8, FloatTensorType, onnx_proto.TensorProto.INT8),
            (np.uint8, FloatTensorType, onnx_proto.TensorProto.UINT8)
        ]
        if Complex64TensorType is not None:
            dtypes.append((np.complex64, Complex64TensorType,
                           onnx_proto.TensorProto.COMPLEX64))
        if Complex128TensorType is not None:
            dtypes.append((np.complex128, Complex128TensorType,
                           onnx_proto.TensorProto.COMPLEX128))
        for dtype, exp, pt in dtypes:
            nt2 = guess_tensor_type(exp([None, 1]))
            self.assertEqual(nt2.__class__, exp)


if __name__ == "__main__":
    unittest.main()
