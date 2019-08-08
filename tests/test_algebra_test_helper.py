import unittest
import numpy as np
from skl2onnx.algebra.type_helper import _guess_type
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType,
    Int32TensorType, StringTensorType,
    BooleanTensorType, DoubleTensorType,
    guess_data_type
)


class TestAlgebraTestHelper(unittest.TestCase):

    def test_guess_type(self):
        dtypes = [
            (np.int32, Int32TensorType),
            (np.int64, Int64TensorType),
            (np.float32, FloatTensorType),
            (np.str, StringTensorType)
        ]
        for dtype, exp in dtypes:
            if dtype == np.str:
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

        ty = guess_data_type("tensor(string)")
        assert isinstance(ty, StringTensorType)

        try:
            guess_data_type(None)
        except TypeError as e:
            assert "cannot be converted into a DataType" in str(e)


if __name__ == "__main__":
    unittest.main()
