import unittest
import numpy as np
from skl2onnx.algebra.type_helper import _guess_type
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType,
    Int32TensorType, StringTensorType
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
            try:
                _guess_type(mat)
                raise AssertionError("It should fail for type "
                                     "{}".format(dtype))
            except NotImplementedError:
                pass


if __name__ == "__main__":
    unittest.main()
