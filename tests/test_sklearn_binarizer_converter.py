# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's binarizer converter.
"""

import unittest
import numpy as np
from sklearn.preprocessing import Binarizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnBinarizer(unittest.TestCase):
    def test_model_binarizer(self):
        data = np.array([[1., -1., 2.],
                         [2., 0., 0.],
                         [0., 1., -1.]], dtype=np.float32)
        model = Binarizer(threshold=0.5)
        model_onnx = convert_sklearn(
            model, "scikit-learn binarizer",
            [("input", FloatTensorType(data.shape))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnBinarizer-SkipDim1")


if __name__ == "__main__":
    unittest.main()
