# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import numpy as np
import onnxruntime
from sklearn.random_projection import GaussianRandomProjection
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET

nort = StrictVersion(onnxruntime.__version__) < StrictVersion('0.5.0')


class TestSklearnRandomProjection(unittest.TestCase):

    @unittest.skipIf(TARGET_OPSET < 9 or nort, reason="MatMul not available")
    def test_gaussian_random_projection_float32(self):
        rng = np.random.RandomState(42)
        pt = GaussianRandomProjection(n_components=4)
        X = rng.rand(10, 5)
        model = pt.fit(X)
        assert model.transform(X).shape[1] == 4
        model_onnx = convert_sklearn(
            model, "scikit-learn GaussianRandomProjection",
            [("inputs", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(np.float32), model,
                            model_onnx, basename="GaussianRandomProjection")

    @unittest.skipIf(TARGET_OPSET < 9 or nort, reason="MatMul not available")
    def test_gaussian_random_projection_float64(self):
        rng = np.random.RandomState(42)
        pt = GaussianRandomProjection(n_components=4)
        X = rng.rand(10, 5).astype(np.float64)
        model = pt.fit(X)
        model_onnx = to_onnx(model, X[:1], target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model,
                            model_onnx, basename="GaussianRandomProjection64")


if __name__ == "__main__":
    unittest.main()
