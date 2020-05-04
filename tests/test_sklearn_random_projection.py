# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnRandomProjection(unittest.TestCase):

    @unittest.skipIf(TARGET_OPSET < 9, reason="MatMul not available")
    def test_gaussian_random_projection_float32(self):
        rng = np.random.RandomState(42)
        pt = GaussianRandomProjection(n_components=4)
        X = rng.rand(10, 5)
        model = pt.fit(X)
        assert model.transform(X).shape[1] == 4
        model_onnx = convert_sklearn(
            model, "scikit-learn GaussianRandomProjection",
            [("inputs", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(np.float32), model,
                            model_onnx, basename="GaussianRandomProjection")

    @unittest.skipIf(TARGET_OPSET < 9, reason="MatMul not available")
    def test_gaussian_random_projection_float64(self):
        rng = np.random.RandomState(42)
        pt = GaussianRandomProjection(n_components=4)
        X = rng.rand(10, 5).astype(np.float64)
        model = pt.fit(X)
        model_onnx = to_onnx(model, X[:1], dtype=np.float64)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model,
                            model_onnx, basename="GaussianRandomProjection64")


if __name__ == "__main__":
    unittest.main()
