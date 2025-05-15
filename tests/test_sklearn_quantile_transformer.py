# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's polynomial features converter.
"""
import unittest
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnQuantileTransformer(unittest.TestCase):
    def test_quantile_transformer_simple(self):
        X = np.empty((100, 2), dtype=np.float32)
        X[:, 0] = np.arange(X.shape[0])
        X[:, 1] = np.arange(X.shape[0]) * 2
        model = QuantileTransformer(n_quantiles=6).fit(X)
        model_onnx = convert_sklearn(
            model,
            "test",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=20,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnQuantileTransformer",
        )

    def test_quantile_transformer_int(self):
        X = np.random.randint(0, 5, (100, 20))
        model = QuantileTransformer(n_quantiles=6).fit(X)
        model_onnx = convert_sklearn(
            model,
            "test",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=20,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnQuantileTransformer",
        )

    def test_quantile_transformer_nan(self):
        X = np.random.randint(0, 5, (100, 20))
        X = X.astype(np.float32)
        X[0][0] = np.nan
        X[1][1] = np.nan
        model = QuantileTransformer(n_quantiles=6).fit(X)
        model_onnx = convert_sklearn(
            model,
            "test",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=20,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnQuantileTransformer",
        )


if __name__ == "__main__":
    unittest.main()
