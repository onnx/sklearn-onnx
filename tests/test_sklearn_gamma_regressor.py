# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import unittest
import numpy as np

try:
    from sklearn.linear_model import GammaRegressor, PoissonRegressor
except ImportError:
    GammaRegressor = None
from onnxruntime import __version__ as ort_version
from skl2onnx import convert_sklearn

from skl2onnx.common.data_types import (
    FloatTensorType,
    DoubleTensorType,
    Int64TensorType,
)

from test_utils import dump_data_and_model, TARGET_OPSET

ort_version = ".".join(ort_version.split(".")[:2])


class TestGammaRegressorConverter(unittest.TestCase):
    @unittest.skipIf(GammaRegressor is None, reason="scikit-learn<1.0")
    def test_gamma_regressor_float(self):
        model = GammaRegressor()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 3]])
        y = np.array([19, 26, 33, 30])
        model.fit(X, y)
        test_x = np.array([[1, 0], [2, 8]])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn Gamma Regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertIsNotNone(model_onnx is not None)
        dump_data_and_model(
            test_x.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnGammaRegressor",
        )

    @unittest.skipIf(GammaRegressor is None, reason="scikit-learn<1.0")
    def test_gamma_regressor_int(self):
        model = GammaRegressor()
        X = np.array([[10, 20], [20, 30], [30, 40], [40, 30]])
        y = np.array([19, 26, 33, 30])
        model.fit(X, y)
        test_x = np.array([[1, 0], [2, 8]])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn Gamma Regressor",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertIsNotNone(model_onnx is not None)
        dump_data_and_model(
            test_x.astype(np.int64), model, model_onnx, basename="SklearnGammaRegressor"
        )

    @unittest.skipIf(GammaRegressor is None, reason="scikit-learn<1.0")
    def test_gamma_regressor_double(self):
        model = GammaRegressor()
        X = np.array([[1.1, 2.1], [2.3, 3.2], [3.2, 4.3], [4.2, 3.1]])
        y = np.array([19, 26, 33, 30])
        model.fit(X, y)
        test_x = np.array([[1.1, 0.1], [2.2, 8.4]])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn Gamma Regressor",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertIsNotNone(model_onnx is not None)
        dump_data_and_model(
            test_x.astype(np.double),
            model,
            model_onnx,
            basename="SklearnGammaRegressor",
        )

    @unittest.skipIf(GammaRegressor is None, reason="scikit-learn<1.0")
    def test_poisson_without_intercept(self):
        # Poisson
        model = PoissonRegressor(fit_intercept=False)
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 3.0]])
        y = np.array([19.0, 26.0, 33.0, 30.0])
        model.fit(X, y)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn Poisson Regressor without Intercept",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )

        self.assertIsNotNone(model_onnx is not None)

    @unittest.skipIf(GammaRegressor is None, reason="scikit-learn<1.0")
    def test_gamma_without_intercept(self):
        # Gamma
        model = GammaRegressor(fit_intercept=False)
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 3.0]])
        y = np.array([19.0, 26.0, 33.0, 30.0])
        model.fit(X, y)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn Gamma Regressor without Intercept",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )

        self.assertIsNotNone(model_onnx is not None)


if __name__ == "__main__":
    unittest.main(verbosity=3)
