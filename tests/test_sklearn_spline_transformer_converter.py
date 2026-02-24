# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's SplineTransformer converter.
"""

import unittest
import numpy as np

try:
    from sklearn.preprocessing import SplineTransformer
except ImportError:
    SplineTransformer = None

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


@unittest.skipIf(SplineTransformer is None, reason="SplineTransformer not available")
class TestSklearnSplineTransformer(unittest.TestCase):
    def _get_train_data(self, n_features=1, n_samples=50, seed=42):
        rng = np.random.default_rng(seed)
        return (rng.random((n_samples, n_features)) * 5).astype(np.float32)

    def _get_test_data(self, n_features=1, seed=0):
        rng = np.random.default_rng(seed)
        # Mix of in-range, at-boundary, and out-of-range points
        return np.vstack(
            [
                (rng.random((8, n_features)) * 5).astype(np.float32),  # in range
                np.zeros((1, n_features), dtype=np.float32),  # at lower boundary
                np.full((1, n_features), 5.0, dtype=np.float32),  # at upper boundary
            ]
        )

    def test_spline_transformer_default(self):
        """Test default parameters (n_knots=5, degree=3, extrapolation='constant')."""
        X_train = self._get_train_data()
        model = SplineTransformer(n_knots=5, degree=3).fit(X_train)
        X_test = self._get_test_data()
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerDefault",
        )

    def test_spline_transformer_degree2(self):
        """Test degree=2 spline."""
        X_train = self._get_train_data()
        model = SplineTransformer(n_knots=4, degree=2).fit(X_train)
        X_test = self._get_test_data()
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerDegree2",
        )

    def test_spline_transformer_multi_feature(self):
        """Test SplineTransformer with multiple input features."""
        n_features = 3
        X_train = self._get_train_data(n_features=n_features)
        model = SplineTransformer(n_knots=4, degree=3).fit(X_train)
        X_test = self._get_test_data(n_features=n_features)
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, n_features]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerMultiFeature",
        )

    def test_spline_transformer_no_bias(self):
        """Test SplineTransformer with include_bias=False."""
        X_train = self._get_train_data()
        model = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(X_train)
        X_test = self._get_test_data()
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerNoBias",
        )

    def test_spline_transformer_extrapolation_constant(self):
        """Test extrapolation='constant' with out-of-range values."""
        X_train = self._get_train_data()
        model = SplineTransformer(n_knots=4, degree=3, extrapolation="constant").fit(
            X_train
        )
        rng = np.random.default_rng(0)
        X_test = np.vstack(
            [
                (rng.random((8, 1)) * 5).astype(np.float32),
                np.array([[-1.0], [6.0]], dtype=np.float32),  # out of range
            ]
        )
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerExtrapolationConstant",
        )

    def test_spline_transformer_extrapolation_linear(self):
        """Test extrapolation='linear' with out-of-range values."""
        X_train = self._get_train_data()
        model = SplineTransformer(n_knots=4, degree=3, extrapolation="linear").fit(
            X_train
        )
        rng = np.random.default_rng(0)
        X_test = np.vstack(
            [
                (rng.random((8, 1)) * 5).astype(np.float32),
                np.array([[-1.0], [6.0]], dtype=np.float32),
            ]
        )
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerExtrapolationLinear",
        )

    def test_spline_transformer_extrapolation_continue(self):
        """Test extrapolation='continue' (polynomial extension)."""
        X_train = self._get_train_data()
        model = SplineTransformer(n_knots=4, degree=3, extrapolation="continue").fit(
            X_train
        )
        rng = np.random.default_rng(0)
        X_test = np.vstack(
            [
                (rng.random((8, 1)) * 5).astype(np.float32),
                np.array([[-1.0], [6.0]], dtype=np.float32),
            ]
        )
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerExtrapolationContinue",
        )

    def test_spline_transformer_extrapolation_periodic(self):
        """Test extrapolation='periodic'."""
        X_train = self._get_train_data()
        model = SplineTransformer(n_knots=5, degree=2, extrapolation="periodic").fit(
            X_train
        )
        rng = np.random.default_rng(0)
        X_test = np.vstack(
            [
                (rng.random((8, 1)) * 5).astype(np.float32),
                np.array([[-1.0], [6.0]], dtype=np.float32),
            ]
        )
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", FloatTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerExtrapolationPeriodic",
        )

    def test_spline_transformer_float64(self):
        """Test with double precision input."""
        X_train = self._get_train_data().astype(np.float64)
        model = SplineTransformer(n_knots=4, degree=3).fit(X_train)
        X_test = self._get_test_data().astype(np.float64)
        model_onnx = convert_sklearn(
            model,
            "SplineTransformer",
            [("input", DoubleTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnSplineTransformerFloat64",
        )


if __name__ == "__main__":
    unittest.main()
