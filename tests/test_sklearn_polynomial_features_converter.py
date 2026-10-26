# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's polynomial features converter.
"""

import unittest
import numpy as np
from onnx import checker
from onnxruntime import InferenceSession, SessionOptions

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.preprocessing import PolynomialFeatures
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnPolynomialFeatures(unittest.TestCase):
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_model_polynomial_features_bias_only(self):
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        for dtype, tensor_type in (
            (np.float32, FloatTensorType),
            (np.float64, DoubleTensorType),
            (np.int64, Int64TensorType),
        ):
            X = np.array([[0, 2], [3, -4], [5, 6]], dtype=dtype)
            for degree, interaction_only in (
                (0, False),
                ((0, 0), False),
                ((3, 3), True),
            ):
                model = PolynomialFeatures(
                    degree=degree, interaction_only=interaction_only
                ).fit(X)
                for opset in sorted({9, min(18, TARGET_OPSET), TARGET_OPSET}):
                    with self.subTest(dtype=dtype, degree=degree, opset=opset):
                        model_onnx = convert_sklearn(
                            model,
                            initial_types=[("input", tensor_type([None, X.shape[1]]))],
                            target_opset=opset,
                        )
                        checker.check_model(model_onnx, full_check=True)
                        session = InferenceSession(
                            model_onnx.SerializeToString(),
                            options,
                            providers=["CPUExecutionProvider"],
                        )
                        self.assertEqual(session.get_outputs()[0].shape, [None, 1])
                        for batch in (X[:1], X):
                            actual = session.run(None, {"input": batch})[0]
                            self.assertEqual(actual.dtype, dtype)
                            self.assertEqual(actual.shape, (batch.shape[0], 1))
                            np.testing.assert_array_equal(
                                actual, model.transform(batch)
                            )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_float_degree_2(self):
        X = np.array(
            [[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0], [0, 3.2, 4.7, -8.9]]
        )
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn polynomial features",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnPolynomialFeaturesFloatDegree2",
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_int_degree_2(self):
        X = np.array(
            [
                [1, 3, 4, 0],
                [2, 3, 4, 1],
                [1, -4, 3, 7],
                [3, 10, -9, 5],
                [1, 0, 10, 5],
            ]
        )
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn polynomial features",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnPolynomialFeaturesIntDegree2",
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_float_degree_3(self):
        X = np.array([[1.2, 3.2, 1.2], [4.3, 3.2, 4.5], [3.2, 4.7, 1.1]])
        model = PolynomialFeatures(degree=3).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn polynomial features",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnPolynomialFeaturesFloatDegree3",
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_int_degree_3(self):
        X = np.array(
            [
                [1, 3, 33],
                [4, 1, -11],
                [3, 7, -3],
                [3, 5, 4],
                [1, 0, 3],
                [5, 4, 9],
            ]
        )
        model = PolynomialFeatures(degree=3).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn polynomial features",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnPolynomialFeaturesIntDegree3",
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_float_degree_4(self):
        X = np.array([[1.2, 3.2, 3.1, 1.3], [4.3, 3.2, 0.5, 1.3], [3.2, 4.7, 5.4, 7.1]])
        model = PolynomialFeatures(degree=4).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn polynomial features",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnPolynomialFeaturesFloatDegree4-Dec4",
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_int_degree_4(self):
        X = np.array([[1, 3, 4, 1], [3, 7, 3, 5], [1, 0, 5, 4]])
        model = PolynomialFeatures(degree=4).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn polynomial features",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnPolynomialFeaturesIntDegree4",
        )


if __name__ == "__main__":
    unittest.main()
