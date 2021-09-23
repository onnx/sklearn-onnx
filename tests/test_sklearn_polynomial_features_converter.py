# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's polynomial features converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy as np
import onnx
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.preprocessing import PolynomialFeatures
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnPolynomialFeatures(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="ConstantOfShape not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_float_degree_2(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = convert_sklearn(
            model, "scikit-learn polynomial features",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnPolynomialFeaturesFloatDegree2")

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="ConstantOfShape not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_int_degree_2(self):
        X = np.array([
            [1, 3, 4, 0],
            [2, 3, 4, 1],
            [1, -4, 3, 7],
            [3, 10, -9, 5],
            [1, 0, 10, 5],
        ])
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = convert_sklearn(
            model, "scikit-learn polynomial features",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnPolynomialFeaturesIntDegree2")

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="ConstantOfShape not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_float_degree_3(self):
        X = np.array([[1.2, 3.2, 1.2], [4.3, 3.2, 4.5], [3.2, 4.7, 1.1]])
        model = PolynomialFeatures(degree=3).fit(X)
        model_onnx = convert_sklearn(
            model, "scikit-learn polynomial features",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnPolynomialFeaturesFloatDegree3")

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="ConstantOfShape not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_int_degree_3(self):
        X = np.array([
            [1, 3, 33],
            [4, 1, -11],
            [3, 7, -3],
            [3, 5, 4],
            [1, 0, 3],
            [5, 4, 9],
        ])
        model = PolynomialFeatures(degree=3).fit(X)
        model_onnx = convert_sklearn(
            model, "scikit-learn polynomial features",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnPolynomialFeaturesIntDegree3")

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="ConstantOfShape not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_float_degree_4(self):
        X = np.array([[1.2, 3.2, 3.1, 1.3], [4.3, 3.2, 0.5, 1.3],
                      [3.2, 4.7, 5.4, 7.1]])
        model = PolynomialFeatures(degree=4).fit(X)
        model_onnx = convert_sklearn(
            model, "scikit-learn polynomial features",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnPolynomialFeaturesFloatDegree4-Dec4")

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="ConstantOfShape not available")
    @ignore_warnings(category=FutureWarning)
    def test_model_polynomial_features_int_degree_4(self):
        X = np.array([[1, 3, 4, 1], [3, 7, 3, 5], [1, 0, 5, 4]])
        model = PolynomialFeatures(degree=4).fit(X)
        model_onnx = convert_sklearn(
            model, "scikit-learn polynomial features",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnPolynomialFeaturesIntDegree4")


if __name__ == "__main__":
    unittest.main()
