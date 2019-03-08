"""
Tests scikit-learn's polynomial features converter.
"""

import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures 
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnPolynomialFeatures(unittest.TestCase):

    def test_model_polynomial_features_float_degree_2(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = to_onnx(model, 'scikit-learn polynomial features',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnPolynomialFeaturesFloatDegree2",
                allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_polynomial_features_int_degree_2(self):
        X = np.array([[1, 3, 4, 0], [2, 3, 4, 1], [1, -4, 3, 7],
                      [3, 10, -9, 5], [1, 0, 10, 5]])
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = to_onnx(model, 'scikit-learn polynomial features',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                    basename="SklearnPolynomialFeaturesIntDegree2",
                    allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_polynomial_features_float_degree_3(self):
        X = np.array([[1.2, 3.2, 1.2], [4.3, 3.2, 4.5], [3.2, 4.7, 1.1]])
        model = PolynomialFeatures(degree=3).fit(X)
        model_onnx = to_onnx(model, 'scikit-learn polynomial features',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnPolynomialFeaturesFloatDegree3",
                allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_polynomial_features_int_degree_3(self):
        X = np.array([[1, 3, 33], [4, 1, -11], [3, 7, -3], [3, 5, 4],
                      [1, 0, 3], [5, 4, 9]])
        model = PolynomialFeatures(degree=3).fit(X)
        model_onnx = to_onnx(model, 'scikit-learn polynomial features',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                    basename="SklearnPolynomialFeaturesIntDegree3",
                    allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_polynomial_features_float_degree_4(self):
        X = np.array([[1.2, 3.2, 3.1, 1.3], [4.3, 3.2, 0.5, 1.3],
                      [3.2, 4.7, 5.4, 7.1]])
        model = PolynomialFeatures(degree=4).fit(X)
        model_onnx = to_onnx(model, 'scikit-learn polynomial features',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnPolynomialFeaturesFloatDegree4-Dec4",
                allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_polynomial_features_int_degree_4(self):
        X = np.array([[1, 3, 4, 1], [3, 7, 3, 5], [1, 0, 5, 4]])
        model = PolynomialFeatures(degree=4).fit(X)
        model_onnx = to_onnx(model, 'scikit-learn polynomial features',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                    basename="SklearnPolynomialFeaturesIntDegree4",
                    allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")


if __name__ == "__main__":
    unittest.main()
