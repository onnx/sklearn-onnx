"""
Tests scikit-learn's polynomial features converter.
"""

import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures 
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnPolynomialFeatures(unittest.TestCase):

    def test_model_polynomial_features_float(self):
        X = np.array([[1.2, 3.2], [4.3, 3.2], [3.2, 4.7]])
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn polynomial features',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnPolynomialFeaturesFloat")

    def test_model_polynomial_features_int(self):
        X = np.array([[1, 3], [4, 1], [3, 7], [3, 5], [1, 0], [5, 4]])
        model = PolynomialFeatures(degree=2).fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn polynomial features',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                    basename="SklearnPolynomialFeaturesInt")


if __name__ == "__main__":
    unittest.main()
