"""
Tests scikit-learn's MLPClassifier and MLPRegressor converters.
"""

import unittest

import numpy as np
from sklearn.datasets import load_diabetes, load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnCalibratedClassifierCVConverters(unittest.TestCase):

    def test_model_mlp_classifier_binary(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPClassifier().fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPClassifier',
                                     [('input', FloatTensorType([1, 4]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32), model, model_onnx,
            basename="SklearnMLPClassifierBinary",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')")

    def test_model_mlp_classifier_multiclass_default(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPClassifier().fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPClassifier',
                                     [('input', FloatTensorType([1, 4]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test.astype(np.float32), model, model_onnx,
                            basename="SklearnMLPClassifierMultiClass")

    def test_model_mlp_regressor_default(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPRegressor().fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPRegressor',
                                     [('input', FloatTensorType([1, 10]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X_test.astype(np.float32), model, model_onnx,
                            basename="SklearnMLPRegressor")

    def test_model_mlp_classifier_multiclass_identity(self):
        data = load_digits()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPClassifier(activation='identity').fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPClassifier',
                                     [('input', Int64TensorType([1, 64]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.int64), model, model_onnx,
            basename="SklearnMLPClassifierMultiClassIdentityActivation")

    def test_model_mlp_regressor_identity(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPRegressor(activation='identity').fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPRegressor',
                                     [('input', FloatTensorType([1, 10]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32), model, model_onnx,
            basename="SklearnMLPRegressorIdentityActivation")

    def test_model_mlp_classifier_multiclass_logistic(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPClassifier(activation='logistic').fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPClassifier',
                                     [('input', FloatTensorType([1, 4]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32), model, model_onnx,
            basename="SklearnMLPClassifierMultiClassLogisticActivation")

    def test_model_mlp_regressor_logistic(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPRegressor(activation='logistic').fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPRegressor',
                                     [('input', FloatTensorType([1, 10]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32), model, model_onnx,
            basename="SklearnMLPRegressorLogisticActivation")

    def test_model_mlp_classifier_multiclass_tanh(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPClassifier(activation='tanh').fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPClassifier',
                                     [('input', FloatTensorType([1, 4]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32), model, model_onnx,
            basename="SklearnMLPClassifierMultiClassTanhActivation")

    def test_model_mlp_regressor_tanh(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = MLPRegressor(activation='tanh').fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'scikit-learn MLPRegressor',
                                     [('input', FloatTensorType([1, 10]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32), model, model_onnx,
            basename="SklearnMLPRegressorTanhActivation")


if __name__ == "__main__":
    unittest.main()
