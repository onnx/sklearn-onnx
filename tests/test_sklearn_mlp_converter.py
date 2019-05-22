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
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model


class TestSklearnMLPConverters(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_mlp_classifier_binary(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPClassifier().fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnMLPClassifierBinary",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_mlp_classifier_multiclass_default(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPClassifier().fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClass",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    def test_model_mlp_regressor_default(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPRegressor().fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnMLPRegressor",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_mlp_classifier_multiclass_identity(self):
        data = load_digits()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPClassifier(activation="identity").fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", Int64TensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClassIdentityActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    def test_model_mlp_regressor_identity(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPRegressor(activation="identity").fit(
            X_train.astype(np.int64), y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", Int64TensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnMLPRegressorIdentityActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_mlp_classifier_multiclass_logistic(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPClassifier(activation="logistic").fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClassLogisticActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_mlp_regressor_logistic(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPRegressor(activation="logistic").fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnMLPRegressorLogisticActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_mlp_classifier_multiclass_tanh(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPClassifier(activation="tanh").fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClassTanhActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    def test_model_mlp_regressor_tanh(self):
        data = load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = MLPRegressor(activation="tanh").fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnMLPRegressorTanhActivation-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
