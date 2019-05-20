"""Tests scikit-learn's SGDClassifier converter."""

import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model


class TestSGDClassifierConverter(unittest.TestCase):
    def _fit_model_classification(self, model, n_classes):
        X, y = make_classification(n_classes=n_classes, n_features=100,
                                   n_samples=10000,
                                   random_state=42, n_informative=5)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                       random_state=42)
        model.fit(X_train, y_train)
        return model, X_test

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_hinge(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='hinge', random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinaryHinge-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_hinge(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='hinge', random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMultiHinge-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='log', random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinaryLog",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_log(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='log', random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType(X.shape))],
        )
        X = X[1:3]
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMultiLog",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log_l1_no_intercept(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='log', penalty='l1', fit_intercept=False,
                          random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinaryLogL1NoIntercept",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_log_l1_no_intercept(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='log', penalty='l1', fit_intercept=False,
                          random_state=42), 5)
        X = X[1:3]
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMultiLogL1NoIntercept",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_elasticnet_power_t(self):
        model, X = self._fit_model_classification(
            SGDClassifier(penalty='elasticnet', l1_ratio=0.3,
                          power_t=2, random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinaryElasticnetPowerT-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_elasticnet_power_t(self):
        model, X = self._fit_model_classification(
            SGDClassifier(penalty='elasticnet', l1_ratio=0.3,
                          power_t=2, random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMultiElasticnetPowerT-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_modified_huber(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='modified_huber', random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinaryModifiedHuber-Dec4",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_squared_hinge(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='squared_hinge', random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinarySquaredHinge-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_squared_hinge(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='squared_hinge', random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMultiSquaredHinge-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_perceptron(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='perceptron', random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinaryPerceptron-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_preceptron(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='perceptron', random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMultiPerceptron-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
