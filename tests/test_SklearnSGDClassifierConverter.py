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
            SGDClassifier(loss='hinge'), 2)
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
            SGDClassifier(loss='hinge'), 5)
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
            SGDClassifier(loss='log'), 2)
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
            SGDClassifier(loss='log'), 5)
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
            basename="SklearnSGDClassifierMultiLog",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log_l1(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='log', penalty='l1', fit_intercept=False), 2)
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
            SGDClassifier(loss='log', penalty='l1', fit_intercept=False), 5)
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
    def test_model_sgd_binary_class_modified_huber(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='modified_huber'), 2)
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
            basename="SklearnSGDClassifierBinaryModifiedHuber",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log_elasticnet_power_t(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.3,
                          power_t=2), 2)
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
            basename="SklearnSGDClassifierBinaryLogElasticnetPowerT",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_log_elasticnet_power_t(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.3,
                          power_t=2), 5)
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
            basename="SklearnSGDClassifierMultiLogElasticnetPowerT",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_modified_huber(self):
        X, y = make_classification(n_classes=5, n_features=10, n_samples=10,
                                   random_state=42, n_informative=7)
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2,
                                                       random_state=42)
        model = SGDClassifier(loss='modified_huber').fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMultiModifiedHuber",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_squared_hinge(self):
        model, X = self._fit_model_classification(
            SGDClassifier(loss='squared_hinge'), 2)
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
            SGDClassifier(loss='squared_hinge'), 5)
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
            SGDClassifier(loss='perceptron'), 2)
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
            SGDClassifier(loss='perceptron'), 5)
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
