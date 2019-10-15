"""Tests scikit-learn's Passive Aggressive Classifier converter."""

import unittest
from sklearn.linear_model import PassiveAggressiveClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model, fit_classification_model


class TestPassiveAggressiveClassifierConverter(unittest.TestCase):

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_passive_aggressive_classifier_binary_class(self):
        model, X = fit_classification_model(
            PassiveAggressiveClassifier(random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn PassiveAggressiveClassifier binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnPassiveAggressiveClassifierBinary-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_passive_aggressive_classifier_multi_class(self):
        model, X = fit_classification_model(
            PassiveAggressiveClassifier(random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn PassiveAggressiveClassifier multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnPassiveAggressiveClassifierMulti-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_passive_aggressive_classifier_binary_class_int(self):
        model, X = fit_classification_model(
            PassiveAggressiveClassifier(random_state=42), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn PassiveAggressiveClassifier binary classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnPassiveAggressiveClassifierBinaryInt-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_passive_aggressive_classifier_multi_class_int(self):
        model, X = fit_classification_model(
            PassiveAggressiveClassifier(random_state=42), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn PassiveAggressiveClassifier multi-class classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnPassiveAggressiveClassifierMultiInt-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
