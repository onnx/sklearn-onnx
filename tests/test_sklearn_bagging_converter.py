# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model, fit_classification_model


class TestSklearnBaggingConverter(unittest.TestCase):
    def test_bagging_classifier_default_binary_int(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierDefaultBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_bagging_classifier_default_multiclass_int(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 4, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierDefaultMulticlass",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_bagging_classifier_default_binary(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 2)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierDefaultBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_bagging_classifier_default_multiclass(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 4)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierDefaultMulticlass",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_bagging_classifier_sgd_binary(self):
        model, X = fit_classification_model(
            BaggingClassifier(SGDClassifier()), 2)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierSGDBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_bagging_classifier_sgd_multiclass(self):
        model, X = fit_classification_model(
            BaggingClassifier(SGDClassifier()), 5)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierSGDMulticlass",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_bagging_classifier_adaboost_binary(self):
        model, X = fit_classification_model(
            BaggingClassifier(AdaBoostClassifier()), 2)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierAdaboostBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_bagging_classifier_adaboost_multiclass(self):
        model, X = fit_classification_model(
            BaggingClassifier(AdaBoostClassifier()), 3)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierAdaboostMulticlass",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
