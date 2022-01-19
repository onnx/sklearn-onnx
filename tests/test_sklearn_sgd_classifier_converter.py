# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.linear_model import SGDClassifier
from onnxruntime import __version__ as ort_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    TARGET_OPSET
)

ort_version = ".".join(ort_version.split(".")[:2])


class TestSGDClassifierConverter(unittest.TestCase):

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_hinge(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='hinge', random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierBinaryHinge-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_hinge(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='hinge', random_state=42), 5)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierMultiHinge-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_hinge_string(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='hinge', random_state=42), 5, label_string=True)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierMultiHinge-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 13,
                     reason="duplicated test")
    def test_model_sgd_binary_class_log_sigmoid(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 2, n_features=2)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=10, options={'zipmap': False})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32)[:5], model, model_onnx,
            basename="SklearnSGDClassifierBinaryLog-Dec4",
            verbose=False)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET, options={'zipmap': False})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32)[:5], model, model_onnx,
            basename="SklearnSGDClassifierBinaryLog13-Dec4",
            verbose=False)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 2)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=min(TARGET_OPSET, 10))
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierBinaryLog-Dec4")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log_decision_function(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 2)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierBinaryLogDecisionFunction-Dec3",
            methods=['predict', 'decision_function_binary'])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_log(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 5)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=min(12, TARGET_OPSET))
        X = np.array([X[1], X[1]])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierMultiLog")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 13, reason="duplicated test")
    def test_model_sgd_multi_class_log_sigmoid(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 5)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET, options={'zipmap': False})
        X = np.array([X[1], X[1]])
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx, verbose=False,
            basename="SklearnSGDClassifierMultiLog13")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_log_decision_function(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 3)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierMultiLogDecisionFunction-Dec3",
            methods=['predict', 'decision_function'])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log_l1_no_intercept(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', penalty='l1', fit_intercept=False,
                          random_state=42), 2)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierBinaryLogL1NoIntercept-Dec4")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion("1.0.0"),
                     reason="discrepencies")
    def test_model_sgd_multi_class_log_l1_no_intercept(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', penalty='l1', fit_intercept=False,
                          random_state=43), 3, n_features=7)
        X = np.array([X[4], X[4]])
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model, model_onnx, verbose=False,
            basename="SklearnSGDClassifierMultiLogL1NoIntercept-Dec4")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_elasticnet_power_t(self):
        model, X = fit_classification_model(
            SGDClassifier(penalty='elasticnet', l1_ratio=0.3,
                          power_t=2, random_state=42), 2)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierBinaryElasticnetPowerT-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_elasticnet_power_t(self):
        model, X = fit_classification_model(
            SGDClassifier(penalty='elasticnet', l1_ratio=0.3,
                          power_t=2, random_state=42), 5)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierMultiElasticnetPowerT-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_squared_hinge(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='squared_hinge', random_state=42), 2)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierBinarySquaredHinge-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_squared_hinge(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='squared_hinge', random_state=42), 5)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierMultiSquaredHinge-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_perceptron(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='perceptron', random_state=42), 2)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierBinaryPerceptron-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_perceptron(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='perceptron', random_state=42), 5)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDClassifierMultiPerceptron-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_hinge_int(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='hinge', random_state=42), 2, is_int=True)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDClassifierBinaryHingeInt-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_hinge_int(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='hinge', random_state=42), 5, is_int=True)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDClassifierMultiHingeInt-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log_int(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 2, is_int=True)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDClassifierBinaryLogInt")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class_log_bool(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 2, is_bool=True)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD binary classifier",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDClassifierBinaryLogBool")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class_log_int(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42), 5, is_int=True)
        model_onnx = convert_sklearn(
            model, "scikit-learn SGD multi-class classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        X = X[6:8]
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDClassifierMultiLogInt")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multi_class_nocl(self):
        model, X = fit_classification_model(
            SGDClassifier(loss='log', random_state=42),
            2, label_string=True)
        model_onnx = convert_sklearn(
            model, "multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X[6:8], model, model_onnx, classes=model.classes_,
            basename="SklearnSGDMultiNoCl", verbose=False)


if __name__ == "__main__":
    # TestSGDClassifierConverter().test_model_sgd_binary_class_log_sigmoid()
    unittest.main(verbosity=3)
