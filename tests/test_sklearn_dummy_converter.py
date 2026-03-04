# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's DummyRegressor and DummyClassifier converters."""

import unittest
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestDummyRegressorConverter(unittest.TestCase):
    def _run_regressor_test(self, model, X, dtype, basename):
        X = X.astype(dtype)
        input_type = (
            FloatTensorType([None, X.shape[1]])
            if dtype == np.float32
            else DoubleTensorType([None, X.shape[1]])
        )
        model_onnx = convert_sklearn(
            model,
            "DummyRegressor",
            [("input", input_type)],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename=basename)

    def test_dummy_regressor_mean_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyRegressor(strategy="mean")
        model.fit(X, np.array([1.0, 2.0, 3.0]))
        self._run_regressor_test(
            model, X, np.float32, "SklearnDummyRegressorMeanFloat"
        )

    def test_dummy_regressor_mean_double(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        model = DummyRegressor(strategy="mean")
        model.fit(X, np.array([1.0, 2.0, 3.0]))
        self._run_regressor_test(
            model, X, np.float64, "SklearnDummyRegressorMeanDouble"
        )

    def test_dummy_regressor_median_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyRegressor(strategy="median")
        model.fit(X, np.array([1.0, 2.0, 10.0]))
        self._run_regressor_test(
            model, X, np.float32, "SklearnDummyRegressorMedianFloat"
        )

    def test_dummy_regressor_constant_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyRegressor(strategy="constant", constant=42.0)
        model.fit(X, np.array([1.0, 2.0, 3.0]))
        self._run_regressor_test(
            model, X, np.float32, "SklearnDummyRegressorConstantFloat"
        )

    def test_dummy_regressor_multioutput_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyRegressor(strategy="mean")
        model.fit(X, np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
        self._run_regressor_test(
            model, X, np.float32, "SklearnDummyRegressorMultiOutputFloat"
        )


class TestDummyClassifierConverter(unittest.TestCase):
    def _run_classifier_test(self, model, X, dtype, basename):
        X = X.astype(dtype)
        input_type = (
            FloatTensorType([None, X.shape[1]])
            if dtype == np.float32
            else DoubleTensorType([None, X.shape[1]])
        )
        model_onnx = convert_sklearn(
            model,
            "DummyClassifier",
            [("input", input_type)],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename=basename)

    def test_dummy_classifier_prior_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyClassifier(strategy="prior")
        model.fit(X, np.array([0, 1, 2]))
        self._run_classifier_test(
            model, X, np.float32, "SklearnDummyClassifierPriorFloat"
        )

    def test_dummy_classifier_prior_double(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        model = DummyClassifier(strategy="prior")
        model.fit(X, np.array([0, 1, 2]))
        self._run_classifier_test(
            model, X, np.float64, "SklearnDummyClassifierPriorDouble"
        )

    def test_dummy_classifier_most_frequent_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, np.array([0, 0, 1]))
        self._run_classifier_test(
            model, X, np.float32, "SklearnDummyClassifierMostFrequentFloat"
        )

    def test_dummy_classifier_constant_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyClassifier(strategy="constant", constant=1)
        model.fit(X, np.array([0, 1, 2]))
        self._run_classifier_test(
            model, X, np.float32, "SklearnDummyClassifierConstantFloat"
        )

    def test_dummy_classifier_binary_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyClassifier(strategy="prior")
        model.fit(X, np.array([0, 1, 1]))
        self._run_classifier_test(
            model, X, np.float32, "SklearnDummyClassifierBinaryFloat"
        )

    def test_dummy_classifier_string_labels_float(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = DummyClassifier(strategy="prior")
        model.fit(X, np.array(["cat", "dog", "cat"]))
        self._run_classifier_test(
            model, X, np.float32, "SklearnDummyClassifierStringLabelsFloat"
        )

    def test_dummy_classifier_stochastic_not_supported(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        for strategy in ("stratified", "uniform"):
            model = DummyClassifier(strategy=strategy)
            model.fit(X, np.array([0, 1, 2]))
            with self.assertRaises(NotImplementedError):
                convert_sklearn(
                    model,
                    "DummyClassifier",
                    [("input", FloatTensorType([None, 2]))],
                    target_opset=TARGET_OPSET,
                    options={"zipmap": False},
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
