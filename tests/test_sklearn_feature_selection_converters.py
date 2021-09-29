# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's feature selection converters
"""
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    RFE,
    RFECV,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
)
from sklearn.feature_selection import (
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
)
from sklearn.svm import SVR
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnFeatureSelectionConverters(unittest.TestCase):
    def test_generic_univariate_select_int(self):
        model = GenericUnivariateSelect()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "generic univariate select",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGenericUnivariateSelect")

    def test_rfe_int(self):
        model = RFE(estimator=SVR(kernel="linear"))
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "rfe", [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRFE",
            methods=["transform"])

    def test_rfecv_int(self):
        model = RFECV(estimator=SVR(kernel="linear"), cv=3)
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "rfecv", [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRFECV",
            methods=["transform"])

    def test_select_fdr_int(self):
        model = SelectFdr()
        X, y = load_breast_cancer(return_X_y=True)
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select fdr",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnSelectFdr")

    def test_select_fpr_int(self):
        model = SelectFpr()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select fpr",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectFpr")

    def test_select_from_model_int(self):
        model = SelectFromModel(estimator=SVR(kernel="linear"))
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select from model",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectFromModel")

    def test_select_fwe_int(self):
        model = SelectFwe()
        X, y = load_breast_cancer(return_X_y=True)
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select fwe",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnSelectFwe")

    def test_select_k_best_int(self):
        model = SelectKBest(k="all")
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select k best",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectKBest")

    def test_select_percentile_int(self):
        model = SelectPercentile()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64,
        )
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select percentile",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectPercentile")

    def test_variance_threshold_int(self):
        model = VarianceThreshold()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.int64)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "variance threshold",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnVarianceThreshold")

    def test_generic_univariate_select_float(self):
        model = GenericUnivariateSelect()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "generic univariate select",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGenericUnivariateSelect")

    def test_rfe_float(self):
        model = RFE(estimator=SVR(kernel="linear"))
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "rfe", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRFE",
            methods=["transform"])

    def test_rfecv_float(self):
        model = RFECV(estimator=SVR(kernel="linear"), cv=3)
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "rfecv", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRFECV",
            methods=["transform"])

    def test_select_fdr_float(self):
        model = SelectFdr()
        X, y = load_breast_cancer(return_X_y=True)
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select fdr",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSelectFdr")

    def test_select_fpr_float(self):
        model = SelectFpr()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select fpr",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectFpr")

    def test_select_from_model_float(self):
        model = SelectFromModel(estimator=SVR(kernel="linear"))
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select from model",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectFromModel")

    def test_select_from_model_float_nomodel(self):
        model = SelectFromModel(
            estimator=SVR(kernel="linear"), threshold=1e5)
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        with self.assertRaises(RuntimeError):
            convert_sklearn(
                model, "select from model",
                [("input", FloatTensorType([None, X.shape[1]]))],
                target_opset=TARGET_OPSET)

    def test_select_fwe_float(self):
        model = SelectFwe()
        X, y = load_breast_cancer(return_X_y=True)
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select fwe",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model, model_onnx, basename="SklearnSelectFwe")

    def test_select_k_best_float(self):
        model = SelectKBest(k="all")
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select k best",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectKBest")

    def test_select_percentile_float(self):
        model = SelectPercentile()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "select percentile",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSelectPercentile")

    def test_variance_threshold_float(self):
        model = VarianceThreshold()
        X = np.array(
            [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
            dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model, "variance threshold",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnVarianceThreshold")


if __name__ == "__main__":
    unittest.main()
