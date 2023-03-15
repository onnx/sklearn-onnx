# SPDX-License-Identifier: Apache-2.0


import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_digits
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx import convert_sklearn
from test_utils import dump_data_and_model, TARGET_OPSET


def _fit_model_pca(model):
    data = load_diabetes()
    X_train, X_test, *_ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)
    model.fit(X_train)
    return model, X_test.astype(np.float32)


class TestSklearnPCAConverter(unittest.TestCase):
    def test_pca_default(self):
        model, X_test = _fit_model_pca(PCA(random_state=42))
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input",
                            FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnPCADefault")

    def test_incrementalpca_default(self):
        model, X_test = _fit_model_pca(IncrementalPCA())
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input",
                            FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnIncrementalPCADefault")

    def test_pca_parameters_auto(self):
        model, X_test = _fit_model_pca(PCA(
            random_state=42, copy=False, tol=0.1, whiten=True,
            n_components=0.9005263157894737, svd_solver="auto"))
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input",
                            FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnPCAParametersAuto")

    def test_pca_parameters_arpack(self):
        model, X_test = _fit_model_pca(PCA(
            random_state=42, n_components=4, svd_solver='arpack'))
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input",
                            FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnPCAParametersArpack")

    def test_pca_parameters_full(self):
        model, X_test = _fit_model_pca(PCA(
            random_state=42, n_components=5, svd_solver='full', whiten=True))
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input",
                            FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnPCAParametersFull")

    def test_pca_default_int_randomised(self):
        data = load_digits()
        X_train, X_test, *_ = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)
        model = PCA(random_state=42, svd_solver='randomized',
                    iterated_power=3).fit(X_train)
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input",
                            Int64TensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.int64), model, model_onnx,
            basename="SklearnPCADefaultIntRandomised")


if __name__ == "__main__":
    unittest.main()
