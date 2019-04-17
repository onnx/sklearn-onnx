# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from test_utils import dump_data_and_model


class TestSklearnPCAConverter(unittest.TestCase):
    def test_pca_default(self):
        data = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                            data.target,
                                                            test_size=0.2,
                                                            random_state=42)
        model = PCA().fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType(shape=X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype("float32"),
            model,
            model_onnx,
            basename="SklearnPCADefault",
        )

    def test_pca_parameters_1(self):
        data = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                            data.target,
                                                            test_size=0.2,
                                                            random_state=42)
        model = PCA(
            copy=True,
            iterated_power="auto",
            n_components=0.9005263157894737,
            random_state=None,
            svd_solver="auto",
            tol=0.0,
            whiten=False,
        ).fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType(shape=X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype("float32"),
            model,
            model_onnx,
            basename="SklearnPCAParameters1",
        )

    def test_pca_parameters_2(self):
        data = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                            data.target,
                                                            test_size=0.2,
                                                            random_state=42)
        model = PCA(n_components=4).fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType(shape=X_test.shape))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype("float32"),
            model,
            model_onnx,
            basename="SklearnPCAParameters2",
        )


if __name__ == "__main__":
    unittest.main()
