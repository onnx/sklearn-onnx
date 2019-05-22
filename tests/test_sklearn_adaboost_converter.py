# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.datasets import load_digits, load_iris
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, onnx_built_with_ml
from test_utils import dump_data_and_model


class TestSklearnAdaBoostModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ada_boost_classifier_samme_r(self):
        data = load_digits()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = AdaBoostClassifier(n_estimators=10, algorithm="SAMME.R")
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test.astype("float32"),
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierSAMMER",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ada_boost_classifier_samme(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = AdaBoostClassifier(n_estimators=15, algorithm="SAMME")
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", FloatTensorType(X_test.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test.astype("float32"),
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierSAMME",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_ada_boost_regressor(self):
        X, y = make_regression(n_features=4, n_samples=1000, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = AdaBoostRegressor(n_estimators=5)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(model, "AdaBoost regression",
                                     [("input", FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test.astype("float32"),
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressor-OneOffArray",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )


if __name__ == "__main__":
    unittest.main()
