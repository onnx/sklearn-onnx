# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model, fit_regression_model


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

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ada_boost_classifier_lr(self):
        data = load_digits()
        X, y = data.data, data.target
        X = X.astype('int64')
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = AdaBoostClassifier(learning_rate=0.3, random_state=42)
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", Int64TensorType(X_test.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierLR",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_ada_boost_regressor(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(n_estimators=5))
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressor-OneOffArray-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    def test_ada_boost_regressor_int(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(n_estimators=5), is_int=True)
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", Int64TensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressorInt-OneOffArray",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    def test_ada_boost_regressor_lr(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(learning_rate=0.5, random_state=42))
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressorLR-OneOffArray-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )


if __name__ == "__main__":
    unittest.main()
