# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy as np
from pandas import DataFrame
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_binary_classification, dump_multiple_classification
from test_utils import dump_data_and_model
from onnxruntime import InferenceSession


class TestSklearnGradientBoostingModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_classifier1(self):
        model = GradientBoostingClassifier(n_estimators=1, max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((1, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict_proba(X)
        if res[1][0] != pred[0, 0]:
            raise AssertionError("{}\n--\n{}".format(pred, DataFrame(res[1])))
        dump_binary_classification(model, suffix="1")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_classifier3(self):
        model = GradientBoostingClassifier(n_estimators=3)
        dump_binary_classification(model, suffix="3")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_classifier_multi(self):
        model = GradientBoostingClassifier(n_estimators=3)
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.3.0')",
        )

    def _fit_regression_model(self, model):
        X, y = make_regression(n_features=4, random_state=42)
        model.fit(X, y)
        return model, X.astype(np.float32)

    def test_gradient_boosting_regressor_ls_loss(self):
        model, X = self._fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="ls"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([1, 4]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionLsLoss",
        )

    def test_gradient_boosting_regressor_lad_loss(self):
        model, X = self._fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="lad"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([1, 4]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionLadLoss",
        )

    def test_gradient_boosting_regressor_huber_loss(self):
        model, X = self._fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="huber"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([1, 4]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionHuberLoss",
        )

    def test_gradient_boosting_regressor_quantile_loss(self):
        model, X = self._fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="quantile"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([1, 4]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionQuantileLoss",
        )


if __name__ == "__main__":
    TestSklearnGradientBoostingModels().test_gradient_boosting_classifier1()
    unittest.main()
