# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy as np
from distutils.version import StrictVersion
from pandas import DataFrame
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_binary_classification, dump_multiple_classification
from test_utils import dump_data_and_model
from onnxruntime import InferenceSession, __version__


class TestSklearnGradientBoostingModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(__version__) <= StrictVersion("0.5.0"),
        reason="Depends on PR #1015 onnxruntime.")
    def test_gradient_boosting_classifier1Deviance(self):
        model = GradientBoostingClassifier(n_estimators=1, max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)

        for cl in [None, 0.231, 1e-6, 0.9]:
            if cl is not None:
                model.init_.class_prior_ = np.array([cl, cl])
            initial_types = [('input', FloatTensorType((1, X.shape[1])))]
            model_onnx = convert_sklearn(model, initial_types=initial_types)
            if "Regressor" in str(model_onnx):
                raise AssertionError(str(model_onnx))
            sess = InferenceSession(model_onnx.SerializeToString())
            res = sess.run(None, {'input': X.astype(np.float32)})
            pred = model.predict_proba(X)
            if res[1][0][0] != pred[0, 0]:
                rows = ["X", str(X),
                        "base_values_", str(model.init_.class_prior_),
                        "predicted_label", str(model.predict(X)),
                        "expected", str(pred),
                        "onnxruntime", str(DataFrame(res[1])),
                        "model", str(model_onnx)]
                raise AssertionError("\n---\n".join(rows))
        dump_binary_classification(
            model, suffix="1Deviance",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.5.0')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_classifier3(self):
        model = GradientBoostingClassifier(n_estimators=3)
        dump_binary_classification(
            model, suffix="3",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.5.0')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_classifier_multi(self):
        model = GradientBoostingClassifier(n_estimators=3)
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
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
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.5.0')"
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
    unittest.main()
