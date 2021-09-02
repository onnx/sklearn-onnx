# SPDX-License-Identifier: Apache-2.0


from distutils.version import StrictVersion
import unittest
import numpy as np
from pandas import DataFrame
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split
from onnxruntime import InferenceSession, __version__
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_binary_classification, dump_multiple_classification
from test_utils import fit_classification_model
from test_utils import dump_data_and_model, fit_regression_model

THRESHOLD = "0.2.1"


class TestSklearnGradientBoostingModels(unittest.TestCase):

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(__version__) <= StrictVersion(THRESHOLD),
        reason="Depends on PR #1015 onnxruntime.")
    def test_gradient_boosting_classifier1Deviance(self):
        model = GradientBoostingClassifier(n_estimators=1, max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)

        for cl in [None, 0.231, 1e-6, 0.9]:
            if cl is not None:
                model.init_.class_prior_ = np.array([cl, cl])
            initial_types = [('input', FloatTensorType((None, X.shape[1])))]
            model_onnx = convert_sklearn(model, initial_types=initial_types)
            if "Regressor" in str(model_onnx):
                raise AssertionError(str(model_onnx))
            sess = InferenceSession(model_onnx.SerializeToString())
            res = sess.run(None, {'input': X.astype(np.float32)})
            pred = model.predict_proba(X)
            delta = abs(res[1][0][0] - pred[0, 0])
            if delta > 1e-5:
                rows = ["diff", str(delta),
                        "X", str(X),
                        "base_values_", str(model.init_.class_prior_),
                        "predicted_label", str(model.predict(X)),
                        "expected", str(pred),
                        "onnxruntime", str(DataFrame(res[1])),
                        "model", str(model_onnx)]
                raise AssertionError("\n---\n".join(rows))
        dump_binary_classification(
            model, suffix="1Deviance",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('%s')" % THRESHOLD)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_classifier3(self):
        model = GradientBoostingClassifier(n_estimators=3)
        dump_binary_classification(
            model, suffix="3",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('%s')" % THRESHOLD)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_classifier_multi(self):
        model = GradientBoostingClassifier(n_estimators=3)
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('%s')" % THRESHOLD,
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_binary_classification(self):
        model, X = fit_classification_model(
            GradientBoostingClassifier(n_estimators=3), 2)
        model_onnx = convert_sklearn(
            model,
            "gradient boosting classifier",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingBinaryClassifier",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_binary_classification_init_zero(self):
        model, X = fit_classification_model(
            GradientBoostingClassifier(n_estimators=4, init='zero'), 2)
        model_onnx = convert_sklearn(
            model,
            "gradient boosting classifier",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingBinaryClassifierInitZero",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_multiclass_classification(self):
        model, X = fit_classification_model(
            GradientBoostingClassifier(n_estimators=4), 5)
        model_onnx = convert_sklearn(
            model,
            "gradient boosting classifier",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingMultiClassClassifier",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_int(self):
        model, X = fit_classification_model(
            GradientBoostingClassifier(n_estimators=4), 5, is_int=True)
        model_onnx = convert_sklearn(
            model, "gradient boosting classifier",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingInt",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_bool(self):
        model, X = fit_classification_model(
            GradientBoostingClassifier(n_estimators=4), 5, is_bool=True)
        model_onnx = convert_sklearn(
            model, "gradient boosting classifier",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingBool",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_multiclass_decision_function(self):
        model, X = fit_classification_model(
            GradientBoostingClassifier(n_estimators=4), 5)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "gradient boosting classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingMultiClassDecisionFunction",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function'],
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_gradient_boosting_multiclass_classification_init_zero(self):
        model, X = fit_classification_model(
            GradientBoostingClassifier(n_estimators=4, init='zero'), 4)
        model_onnx = convert_sklearn(
            model,
            "gradient boosting classifier",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingMultiClassClassifierInitZero",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_gradient_boosting_regressor_ls_loss(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="ls"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionLsLoss",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    def test_gradient_boosting_regressor_lad_loss(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="lad"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionLadLoss",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    def test_gradient_boosting_regressor_huber_loss(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="huber"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionHuberLoss",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    def test_gradient_boosting_regressor_quantile_loss(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="quantile"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionQuantileLoss",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    def test_gradient_boosting_regressor_int(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(random_state=42), is_int=True)
        model_onnx = convert_sklearn(
            model, "gradient boosting regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionInt-Dec3",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    def test_gradient_boosting_regressor_zero_init(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(n_estimators=30, init="zero",
                                      random_state=42))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionZeroInit-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    @unittest.skipIf(
        StrictVersion(__version__) <= StrictVersion(THRESHOLD),
        reason="Depends on PR #1015 onnxruntime.")
    def test_gradient_boosting_regressor_learning_rate(self):
        X, y = make_classification(
            n_features=100, n_samples=1000, n_classes=2, n_informative=8)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.5, random_state=42)
        model = GradientBoostingClassifier().fit(X_train, y_train)
        onnx_model = convert_sklearn(
            model, 'lr2', [('input', FloatTensorType(X_test.shape))])
        sess = InferenceSession(onnx_model.SerializeToString())
        res = sess.run(None, input_feed={'input': X_test.astype(np.float32)})
        r1 = np.mean(
            np.isclose(model.predict_proba(X_test),
                       list(map(lambda x: list(map(lambda y: x[y], x)),
                                res[1])), atol=1e-4))
        r2 = np.mean(res[0] == model.predict(X_test))
        assert r1 == r2

    def test_gradient_boosting_regressor_bool(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(random_state=42), is_bool=True)
        model_onnx = convert_sklearn(
            model, "gradient boosting regressor",
            [("input", BooleanTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGradientBoostingRegressorBool-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")


if __name__ == "__main__":
    unittest.main()
