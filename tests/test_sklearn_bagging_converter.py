# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import onnxruntime
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType)
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_regression_model,
    TARGET_OPSET)


class TestSklearnBaggingConverter(unittest.TestCase):
    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_default_binary_int(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBaggingClassifierDefaultBinary")

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_default_multiclass_int(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 4, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierDefaultMulticlass",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_default_binary(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 2)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierDefaultBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_max_features(self):
        model, X = fit_classification_model(
            BaggingClassifier(max_features=0.5), 2)
        model_onnx = convert_sklearn(
            model, "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnBaggingClassifierMaxFeatures",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_bootstrap_features(self):
        model, X = fit_classification_model(
            BaggingClassifier(bootstrap_features=True), 2)
        model_onnx = convert_sklearn(
            model, "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5], model, model_onnx, verbose=False,
            basename="SklearnBaggingClassifierBootstrapFeatures",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_default_binary_nozipmap(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 2)
        model_onnx = convert_sklearn(
            model, "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'zipmap': False}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBaggingClassifierDefaultBinaryNoZipMap",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_default_multiclass(self):
        model, X = fit_classification_model(
            BaggingClassifier(), 4)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierDefaultMulticlass",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_sgd_binary(self):
        model, X = fit_classification_model(
            BaggingClassifier(
                SGDClassifier(loss='modified_huber', random_state=42),
                random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierSGDBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_sgd_binary_decision_function(self):
        model, X = fit_classification_model(
            BaggingClassifier(SGDClassifier(random_state=42),
                              random_state=42), 2)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5], model, model_onnx,
            basename="SklearnBaggingClassifierSGDBinaryDecisionFunction-Dec3",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function_binary'])

    @unittest.skipIf(StrictVersion(onnxruntime.__version__)
                     <= StrictVersion("0.4.0"),
                     reason="Not implemented.")
    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_sgd_multiclass(self):
        model, X = fit_classification_model(
            BaggingClassifier(
                SGDClassifier(loss='modified_huber', random_state=42),
                random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5], model, model_onnx,
            basename="SklearnBaggingClassifierSGDMulticlass-Dec3",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_sgd_multiclass_decision_function(self):
        model, X = fit_classification_model(
            BaggingClassifier(
                GradientBoostingClassifier(random_state=42, n_estimators=4),
                random_state=42), 4, n_features=10)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model, "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:15], model, model_onnx,
            basename="SklearnBaggingClassifierSGDMultiDecisionFunction-Dec3",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function'])

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_gradient_boosting_binary(self):
        model, X = fit_classification_model(
            BaggingClassifier(
                GradientBoostingClassifier(n_estimators=10)), 2)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierGradientBoostingBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_classifier_gradient_boosting_multiclass(self):
        model, X = fit_classification_model(
            BaggingClassifier(
                GradientBoostingClassifier(n_estimators=10)), 3)
        model_onnx = convert_sklearn(
            model,
            "bagging classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingClassifierGradientBoostingMulticlass",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_regressor_default(self):
        model, X = fit_regression_model(
            BaggingRegressor())
        model_onnx = convert_sklearn(
            model,
            "bagging regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingRegressorDefault-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_regressor_max_features(self):
        model, X = fit_regression_model(
            BaggingRegressor(max_features=0.5, n_estimators=3))
        model_onnx = convert_sklearn(
            model, "bagging regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnBaggingRegressorMaxFeatures-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    def test_bagging_regressor_bootstrap_features(self):
        model, X = fit_regression_model(
            BaggingRegressor(bootstrap_features=False))
        model_onnx = convert_sklearn(
            model, "bagging regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnBaggingRegressorBootstrapFeatures-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @ignore_warnings(category=FutureWarning)
    def test_bagging_regressor_sgd(self):
        model, X = fit_regression_model(
            BaggingRegressor(SGDRegressor()))
        model_onnx = convert_sklearn(
            model, "bagging regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingRegressorSGD-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=FutureWarning)
    def test_bagging_regressor_gradient_boosting(self):
        model, X = fit_regression_model(
            BaggingRegressor(
                GradientBoostingRegressor(n_estimators=10)))
        model_onnx = convert_sklearn(
            model, "bagging regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBaggingRegressorGradientBoosting-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @ignore_warnings(category=FutureWarning)
    def test_bagging_regressor_bool(self):
        model, X = fit_regression_model(
            BaggingRegressor(), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "bagging regressor",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBaggingRegressorBool-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
