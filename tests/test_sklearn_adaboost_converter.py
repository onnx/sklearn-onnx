# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
from onnx.defs import onnx_opset_version
from onnxruntime import __version__ as ort_version
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_regression_model,
    TARGET_OPSET
)


ort_version = '.'.join(ort_version.split('.')[:2])


class TestSklearnAdaBoostModels(unittest.TestCase):
    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_classifier_samme_r(self):
        model, X_test = fit_classification_model(AdaBoostClassifier(
            n_estimators=10, algorithm="SAMME.R", random_state=42,
            base_estimator=DecisionTreeClassifier(
                max_depth=2, random_state=42)), 3)
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=10
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierSAMMER")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_classifier_samme_r_decision_function(self):
        model, X_test = fit_classification_model(AdaBoostClassifier(
            n_estimators=10, algorithm="SAMME.R", random_state=42,
            base_estimator=DecisionTreeClassifier(
                max_depth=2, random_state=42)), 4)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=10,
            options=options,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierSAMMERDecisionFunction",
            methods=['predict', 'decision_function'])

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_classifier_samme_r_logreg(self):
        model, X_test = fit_classification_model(AdaBoostClassifier(
            n_estimators=5, algorithm="SAMME.R",
            base_estimator=LogisticRegression(
                solver='liblinear')), 4)
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=10
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierSAMMERLogReg")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_classifier_samme(self):
        model, X_test = fit_classification_model(AdaBoostClassifier(
            n_estimators=5, algorithm="SAMME", random_state=42,
            base_estimator=DecisionTreeClassifier(
                max_depth=6, random_state=42)), 2, n_features=7)
        model_onnx = convert_sklearn(
            model,
            "AdaBoostClSamme",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=10,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierSAMMEDT")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_classifier_samme_decision_function(self):
        model, X_test = fit_classification_model(AdaBoostClassifier(
            n_estimators=5, algorithm="SAMME", random_state=42,
            base_estimator=DecisionTreeClassifier(
                max_depth=6, random_state=42)), 2)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "AdaBoostClSamme",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=10,
            options=options,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierSAMMEDTDecisionFunction",
            methods=['predict', 'decision_function_binary'],
        )

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_classifier_lr(self):
        model, X_test = fit_classification_model(
            AdaBoostClassifier(learning_rate=0.3, random_state=42), 3,
            is_int=True)
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", Int64TensorType((None, X_test.shape[1])))],
            target_opset=10
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierLR")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_classifier_bool(self):
        model, X_test = fit_classification_model(
            AdaBoostClassifier(random_state=42), 3,
            is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "AdaBoost classification",
            [("input", BooleanTensorType((None, X_test.shape[1])))],
            target_opset=10,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnAdaBoostClassifierBool")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_regressor(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(n_estimators=5))
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=10)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressor-Dec4")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_regressor_lreg(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(n_estimators=5,
                              base_estimator=LinearRegression()))
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=10)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressorLReg-Dec4")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_regressor_int(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(n_estimators=5), is_int=True)
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=10)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressorInt-Dec4")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_regressor_lr10(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(learning_rate=0.5, random_state=42))
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=10)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressorLR-Dec4")

    @unittest.skipIf((StrictVersion(ort_version) <
                      StrictVersion("0.5.9999")),
                     reason="not available")
    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_regressor_lr11(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(learning_rate=0.5, random_state=42))
        if onnx_opset_version() < 11:
            try:
                convert_sklearn(
                    model, "AdaBoost regression",
                    [("input", FloatTensorType([None, X.shape[1]]))])
            except RuntimeError:
                return
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressorLR-Dec4")

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_ada_boost_regressor_bool(self):
        model, X = fit_regression_model(
            AdaBoostRegressor(learning_rate=0.5, random_state=42),
            is_bool=True)
        model_onnx = convert_sklearn(
            model, "AdaBoost regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=10,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnAdaBoostRegressorBool")


if __name__ == "__main__":
    unittest.main()
