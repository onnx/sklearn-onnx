# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import onnx
from onnx.defs import onnx_opset_version
import onnxruntime
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_regression_model,
    TARGET_OPSET
)


class TestSklearnAdaBoostModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostClassifierSAMMER",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function'],
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostClassifierSAMMERLogReg",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostClassifierSAMMEDT",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "< StrictVersion('0.5.0')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "< StrictVersion('0.5.0')",
            methods=['predict', 'decision_function_binary'],
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostClassifierLR",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostClassifierBool",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "< StrictVersion('0.5.0') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostRegressorLReg-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "< StrictVersion('0.5.0') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostRegressorInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "< StrictVersion('0.5.0') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostRegressorLR-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "< StrictVersion('0.5.0') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
            verbose=False
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnxruntime.__version__) <
                      StrictVersion("0.5.9999")),
                     reason="not available")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostRegressorLR-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "< StrictVersion('0.5.9999') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
            verbose=False
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.5.0")),
                     reason="not available")
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
            basename="SklearnAdaBoostRegressorBool",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "< StrictVersion('0.5.0') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
            verbose=False,
        )


if __name__ == "__main__":
    unittest.main()
