from distutils.version import StrictVersion
import unittest
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_multilabel_classification_model,
)


def _sklearn_version():
    # Remove development version 0.22.dev0 becomes 0.22.
    v = ".".join(sklearn.__version__.split('.')[:2])
    return StrictVersion(v)


class TestGLMClassifierConverter(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionBinary",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_linear_discriminant_analysis(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [-2, -1]], dtype=np.float32)
        model = LinearDiscriminantAnalysis().fit(X, y)
        model_onnx = convert_sklearn(
            model, "linear model",
            [("input", FloatTensorType([None, X_test.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBin-Dec3",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_linear_discriminant_analysis_decfunc(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [0, 1]], dtype=np.float32)
        model = LinearDiscriminantAnalysis().fit(X, y)
        model_onnx = convert_sklearn(
            model, "linear model",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options={id(model): {'raw_scores': True}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBinRawScore-Out0",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function']
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_linear_discriminant_analysis_decfunc3(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 3])
        X_test = np.array([[-0.8, -1], [0, 1]], dtype=np.float32)
        model = LinearDiscriminantAnalysis().fit(X, y)
        model_onnx = convert_sklearn(
            model, "linear model",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options={id(model): {'raw_scores': True}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBinRawScore3-Out0",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function']
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_cv_binary_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegressionCV(), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticCVRegressionBinary",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class_nointercept(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(fit_intercept=False), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionBinaryNoIntercept",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 4)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMulti",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_nocl(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 4,
            label_string=True)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}})
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X, model, model_onnx, classes=model.classes_,
            basename="SklearnLogitisticRegressionMultiNoCl",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_ovr(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(multi_class='ovr'), 3)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMulti",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_multinomial(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                multi_class="multinomial", solver="lbfgs"), 4)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMulti",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_no_intercept(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(fit_intercept=False), 3)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiNoIntercept",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_lbfgs(self):
        penalty = (
            'l2'
            if _sklearn_version() < StrictVersion('0.21.0')
            else 'none')
        model, X = fit_classification_model(
            linear_model.LogisticRegression(solver='lbfgs', penalty=penalty),
            5)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiLbfgs",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_liblinear_l1(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(solver='liblinear', penalty='l1'),
            4)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiLiblinearL1",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_saga_elasticnet(self):
        if _sklearn_version() < StrictVersion('0.21.0'):
            model, X = fit_classification_model(
                linear_model.LogisticRegression(solver='saga'), 3)
        else:
            model, X = fit_classification_model(
                linear_model.LogisticRegression(
                    solver='saga', penalty='elasticnet', l1_ratio=0.1), 3)
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiSagaElasticnet",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_binary_class(self):
        model, X = fit_classification_model(LinearSVC(), 2)
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLinearSVCBinary-NoProb",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_multi_class(self):
        model, X = fit_classification_model(LinearSVC(), 5)
        model_onnx = convert_sklearn(
            model,
            "multi-class linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLinearSVCMulti",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_binary(self):
        model, X = fit_classification_model(linear_model.RidgeClassifier(), 2)
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnRidgeClassifierBin",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_binary_nozipmap(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 2)

        model_onnx = convert_sklearn(
            model, "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))])
        assert 'zipmap' in str(model_onnx).lower()

        options = {id(model): {'zipmap': True}}
        model_onnx = convert_sklearn(
            model, "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options)
        assert 'zipmap' in str(model_onnx).lower()

        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options)
        assert 'zipmap' not in str(model_onnx).lower()

        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnRidgeClassifierNZMBin",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_multi_class(self):
        model, X = fit_classification_model(linear_model.RidgeClassifier(), 5)
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnRidgeClassifierMulti",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_cv_binary(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 2)
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnRidgeClassifierCVBin",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_cv_multi_class(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 5)
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnRidgeClassifierCVMulti",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class_decision_function(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.LogisticRegression: {'raw_scores': True}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5],
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionBinaryRawScore",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function_binary'])

    @unittest.skip(
        reason="Scikit-learn doesn't return multi-label output.")
    def test_model_ridge_classifier_cv_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            linear_model.RidgeClassifierCV(random_state=42))
        model_onnx = convert_sklearn(
            model,
            "scikit-learn RidgeClassifierCV",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnRidgeClassifierCVMultiLabel",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
