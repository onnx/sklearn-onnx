# SPDX-License-Identifier: Apache-2.0

import packaging.version as pv
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sklearn
from sklearn import linear_model, __version__ as sklearn_version
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from onnxruntime import __version__ as ort_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET,
    InferenceSessionEx as InferenceSession,
)


ort_version = ort_version.split("+")[0]
skl_version = ".".join(sklearn_version.split(".")[:2])


def _sklearn_version():
    # Remove development version 0.22.dev0 becomes 0.22.
    v = ".".join(sklearn.__version__.split(".")[:2])
    return pv.Version(v)


def skl12():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("1.2")


class TestGLMClassifierConverter(unittest.TestCase):
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_binary_class_boolean(self):
        X = np.array(
            [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
        )
        y = np.array([True, True, True, False, False, False])
        model = linear_model.LogisticRegression(max_iter=100).fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "linear model",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {"zipmap": False}},
            target_opset=TARGET_OPSET,
        )
        self.assertIn('name: "classlabels_ints"', str(model_onnx))
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionBinaryBoolean"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_binary_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 2
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionBinary"
        )
        if pv.Version(ort_version) >= pv.Version("1.0.0"):
            sess = InferenceSession(model_onnx.SerializeToString())
            out = sess.get_outputs()
            lb = out[0].type
            sh = out[0].shape
            self.assertEqual(str(lb), "tensor(int64)")
            self.assertEqual(sh, [None])

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_binary_class_blacklist(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 2
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            black_op={"LinearClassifier"},
        )
        self.assertNotIn("LinearClassifier", str(model_onnx))
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionBinaryBlackList"
        )
        if pv.Version(ort_version) >= pv.Version("1.0.0"):
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            out = sess.get_outputs()
            lb = out[0].type
            sh = out[0].shape
            self.assertEqual(str(lb), "tensor(int64)")
            self.assertEqual(sh, [None])

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_binary_class_string(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 2, label_string=True
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionBinary"
        )
        if pv.Version(ort_version) >= pv.Version("1.0.0"):
            sess = InferenceSession(model_onnx.SerializeToString())
            out = sess.get_outputs()
            lb = out[0].type
            sh = out[0].shape
            self.assertEqual(str(lb), "tensor(string)")
            self.assertEqual(sh, [None])

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_int(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 3, is_int=True
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionInt"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_bool(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 3, is_bool=True
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionBool"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_linear_discriminant_analysis(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [-2, -1]], dtype=np.float32)
        model = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "linear model",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBin-Dec3",
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_linear_discriminant_analysis_decfunc(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [0, 1]], dtype=np.float32)
        model = LinearDiscriminantAnalysis().fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "linear model",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options={id(model): {"raw_scores": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBinRawScore-Out0",
            methods=["predict", "decision_function"],
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_linear_discriminant_analysis_decfunc3(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 3])
        X_test = np.array([[-0.8, -1], [0, 1]], dtype=np.float32)
        model = LinearDiscriminantAnalysis().fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "linear model",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options={id(model): {"raw_scores": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBinRawScore3-Out0",
            methods=["predict", "decision_function"],
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_cv_binary_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegressionCV(max_iter=100), 2
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticCVRegressionBinary"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_cv_int(self):
        try:
            model, X = fit_classification_model(
                linear_model.LogisticRegressionCV(max_iter=100), 7, is_int=True
            )
        except AttributeError:
            # AttributeError: 'str' object has no attribute 'decode'
            # Bug fixed in scikit-learn 0.24 due to a warning using encoding.
            return
        model_onnx = convert_sklearn(
            model,
            "logistic regression cv",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionCVInt"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_cv_bool(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegressionCV(max_iter=100), 3, is_bool=True
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression cv",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionCVBool"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_binary_class_nointercept(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(fit_intercept=False, max_iter=10000), 2
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionBinaryNoIntercept",
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 4
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionMulti"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class_nocl(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 4, label_string=True
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {"nocl": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert "classlabels_strings" not in sonx
        assert "cl0" not in sonx
        dump_data_and_model(
            X,
            model,
            model_onnx,
            classes=model.classes_,
            basename="SklearnLogitisticRegressionMultiNoCl",
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class_ovr(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(multi_class="ovr", max_iter=10000), 3
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionMulti"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class_multinomial(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                multi_class="multinomial", solver="lbfgs", max_iter=10000
            ),
            4,
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionMulti"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class_no_intercept(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(fit_intercept=False, max_iter=10000), 3
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionMultiNoIntercept"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class_lbfgs(self):
        penalty = "l2"
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                solver="lbfgs", penalty=penalty, max_iter=10000
            ),
            5,
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionMultiLbfgs"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class_liblinear_l1(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                solver="liblinear", penalty="l1", max_iter=10000
            ),
            4,
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLogitisticRegressionMultiLiblinearL1"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_multi_class_saga_elasticnet(self):
        if _sklearn_version() < pv.Version("0.21.0"):
            model, X = fit_classification_model(
                linear_model.LogisticRegression(solver="saga", max_iter=10000), 3
            )
        else:
            model, X = fit_classification_model(
                linear_model.LogisticRegression(
                    solver="saga", penalty="elasticnet", l1_ratio=0.1, max_iter=10000
                ),
                3,
            )
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiSagaElasticnet",
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    @unittest.skipIf(not skl12(), reason="sparse_output")
    def test_model_linear_svc_binary_class(self):
        model, X = fit_classification_model(LinearSVC(max_iter=10000), 2)
        model_onnx = convert_sklearn(
            model,
            "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLinearSVCBinary-NoProb"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_linear_svc_multi_class(self):
        model, X = fit_classification_model(LinearSVC(max_iter=100), 5)
        model_onnx = convert_sklearn(
            model,
            "multi-class linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLinearSVCMulti")

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_linear_svc_int(self):
        model, X = fit_classification_model(LinearSVC(max_iter=100), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "multi-class linear SVC",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLinearSVCInt")

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_linear_svc_bool(self):
        model, X = fit_classification_model(LinearSVC(max_iter=100), 5, is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "multi-class linear SVC",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLinearSVCBool")

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_binary(self):
        model, X = fit_classification_model(linear_model.RidgeClassifier(), 2)
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRidgeClassifierBin")

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_binary_nozipmap(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2
        )

        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" in str(model_onnx).lower()

        options = {id(model): {"zipmap": True}}
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" in str(model_onnx).lower()

        options = {id(model): {"zipmap": False}}
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" not in str(model_onnx).lower()

        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeClassifierNZMBin"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_binary_mispelled_zipmap(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2
        )

        options = {id(model): {"zipmap ": True}}
        try:
            convert_sklearn(
                model,
                "binary ridge classifier",
                [("input", FloatTensorType([None, X.shape[1]]))],
                options=options,
                target_opset=TARGET_OPSET,
            )
            raise AssertionError("Expecting an error.")
        except NameError as e:
            assert "Option 'zipmap ' not in" in str(e)

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_binary_mispelled_zipmap_wrong_value(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2
        )

        options = {id(model): {"zipmap": "True"}}
        try:
            convert_sklearn(
                model,
                "binary ridge classifier",
                [("input", FloatTensorType([None, X.shape[1]]))],
                options=options,
                target_opset=TARGET_OPSET,
            )
            raise AssertionError("Expecting an error.")
        except ValueError as e:
            assert "Unexpected value ['True'] for option 'zipmap'" in str(e)

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_multi_class(self):
        model, X = fit_classification_model(linear_model.RidgeClassifier(), 5)
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeClassifierMulti"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_int(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifier(), 5, is_int=True
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRidgeClassifierInt")

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_bool(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifier(), 4, is_bool=True
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRidgeClassifierBool")

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_cv_binary(self):
        model, X = fit_classification_model(linear_model.RidgeClassifierCV(), 2)
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeClassifierCVBin"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_cv_int(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 2, is_int=True
        )
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier cv",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeClassifierCVInt"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_cv_bool(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 2, is_bool=True
        )
        model_onnx = convert_sklearn(
            model,
            "binary ridge classifier cv",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeClassifierCVBool"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_cv_multi_class(self):
        model, X = fit_classification_model(linear_model.RidgeClassifierCV(), 5)
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeClassifierCVMulti"
        )

    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_logistic_regression_binary_class_decision_function(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2
        )
        model_onnx = convert_sklearn(
            model,
            "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.LogisticRegression: {"raw_scores": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5],
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionBinaryRawScore",
            methods=["predict", "decision_function_binary"],
        )

    @unittest.skip(reason="Scikit-learn doesn't return multi-label output.")
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_classifier_cv_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            linear_model.RidgeClassifierCV(random_state=42)
        )
        model_onnx = convert_sklearn(
            model,
            "scikit-learn RidgeClassifierCV",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnRidgeClassifierCVMultiLabel"
        )

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_classifier_multi_zipmap_columns(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 3, n_features=4, label_string=True
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.LogisticRegression: {"zipmap": "columns"}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        sess = InferenceSession(model_onnx.SerializeToString())
        if sess is None:
            return
        names = [_.name for _ in sess.get_outputs()]
        self.assertEqual(["output_label", "scl0", "scl1", "scl2"], names)
        xt = X[:10].astype(np.float32)
        got = sess.run(None, {"input": xt})
        prob = model.predict_proba(xt)
        for i in range(prob.shape[1]):
            assert_almost_equal(prob[:, i], got[i + 1])

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_classifier_multi_class_string_zipmap_columns(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 3, n_features=4, label_string=False
        )
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.LogisticRegression: {"zipmap": "columns"}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        sess = InferenceSession(model_onnx.SerializeToString())
        if sess is None:
            return
        names = [_.name for _ in sess.get_outputs()]
        self.assertEqual(["output_label", "i0", "i1", "i2"], names)
        xt = X[:10].astype(np.float32)
        got = sess.run(None, {"input": xt})
        prob = model.predict_proba(xt)
        for i in range(prob.shape[1]):
            assert_almost_equal(prob[:, i], got[i + 1])


if __name__ == "__main__":
    unittest.main()
