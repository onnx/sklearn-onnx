# SPDX-License-Identifier: Apache-2.0

from distutils.version import StrictVersion
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
import sklearn
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from onnxruntime import InferenceSession, __version__ as ort_version
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
    fit_multilabel_classification_model,
    TARGET_OPSET)


ort_version = ort_version.split('+')[0]


def _sklearn_version():
    # Remove development version 0.22.dev0 becomes 0.22.
    v = ".".join(sklearn.__version__.split('.')[:2])
    return StrictVersion(v)


class TestGLMClassifierConverter(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionBinary")
        if StrictVersion(ort_version) >= StrictVersion("1.0.0"):
            sess = InferenceSession(model_onnx.SerializeToString())
            out = sess.get_outputs()
            lb = out[0].type
            sh = out[0].shape
            self.assertEqual(str(lb), "tensor(int64)")
            self.assertEqual(sh, [None])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class_blacklist(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            black_op={'LinearClassifier'})
        self.assertNotIn('LinearClassifier', str(model_onnx))
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionBinaryBlackList")
        if StrictVersion(ort_version) >= StrictVersion("1.0.0"):
            sess = InferenceSession(model_onnx.SerializeToString())
            out = sess.get_outputs()
            lb = out[0].type
            sh = out[0].shape
            self.assertEqual(str(lb), "tensor(int64)")
            self.assertEqual(sh, [None])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class_string(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 2,
            label_string=True)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionBinary")
        if StrictVersion(ort_version) >= StrictVersion("1.0.0"):
            sess = InferenceSession(model_onnx.SerializeToString())
            out = sess.get_outputs()
            lb = out[0].type
            sh = out[0].shape
            self.assertEqual(str(lb), "tensor(string)")
            self.assertEqual(sh, [None])

    def test_model_logistic_regression_int(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 3, is_int=True)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionInt")

    def test_model_logistic_regression_bool(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=100), 3, is_bool=True)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionBool")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_linear_discriminant_analysis(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [-2, -1]], dtype=np.float32)
        model = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
        model_onnx = convert_sklearn(
            model, "linear model",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBin-Dec3")

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
            options={id(model): {'raw_scores': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBinRawScore-Out0",
            methods=['predict', 'decision_function'])

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
            options={id(model): {'raw_scores': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnLinearDiscriminantAnalysisBinRawScore3-Out0",
            methods=['predict', 'decision_function'])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_cv_binary_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegressionCV(max_iter=100), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticCVRegressionBinary")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_cv_int(self):
        try:
            model, X = fit_classification_model(
                linear_model.LogisticRegressionCV(max_iter=100),
                7, is_int=True)
        except AttributeError:
            # AttributeError: 'str' object has no attribute 'decode'
            # Bug fixed in scikit-learn 0.24 due to a warning using encoding.
            return
        model_onnx = convert_sklearn(
            model, "logistic regression cv",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionCVInt")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_cv_bool(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegressionCV(max_iter=100), 3, is_bool=True)
        model_onnx = convert_sklearn(
            model, "logistic regression cv",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionCVBool")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class_nointercept(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                fit_intercept=False, max_iter=10000), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionBinaryNoIntercept")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 4)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_nocl(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 4,
            label_string=True)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X, model, model_onnx, classes=model.classes_,
            basename="SklearnLogitisticRegressionMultiNoCl")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_ovr(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                multi_class='ovr', max_iter=10000), 3)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_multinomial(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=10000), 4)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_no_intercept(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                fit_intercept=False, max_iter=10000), 3)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionMultiNoIntercept")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_lbfgs(self):
        penalty = (
            'l2' if _sklearn_version() < StrictVersion('0.21.0')
            else 'none')
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                solver='lbfgs', penalty=penalty, max_iter=10000), 5)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionMultiLbfgs")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_liblinear_l1(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(
                solver='liblinear', penalty='l1', max_iter=10000), 4)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionMultiLiblinearL1")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_saga_elasticnet(self):
        if _sklearn_version() < StrictVersion('0.21.0'):
            model, X = fit_classification_model(
                linear_model.LogisticRegression(
                    solver='saga', max_iter=10000), 3)
        else:
            model, X = fit_classification_model(
                linear_model.LogisticRegression(
                    solver='saga', penalty='elasticnet', l1_ratio=0.1,
                    max_iter=10000), 3)
        model_onnx = convert_sklearn(
            model, "multi-class logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLogitisticRegressionMultiSagaElasticnet")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_binary_class(self):
        model, X = fit_classification_model(LinearSVC(max_iter=10000), 2)
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSVCBinary-NoProb")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_multi_class(self):
        model, X = fit_classification_model(LinearSVC(max_iter=100), 5)
        model_onnx = convert_sklearn(
            model, "multi-class linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSVCMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_int(self):
        model, X = fit_classification_model(
            LinearSVC(max_iter=100), 5, is_int=True)
        model_onnx = convert_sklearn(
            model, "multi-class linear SVC",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSVCInt")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_bool(self):
        model, X = fit_classification_model(
            LinearSVC(max_iter=100), 5, is_bool=True)
        model_onnx = convert_sklearn(
            model, "multi-class linear SVC",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSVCBool")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_binary(self):
        model, X = fit_classification_model(linear_model.RidgeClassifier(), 2)
        model_onnx = convert_sklearn(
            model, "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierBin")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_binary_nozipmap(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2)

        model_onnx = convert_sklearn(
            model, "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        assert 'zipmap' in str(model_onnx).lower()

        options = {id(model): {'zipmap': True}}
        model_onnx = convert_sklearn(
            model, "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        assert 'zipmap' in str(model_onnx).lower()

        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "binary ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        assert 'zipmap' not in str(model_onnx).lower()

        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierNZMBin")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_binary_mispelled_zipmap(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2)

        options = {id(model): {'zipmap ': True}}
        try:
            convert_sklearn(
                model, "binary ridge classifier",
                [("input", FloatTensorType([None, X.shape[1]]))],
                options=options, target_opset=TARGET_OPSET)
            raise AssertionError("Expecting an error.")
        except NameError as e:
            assert "Option 'zipmap ' not in" in str(e)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_binary_mispelled_zipmap_wrong_value(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2)

        options = {id(model): {'zipmap': 'True'}}
        try:
            convert_sklearn(
                model, "binary ridge classifier",
                [("input", FloatTensorType([None, X.shape[1]]))],
                options=options, target_opset=TARGET_OPSET)
            raise AssertionError("Expecting an error.")
        except ValueError as e:
            assert "Unexpected value ['True'] for option 'zipmap'" in str(e)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_multi_class(self):
        model, X = fit_classification_model(linear_model.RidgeClassifier(), 5)
        model_onnx = convert_sklearn(
            model, "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_int(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifier(), 5, is_int=True)
        model_onnx = convert_sklearn(
            model, "multi-class ridge classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierInt")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_bool(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifier(), 4, is_bool=True)
        model_onnx = convert_sklearn(
            model, "multi-class ridge classifier",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierBool")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_cv_binary(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 2)
        model_onnx = convert_sklearn(
            model, "binary ridge classifier cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierCVBin")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_cv_int(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 2, is_int=True)
        model_onnx = convert_sklearn(
            model, "binary ridge classifier cv",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierCVInt")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_cv_bool(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 2, is_bool=True)
        model_onnx = convert_sklearn(
            model, "binary ridge classifier cv",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierCVBool")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_ridge_classifier_cv_multi_class(self):
        model, X = fit_classification_model(
            linear_model.RidgeClassifierCV(), 5)
        model_onnx = convert_sklearn(
            model, "multi-class ridge classifier cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRidgeClassifierCVMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class_decision_function(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(max_iter=10000), 2)
        model_onnx = convert_sklearn(
            model, "logistic regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.LogisticRegression: {'raw_scores': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5], model, model_onnx,
            basename="SklearnLogitisticRegressionBinaryRawScore",
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
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnRidgeClassifierCVMultiLabel")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion('1.6'),
                     reason="Requires onnx 1.6")
    def test_model_classifier_multi_zipmap_columns(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 3,
            n_features=4, label_string=True)
        model_onnx = convert_sklearn(
            model, "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.LogisticRegression: {'zipmap': 'columns'}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sess = InferenceSession(model_onnx.SerializeToString())
        names = [_.name for _ in sess.get_outputs()]
        self.assertEqual(['output_label', 'scl0', 'scl1', 'scl2'], names)
        xt = X[:10].astype(np.float32)
        got = sess.run(None, {'input': xt})
        prob = model.predict_proba(xt)
        for i in range(prob.shape[1]):
            assert_almost_equal(prob[:, i], got[i+1])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion('1.6'),
                     reason="Requires onnx 1.6")
    def test_model_classifier_multi_class_string_zipmap_columns(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 3,
            n_features=4, label_string=False)
        model_onnx = convert_sklearn(
            model, "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.LogisticRegression: {'zipmap': 'columns'}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sess = InferenceSession(model_onnx.SerializeToString())
        names = [_.name for _ in sess.get_outputs()]
        self.assertEqual(['output_label', 'i0', 'i1', 'i2'], names)
        xt = X[:10].astype(np.float32)
        got = sess.run(None, {'input': xt})
        prob = model.predict_proba(xt)
        for i in range(prob.shape[1]):
            assert_almost_equal(prob[:, i], got[i+1])


if __name__ == "__main__":
    unittest.main()
