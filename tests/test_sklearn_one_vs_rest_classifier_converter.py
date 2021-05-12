# SPDX-License-Identifier: Apache-2.0

from distutils.version import StrictVersion
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    onnx_built_with_ml)
from test_utils import (
    dump_data_and_model,
    dump_multiple_classification,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET)

warnings_to_skip = (DeprecationWarning, FutureWarning, ConvergenceWarning)


class TestOneVsRestClassifierConverter(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
            target_opset=TARGET_OPSET
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('1.4.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_rf(self):
        model = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=2, max_depth=2))
        model, X = fit_classification_model(
            model, 3, is_int=True, n_features=3)
        model_onnx = convert_sklearn(
            model, initial_types=[
                ('input', Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'zipmap': False}})

        sess = InferenceSession(model_onnx.SerializeToString())
        XI = X.astype(np.int64)
        got = sess.run(None, {'input': XI})
        exp_label = model.predict(XI)
        exp_proba = model.predict_proba(XI)
        assert_almost_equal(exp_proba, got[1], decimal=5)
        diff = np.abs(exp_label - got[0]).sum()
        if diff >= 3:
            # Both scikit-learn and onnxruntime do the computation
            # by parallelizing by trees. However, scikit-learn
            # always adds tree outputs in the same order,
            # onnxruntime does not. It may lead to small discrepencies.
            # This test ensures that probabilities are almost the same.
            # But a discrepencies around 0.5 may change the label.
            # That explains why the test allows less than 3 differences.
            assert_almost_equal(exp_label, got[0])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('1.3.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_rf_multilabel_float(self):
        for opset in [12, TARGET_OPSET]:
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                model = OneVsRestClassifier(
                    RandomForestClassifier(n_estimators=2, max_depth=3))
                model, X = fit_multilabel_classification_model(
                    model, 3, is_int=False, n_features=5)
                model_onnx = convert_sklearn(
                    model, initial_types=[
                        ('input', FloatTensorType([None, X.shape[1]]))],
                    target_opset=opset)
                dump_data_and_model(
                    X.astype(np.float32), model, model_onnx,
                    basename="SklearnOVRRFMultiLabelFloat%d" % opset)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('1.3.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_rf_multilabel_float_11(self):
        for opset in [9, 10, 11]:
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                model = OneVsRestClassifier(
                    RandomForestClassifier(n_estimators=2, max_depth=3))
                model, X = fit_multilabel_classification_model(
                    model, 3, is_int=False, n_features=5)
                model_onnx = convert_sklearn(
                    model, initial_types=[
                        ('input', FloatTensorType([None, X.shape[1]]))],
                    target_opset=opset)
                self.assertNotIn('"Clip"', str(model_onnx))
                dump_data_and_model(
                    X.astype(np.float32), model, model_onnx,
                    basename="SklearnOVRRFMultiLabelFloat%d" % opset)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('1.3.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_rf_multilabel_int(self):
        for opset in [12, TARGET_OPSET]:
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                model = OneVsRestClassifier(
                    RandomForestClassifier(n_estimators=2, max_depth=3))
                model, X = fit_multilabel_classification_model(
                    model, 3, is_int=True, n_features=5)
                model_onnx = convert_sklearn(
                    model, initial_types=[
                        ('input', Int64TensorType([None, X.shape[1]]))],
                    target_opset=opset)
                dump_data_and_model(
                    X.astype(np.int64), model, model_onnx,
                    basename="SklearnOVRRFMultiLabelInt64%d" % opset)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('1.3.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_rf_multilabel_int_11(self):
        for opset in [9, 10, 11]:
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                model = OneVsRestClassifier(
                    RandomForestClassifier(n_estimators=2, max_depth=3))
                model, X = fit_multilabel_classification_model(
                    model, 3, is_int=True, n_features=5)
                model_onnx = convert_sklearn(
                    model, initial_types=[
                        ('input', Int64TensorType([None, X.shape[1]]))],
                    target_opset=opset)
                self.assertNotIn('"Clip"', str(model_onnx))
                dump_data_and_model(
                    X.astype(np.int64), model, model_onnx,
                    basename="SklearnOVRRFMultiLabelInt64%d" % opset)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_02(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            first_class=2,
            suffix="F2",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
            target_opset=TARGET_OPSET
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_string(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            verbose=False,
            label_string=True,
            suffix="String",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
            target_opset=TARGET_OPSET
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_float(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression(solver='liblinear')), 3)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloat",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_decision_function(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 4)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationDecisionFunction",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function'],
        )
        if StrictVersion(ort_version) < StrictVersion("1.0.0"):
            return
        options = {id(model): {'raw_scores': True, 'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'input': X})[1]
        dec = model.decision_function(X)
        assert_almost_equal(got, dec, decimal=4)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_decision_function_binary(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 2)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationDecisionFunctionBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function_binary'],
        )
        if StrictVersion(ort_version) < StrictVersion("1.0.0"):
            return
        options = {id(model): {'raw_scores': True, 'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'input': X})[1]
        dec = model.decision_function(X)
        assert_almost_equal(got[:, 1], dec, decimal=4)
        assert_almost_equal(-got[:, 0], dec, decimal=4)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_int(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationInt",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_float_binary(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 2)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloatBin",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_float_binary_nozipmap(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 2)
        model_onnx = convert_sklearn(
            model, "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'zipmap': False}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnOVRClassificationFloatBinNoZipMap",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_int_binary(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationIntBin",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_float_mlp(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(MLPClassifier()), 4)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloatMLP",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_int_ensemble(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(GradientBoostingClassifier()), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationIntEnsemble",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_float_binary_ensemble(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(GradientBoostingClassifier()), 2)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloatBinEnsemble",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_classification_int_binary_mlp(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(MLPClassifier()), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationIntBinMLP",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_regression_float(self):
        """The test is unstable, some observations
        are equidistant to more than one class,
        the chosen is difficult to predict. So we
        check only probabilities."""
        rs = 11
        model, X = fit_classification_model(
            OneVsRestClassifier(
                LinearRegression()), 3, random_state=rs)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5],
            model,
            model_onnx,
            basename="SklearnOVRRegressionFloat-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_regression_int(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LinearRegression()), 10, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRRegressionInt-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_regression_float_mlp(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(MLPRegressor()), 5)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRRegressionFloatMLP-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_regression_int_ensemble(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(GradientBoostingRegressor()), 4, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRRegressionIntEnsemble-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.2.0"),
                     reason="fails to load the model")
    def test_ovr_raw_scores(self):
        X, y = make_classification(
            n_classes=2, n_samples=100, random_state=42,
            n_features=100, n_informative=7)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.5, random_state=42)
        model = OneVsRestClassifier(
            estimator=GradientBoostingClassifier(random_state=42))
        model.fit(X_train, y_train)

        options = {id(model): {'raw_scores': True, 'zipmap': False}}
        onnx_model = convert_sklearn(
            model, 'lr',
            [('input', FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        sess = InferenceSession(onnx_model.SerializeToString())
        res = sess.run(None, input_feed={'input': X_test.astype(np.float32)})
        exp = model.predict(X_test)
        assert_almost_equal(exp, res[0])
        exp = model.decision_function(X_test)
        assert_almost_equal(exp, res[1][:, 1], decimal=5)


if __name__ == "__main__":
    unittest.main()
