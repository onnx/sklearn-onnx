# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's CalibratedClassifierCV converters
"""

import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import onnxruntime
try:
    from skl2onnx.common._apply_operation import apply_less
except ImportError:
    # onnxconverter-common is too old
    apply_less = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, onnx_built_with_ml)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnCalibratedClassifierCVConverters(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_calibrated_classifier_cv_float(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCVMNB",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVFloat",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_calibrated_classifier_cv_float_nozipmap(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model, "scikit-learn CalibratedClassifierCVMNB",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'zipmap': False}})
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnCalibratedClassifierCVFloatNoZipMap",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_calibrated_classifier_cv_int(self):
        data = load_digits()
        X, y = data.data, data.target
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCVMNB",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_calibrated_classifier_cv_isotonic_float(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = KNeighborsClassifier().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="isotonic").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCVKNN",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        try:
            dump_data_and_model(
                X.astype(np.float32),
                model,
                model_onnx,
                basename="SklearnCalibratedClassifierCVIsotonicFloat")
        except Exception as e:
            raise AssertionError("Issue with model\n{}".format(
                model_onnx)) from e

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_calibrated_classifier_cv_binary(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVBinaryMNB",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_calibrated_classifier_cv_isotonic_binary(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        clf = KNeighborsClassifier().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="isotonic").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVIsotonicBinaryKNN")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_calibrated_classifier_cv_logistic_regression(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        model = CalibratedClassifierCV(
            base_estimator=LogisticRegression(), method='sigmoid').fit(X, y)
        model_onnx = convert_sklearn(
            model, "unused",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVBinaryLogReg",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_calibrated_classifier_cv_rf(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        model = CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(n_estimators=2),
            method='sigmoid').fit(X, y)
        try:
            convert_sklearn(
                model, "unused",
                [("input", FloatTensorType([None, X.shape[1]]))])
            raise AssertionError(
                "RandomForestClassifier has no decision_function")
        except RuntimeError as e:
            assert "cannot implement decision_function" in str(e)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    @unittest.skipIf(apply_less is None, reason="onnxconverter-common old")
    def test_model_calibrated_classifier_cv_svc(self):
        data = load_iris()
        X, y = data.data, data.target
        model = CalibratedClassifierCV(
            base_estimator=SVC(),
            method='sigmoid').fit(X, y)
        model_onnx = convert_sklearn(
            model, "unused",
            [("input", FloatTensorType([None, X.shape[1]]))])
        assert model_onnx is not None


if __name__ == "__main__":
    unittest.main()
