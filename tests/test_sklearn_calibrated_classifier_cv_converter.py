"""
Tests scikit-learn's CalibratedClassifierCV converters
"""

import unittest

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits, load_iris
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model


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
            "scikit-learn CalibratedClassifierCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
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
    def test_model_calibrated_classifier_cv_int(self):
        data = load_digits()
        X, y = data.data, data.target
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCV",
            [("input", Int64TensorType([None, X.shape[1]]))],
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
    def test_model_calibrated_classifier_cv_isotonic_float(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = KNeighborsClassifier().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="isotonic").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVIsotonicFloat-OneOffArray",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )

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
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
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
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVIsotonicBinary-OneOffArray",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )


if __name__ == "__main__":
    unittest.main()
