"""
Tests scikit-learn's CalibratedClassifierCV converters
"""

import numpy as np
import unittest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits, load_iris
from sklearn.svm import LinearSVC

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnCalibratedClassifierCVConverters(unittest.TestCase):

    def test_model_calibrated_classifier_cv_float(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = LinearSVC(C=0.001).fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method='sigmoid').fit(X, y)
        model_onnx = to_onnx(model, 'scikit-learn CalibratedClassifierCV',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                        basename="SklearnCalibratedClassifierCVFloat",
                        allow_failure="StrictVersion(onnxruntime.__version__)"
                                       "<= StrictVersion('0.2.1')")

    def test_model_calibrated_classifier_cv_int(self):
        data = load_digits()
        X, y = data.data, data.target
        clf = LinearSVC(C=0.001).fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method='sigmoid').fit(X, y)
        model_onnx = to_onnx(model, 'scikit-learn CalibratedClassifierCV',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                        basename="SklearnCalibratedClassifierCVInt",
                        allow_failure="StrictVersion(onnxruntime.__version__)"
                                       "<= StrictVersion('0.2.1')")


if __name__ == "__main__":
    unittest.main()
