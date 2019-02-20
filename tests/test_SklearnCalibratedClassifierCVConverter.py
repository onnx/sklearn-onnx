"""
Tests scikit-learn's CalibratedClassifierCV converters
"""

import numpy as np
import unittest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits, load_iris
from sklearn.svm import LinearSVC

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnCalibratedClassifierCVConverters(unittest.TestCase):

    def test_model_calibrated_classifier_cv_float(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = LinearSVC(C=0.001).fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method='sigmoid').fit(X, y)
        model_onnx = convert_sklearn(model, 'scikit-learn CalibratedClassifierCV',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                            basename="SklearnCalibratedClassifierCVFloat")

    def test_model_calibrated_classifier_cv_int(self):
        data = load_digits()
        X, y = data.data, data.target
        clf = LinearSVC(C=0.001).fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method='sigmoid').fit(X, y)
        model_onnx = convert_sklearn(model, 'scikit-learn CalibratedClassifierCV',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                            basename="SklearnCalibratedClassifierCVInt")


if __name__ == "__main__":
    unittest.main()
