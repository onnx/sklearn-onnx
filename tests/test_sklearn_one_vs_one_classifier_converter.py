# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType)
from test_utils import (
    dump_data_and_model,
    dump_multiple_classification,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET)

warnings_to_skip = (DeprecationWarning, FutureWarning, ConvergenceWarning)


ort_version = '.'.join(ort_version.split('.')[:2])


class TestOneVsOneClassifierConverter(unittest.TestCase):

    def test_one_vs_one_classifier_converter_linearsvc(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
        model = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
        exp_label = model.predict(X_test[:10])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn OneVsOne Classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        XI = X_test[:10].astype(np.float32)

        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'input': XI})
        assert_almost_equal(exp_label.ravel(), got[0].ravel())

    def test_one_vs_one_classifier_converter_logisticregression(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
        model = OneVsOneClassifier(LogisticRegression(random_state=0)).fit(X_train, y_train)
        exp_label = model.predict(X_test[:10])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn OneVsOne Classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        XI = X_test[:10].astype(np.float32)

        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'input': XI})
        assert_almost_equal(exp_label.ravel(), got[0].ravel())


if __name__ == "__main__":
    # TestOneVsOneClassifierConverter().test_one_vs_one_classifier_converter_logisticregression()
    unittest.main()
