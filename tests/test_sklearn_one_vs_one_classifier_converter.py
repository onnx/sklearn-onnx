# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType
from test_utils import TARGET_OPSET

warnings_to_skip = (DeprecationWarning, FutureWarning, ConvergenceWarning)


ort_version = ".".join(ort_version.split(".")[:2])


class TestOneVsOneClassifierConverter(unittest.TestCase):
    def test_one_vs_one_classifier_converter_linearsvc(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, shuffle=True, random_state=0
        )
        model = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
        exp_label = model.predict(X_test[:10])
        exp_prob = model.decision_function(X_test[:10])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn OneVsOne Classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        XI = X_test[:10].astype(np.float32)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": XI})
        assert_almost_equal(exp_label.ravel(), got[0].ravel())
        assert_almost_equal(exp_prob, got[1])

    def test_one_vs_one_classifier_converter_logisticregression(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, shuffle=True, random_state=0
        )
        model = OneVsOneClassifier(LogisticRegression(random_state=0)).fit(
            X_train, y_train
        )
        exp_label = model.predict(X_test[:10])
        exp_prob = model.decision_function(X_test[:10])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn OneVsOne Classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        XI = X_test[:10].astype(np.float32)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": XI})
        assert_almost_equal(exp_label.ravel(), got[0].ravel())
        assert_almost_equal(exp_prob, got[1])

    def test_one_vs_one_classifier_converter_logisticregression_double(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, shuffle=True, random_state=0
        )
        model = OneVsOneClassifier(LogisticRegression(random_state=0)).fit(
            X_train, y_train
        )
        exp_label = model.predict(X_test[:10])
        exp_prob = model.decision_function(X_test[:10])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn OneVsOne Classifier",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        XI = X_test[:10].astype(np.float64)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": XI})
        assert_almost_equal(exp_label.ravel(), got[0].ravel())
        assert_almost_equal(exp_prob, got[1])

    def test_one_vs_one_classifier_converter_decisiontree(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, shuffle=True, random_state=0
        )
        model = OneVsOneClassifier(DecisionTreeClassifier(max_depth=3)).fit(
            X_train, y_train
        )
        limit = 10
        exp_label = model.predict(X_test[:limit])
        exp_prob = model.decision_function(X_test[:limit])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn OneVsOne Classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        XI = X_test[:limit].astype(np.float32)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": XI})
        assert_almost_equal(exp_label.ravel(), got[0].ravel())
        assert_almost_equal(exp_prob, got[1])


if __name__ == "__main__":
    unittest.main()
