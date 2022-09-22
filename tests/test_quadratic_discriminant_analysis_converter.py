# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import unittest
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from onnxruntime import __version__ as ort_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
)

from tests.test_utils import (
    dump_data_and_model,
    TARGET_OPSET
)

ort_version = ".".join(ort_version.split(".")[:2])


class TestQuadraticDiscriminantAnalysisConverter(unittest.TestCase):
    def test_model_qda_svm_2c2f(self):
        # 2 classes, 2 features
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [0.8, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)
        print(skl_model.predict(X_test))
        print(skl_model.predict_proba(X_test))

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(X_test.astype(np.float32), skl_model, onnx_model,
                            basename="SklearnQDA_2c2f")

    def test_model_qda_svm_2c3f(self):
        # 2 classes, 3 features
        X = np.array([[-1, -1, 0], [-2, -1, 1], [-3, -2, 0],
                     [1, 1, 0], [2, 1, 1], [3, 2, 1]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1, 0], [-1, -1.6, 0],
                          [1, 1.5, 1], [3.1, 2.1, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)
        print(skl_model.predict(X_test))
        print(skl_model.predict_proba(X_test))

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(X_test.astype(np.float32), skl_model, onnx_model,
                            basename="SklearnQDA_2c3f")

    def test_model_qda_svm_3c2f(self):
        # 3 classes, 2 features
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1],
                     [2, 1], [3, 2], [-1, 2], [-2, 3], [-2, 2]])
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        X_test = np.array([[-0.8, -1], [0.8, 1], [-0.8, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)
        print(skl_model.predict(X_test))
        print(skl_model.predict_proba(X_test))

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(X_test.astype(np.float32), skl_model, onnx_model,
                            basename="SklearnQDA_3c2f")


if __name__ == "__main__":
    unittest.main(verbosity=3)
