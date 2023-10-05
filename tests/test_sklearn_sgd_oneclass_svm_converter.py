# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import unittest
import numpy as np

try:
    from sklearn.linear_model import SGDOneClassSVM
except ImportError:
    SGDOneClassSVM = None
from onnxruntime import __version__ as ort_version
from skl2onnx import convert_sklearn

from skl2onnx.common.data_types import (
    FloatTensorType,
)

from test_utils import dump_data_and_model, TARGET_OPSET

ort_version = ".".join(ort_version.split(".")[:2])


class TestSGDOneClassSVMConverter(unittest.TestCase):
    @unittest.skipIf(SGDOneClassSVM is None, reason="scikit-learn<1.0")
    def test_model_sgd_oneclass_svm(self):
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        model = SGDOneClassSVM(random_state=42)
        model.fit(X)
        test_x = np.array([[0, 0], [-1, -1], [1, 1]]).astype(np.float32)
        model.predict(test_x)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD OneClass SVM",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            test_x.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnSGDOneClassSVMBinaryHinge",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
