# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from onnxruntime import __version__ as ort_version
#from skl2onnx import convert_sklearn
import sys
sys.path.append("D:\GitHub\onnx\sklearn-onnx")
from skl2onnx.convert import convert_sklearn

from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    TARGET_OPSET
)

ort_version = ".".join(ort_version.split(".")[:2])

import onnxmltools

class TestSGDOneClassSVMConverter(unittest.TestCase):

    def test_model_sgd_oneclass_svm(self):
#       model, X = fit_classification_model(SGDOneClassSVM(random_state=42), 2)
        X = np.array([
            [-1,-1],[-2,-1],[1,1],[2,1]
        ])
        model = SGDOneClassSVM(random_state=42)
        model.fit(X)
        test_x = np.array([[0,0]])
        a = model.predict(test_x)
        print("predict:", test_x, a)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD OneClass SVM",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        onnxmltools.utils.save_model(model_onnx, "sk_SGD_OneClass_SVM.onnx")
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnSGDOneClassSVMBinaryHinge-Out0")


if __name__ == "__main__":
    unittest.main(verbosity=3)
