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
from onnxruntime import InferenceSession


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
        X = np.array([
            [-1,-1],[-2,-1],[1,1],[2,1]
        ])
        model = SGDOneClassSVM(random_state=42)
        model.fit(X)
        test_x = np.array([[0,0],[-1,-1],[1,1]]).astype(np.float32)
        result = model.predict(test_x)
        print("predict:\n", test_x)
        print("result:\n", result)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD OneClass SVM",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        onnxmltools.utils.save_model(model_onnx, "sk_SGD_OneClass_SVM.onnx")

        # sess = InferenceSession(model_onnx.SerializeToString())
        # output = sess.run(None, {'input':test_x})
        # print(output[0].shape, output[1].shape)

        self.assertIsNotNone(model_onnx)
        dump_data_and_model(test_x.astype(np.float32), model, model_onnx,
            basename="SklearnSGDOneClassSVMBinaryHinge-Out0")


if __name__ == "__main__":
    unittest.main(verbosity=3)
