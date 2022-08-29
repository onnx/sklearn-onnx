# SPDX-License-Identifier: Apache-2.0

import numpy as np
import unittest
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType)
from tests.test_utils import (
    dump_data_and_model,
    dump_multiple_classification,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET)

import onnx

warnings_to_skip = (DeprecationWarning, FutureWarning, ConvergenceWarning)


ort_version = '.'.join(ort_version.split('.')[:2])


class TestOvrDecisionFunction(unittest.TestCase):
    def test_ovr_function(self):

        label = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ])

        score = np.array([
            [ 2.7691362 ,  1.3476608 ,  1.9489377 ],
            [ 1.4604166 ,  0.45662668, -1.5390667 ],
            [-1.7293018 , -1.5082803 , -7.805672  ],
            [ 3.1502779 ,  1.4747248 ,  1.3853829 ],
            [-1.2183753 , -1.1599588 , -6.503545  ],
            [ 3.2527397 ,  1.6418446 ,  2.6538432 ],
            [-1.3828267 , -1.2654455 , -6.772983  ],
            [ 1.7424656 ,  0.6119384 , -1.1415086 ],
            [ 1.8815211 ,  0.6829882 , -1.0046322 ],
            [ 1.3255664 ,  0.3806743 , -1.6823313 ]
        ])


#        result = np.array([2 1 0 2 0 2 0 1 1 1])

        model_onnx = convert_sklearn(
            model,
            "scikit-learn OneVsOne Classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'zipmap': False}})

        self.assertIsNotNone(model_onnx)

        onnx.save_model(model_onnx, "testovo.onnx")

        print("onnx model saved")

        sess = InferenceSession(model_onnx.SerializeToString())
        XI = X_test[:10].astype(np.float32)
        got = sess.run(None, {'input': XI})
        assert_almost_equal(exp_label, got[0])


if __name__ == "__main__":
    unittest.main()
