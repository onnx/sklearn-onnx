# SPDX-License-Identifier: Apache-2.0


import unittest

import numpy as np
from sklearn.decomposition import TruncatedSVD

from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx import convert_sklearn
from test_utils import create_tensor
from test_utils import dump_data_and_model, TARGET_OPSET


class TestTruncatedSVD(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_truncated_svd(self):
        N, C, K = 2, 3, 2
        x = create_tensor(N, C)

        svd = TruncatedSVD(n_components=K)
        svd.fit(x)
        model_onnx = convert_sklearn(svd,
                                     initial_types=[
                                         ("input",
                                          FloatTensorType(shape=[None, C]))
                                     ], target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(x, svd, model_onnx, basename="SklearnTruncatedSVD")

    def test_truncated_svd_arpack(self):
        X = create_tensor(10, 10)
        svd = TruncatedSVD(n_components=5, algorithm='arpack', n_iter=10,
                           tol=0.1, random_state=42).fit(X)
        model_onnx = convert_sklearn(svd,
                                     initial_types=[
                                         ("input",
                                          FloatTensorType(shape=X.shape))
                                     ], target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, svd, model_onnx,
                            basename="SklearnTruncatedSVDArpack")

    def test_truncated_svd_int(self):
        X = create_tensor(5, 5).astype(np.int64)
        svd = TruncatedSVD(n_iter=20, random_state=42).fit(X)
        model_onnx = convert_sklearn(svd,
                                     initial_types=[
                                         ("input",
                                          Int64TensorType([None, X.shape[1]]))
                                     ], target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X, svd, model_onnx,
            basename="SklearnTruncatedSVDInt")


if __name__ == "__main__":
    unittest.main()
