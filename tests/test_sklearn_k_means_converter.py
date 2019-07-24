# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnKMeansModel(unittest.TestCase):
    def test_kmeans_clustering(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeans-Dec4",
            # Operator gemm is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )

    def test_batchkmeans_clustering(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeans-Dec4",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )


if __name__ == "__main__":
    unittest.main()
