# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from distutils.version import StrictVersion
import numpy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model
import onnx


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

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="OnnxOperator not working")
    def test_batchkmeans_clustering_opset9(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([1, 4]))],
                                     target_opset=9)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeansOp9-Dec4",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )

    def test_batchkmeans_clustering_opset1(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3)
        model.fit(X)
        try:
            convert_sklearn(model, "kmeans",
                            [("input", FloatTensorType([1, 4]))],
                            target_opset=1)
        except RuntimeError as e:
            assert "Node 'OnnxAdd' has been changed since version" in str(e)


if __name__ == "__main__":
    unittest.main()
