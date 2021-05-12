# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import numpy
import onnx
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_digits, load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnKMeansModel(unittest.TestCase):
    def test_kmeans_clustering(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model, model_onnx,
            basename="SklearnKMeans-Dec4",
            # Operator gemm is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')")

    def test_kmeans_clustering_noshape(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model, model_onnx,
            basename="SklearnKMeans-Dec4",
            # Operator gemm is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')")

    def test_batchkmeans_clustering(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
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
                                     [("input", FloatTensorType([None, 4]))],
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

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.6.0"),
                     reason="OnnxOperator not working")
    def test_batchkmeans_clustering_opset11(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=11)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeansOp9-Dec4",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')")

    def test_batchkmeans_clustering_opset1(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3)
        model.fit(X)
        try:
            convert_sklearn(model, "kmeans",
                            [("input", FloatTensorType([None, 4]))],
                            target_opset=1)
        except RuntimeError as e:
            assert "Node 'OnnxAdd' has been changed since version" in str(e)

    def test_kmeans_clustering_int(self):
        data = load_digits()
        X = data.data
        model = KMeans(n_clusters=4)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", Int64TensorType([None,
                                      X.shape[1]]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.int64)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeansInt-Dec4",
            # Operator gemm is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__) "
                          "<= StrictVersion('0.2.1')",
        )

    def test_batchkmeans_clustering_int(self):
        data = load_digits()
        X = data.data
        model = MiniBatchKMeans(n_clusters=4)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", Int64TensorType([None,
                                      X.shape[1]]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.int64)[40:60],
            model,
            model_onnx,
            basename="SklearnBatchKMeansInt-Dec4",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__) "
                          "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
