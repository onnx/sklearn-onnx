# SPDX-License-Identifier: Apache-2.0


import unittest
import numpy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_digits, load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnKMeansModel(unittest.TestCase):
    def test_kmeans_clustering(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3, n_init=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model,
            "kmeans",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeans-Dec4",
        )

    def test_kmeans_clustering_noshape(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3, n_init=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "kmeans", [("input", FloatTensorType([]))], target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeans-Dec4",
        )

    def test_batchkmeans_clustering(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3, n_init=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model,
            "kmeans",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeans-Dec4",
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_batchkmeans_clustering_opset9(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3, n_init=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "kmeans", [("input", FloatTensorType([None, 4]))], target_opset=9
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeansOp9-Dec4",
        )

    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    def test_batchkmeans_clustering_opset11(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3, n_init=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "kmeans", [("input", FloatTensorType([None, 4]))], target_opset=11
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeansOp9-Dec4",
        )

    def test_batchkmeans_clustering_opset1(self):
        data = load_iris()
        X = data.data
        model = MiniBatchKMeans(n_clusters=3, n_init=3)
        model.fit(X)
        try:
            convert_sklearn(
                model, "kmeans", [("input", FloatTensorType([None, 4]))], target_opset=1
            )
        except RuntimeError as e:
            assert "Node 'OnnxAdd' has been changed since version" in str(e)

    def test_kmeans_clustering_int(self):
        data = load_digits()
        X = data.data
        model = KMeans(n_clusters=4, n_init=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model,
            "kmeans",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.int64)[40:60],
            model,
            model_onnx,
            basename="SklearnKMeansInt-Dec4",
        )

    def test_batchkmeans_clustering_int(self):
        data = load_digits()
        X = data.data
        model = MiniBatchKMeans(n_clusters=4, n_init=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model,
            "kmeans",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.int64)[40:60],
            model,
            model_onnx,
            basename="SklearnBatchKMeansInt-Dec4",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
