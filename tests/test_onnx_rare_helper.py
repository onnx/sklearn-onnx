# SPDX-License-Identifier: Apache-2.0

"""
Tests on functions in *onnx_helper*.
"""
import unittest
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from onnx.defs import onnx_opset_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_rare_helper import upgrade_opset_number
from test_utils import TARGET_OPSET


class TestOnnxRareHelper(unittest.TestCase):

    def test_kmeans_upgrade(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(model, "kmeans",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=7)
        model8 = upgrade_opset_number(model_onnx, 8)
        assert "version: 8" in str(model8)

    @unittest.skipIf(onnx_opset_version() < 11,
                     reason="Needs opset >= 11")
    def test_knn_upgrade(self):
        iris = load_iris()
        X, _ = iris.data, iris.target

        clr = NearestNeighbors(n_neighbors=3, radius=None)
        clr.fit(X)

        model_onnx = convert_sklearn(clr, "up",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=9)
        try:
            upgrade_opset_number(model_onnx, 8)
            raise AssertionError()
        except RuntimeError:
            pass
        try:
            upgrade_opset_number(model_onnx, TARGET_OPSET)
        except RuntimeError as e:
            assert "was updated" in str(e)


if __name__ == "__main__":
    unittest.main()
