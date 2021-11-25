# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's KBinsDiscretiser converter.
"""

import unittest
import numpy as np
try:
    from sklearn.preprocessing import KBinsDiscretizer
except ImportError:
    # available since 0.20
    KBinsDiscretizer = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnKBinsDiscretiser(unittest.TestCase):
    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_ordinal_uniform(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(n_bins=3,
                                 encode="ordinal",
                                 strategy="uniform").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnKBinsDiscretiserOrdinalUniform")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_ordinal_quantile(self):
        X = np.array([
            [1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
            [0, 3.2, 4.7, -8.9], [0.2, 1.3, 0.6, -9.4],
            [0.8, 4.2, -14.7, -28.9], [8.2, 1.9, 2.6, -5.4],
            [4.8, -9.2, 33.7, 3.9], [81.2, 1., 0.6, 12.4],
            [6.8, 11.2, -1.7, -2.9], [11.2, 12.9, 4.3, -1.4],
        ])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4],
                                 encode="ordinal",
                                 strategy="quantile").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnKBinsDiscretiserOrdinalQuantile")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_ordinal_kmeans(self):
        X = np.array([
            [1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
            [0, 3.2, 4.7, -8.9], [0.2, 1.3, 0.6, -9.4],
            [0.8, 4.2, -14.7, -28.9], [8.2, 1.9, 2.6, -5.4],
            [4.8, -9.2, 33.7, 3.9], [81.2, 1., 0.6, 12.4],
            [6.8, 11.2, -1.7, -2.9], [11.2, 12.9, 4.3, -1.4],
        ])
        model = KBinsDiscretizer(n_bins=3, encode="ordinal",
                                 strategy="kmeans").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnKBinsDiscretiserOrdinalKMeans")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_onehot_dense_uniform(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4],
                                 encode="onehot-dense",
                                 strategy="uniform").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnKBinsDiscretiserOneHotDenseUniform")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_onehot_dense_quantile(self):
        X = np.array([
            [1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
            [0, 3.2, 4.7, -8.9], [0.2, 1.3, 0.6, -9.4],
            [0.8, 4.2, -14.7, -28.9], [8.2, 1.9, 2.6, -5.4],
            [4.8, -9.2, 33.7, 3.9], [81.2, 1., 0.6, 12.4],
            [6.8, 11.2, -1.7, -2.9], [11.2, 12.9, 4.3, -1.4],
        ])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4],
                                 encode="onehot-dense",
                                 strategy="quantile").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnKBinsDiscretiserOneHotDenseQuantile")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_onehot_dense_kmeans(self):
        X = np.array([
            [1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
            [0, 3.2, 4.7, -8.9], [0.2, 1.3, 0.6, -9.4],
            [0.8, 4.2, -14.7, -28.9], [8.2, 1.9, 2.6, -5.4],
            [4.8, -9.2, 33.7, 3.9], [81.2, 1., 0.6, 12.4],
            [6.8, 11.2, -1.7, -2.9], [11.2, 12.9, 4.3, -1.4],
        ])
        model = KBinsDiscretizer(n_bins=3,
                                 encode="onehot-dense",
                                 strategy="kmeans").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32), model, model_onnx,
            basename="SklearnKBinsDiscretiserOneHotDenseKMeans")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_ordinal_uniform_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0], [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=3,
                                 encode="ordinal",
                                 strategy="uniform").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnKBinsDiscretiserOrdinalUniformInt")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_ordinal_quantile_int(self):
        X = np.array([
            [1, 3, 3, -6], [3, -2, 5, 0], [0, 2, 7, -9],
            [-1, 0, 1, -16], [31, -5, 15, 10], [12, -2, 8, -19],
            [12, 13, 31, -16], [0, -21, 15, 30], [10, 22, 71, -91]
        ])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4],
                                 encode="ordinal",
                                 strategy="quantile").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnKBinsDiscretiserOrdinalQuantileInt")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_ordinal_kmeans_int(self):
        X = np.array([
            [1, 3, 3, -6], [3, -2, 5, 0], [0, 2, 7, -9],
            [-1, 0, 1, -16], [31, -5, 15, 10], [12, -2, 8, -19]
        ])
        model = KBinsDiscretizer(n_bins=3, encode="ordinal",
                                 strategy="kmeans").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnKBinsDiscretiserOrdinalKMeansInt")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_onehot_dense_uniform_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0], [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4],
                                 encode="onehot-dense",
                                 strategy="uniform").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnKBinsDiscretiserOneHotDenseUniformInt")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_onehot_dense_quantile_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0], [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4],
                                 encode="onehot-dense",
                                 strategy="quantile").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnKBinsDiscretiserOneHotDenseQuantileInt")

    @unittest.skipIf(
        KBinsDiscretizer is None,
        reason="KBinsDiscretizer available since 0.20",
    )
    def test_model_k_bins_discretiser_onehot_dense_kmeans_int(self):
        X = np.array([
            [1, 3, 3, -6], [3, -2, 5, 0], [0, 2, 7, -9],
            [-1, 12, 32, -16], [31, -20, 51, 7], [10, 23, 73, -90],
            [1, 23, 36, -61], [93, -12, 15, 10], [20, 12, 17, -19]
        ])
        model = KBinsDiscretizer(n_bins=3,
                                 encode="onehot-dense",
                                 strategy="kmeans").fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KBinsDiscretiser",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64), model, model_onnx,
            basename="SklearnKBinsDiscretiserOneHotDenseKMeansInt")


if __name__ == "__main__":
    unittest.main()
