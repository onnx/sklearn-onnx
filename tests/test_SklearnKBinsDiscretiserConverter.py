"""
Tests scikit-learn's KBinsDiscretiser converter.
"""

import numpy as np
import unittest
from sklearn.preprocessing import KBinsDiscretizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnKBinsDiscretiser(unittest.TestCase):

    def test_model_k_bins_discretiser_ordinal_uniform(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(n_bins=3, encode='ordinal',
                                 strategy='uniform').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalUniform")

    def test_model_k_bins_discretiser_ordinal_quantile(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4], encode='ordinal',
                                 strategy='quantile').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalQuantile")

    def test_model_k_bins_discretiser_ordinal_kmeans(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(n_bins=3, encode='ordinal',
                                 strategy='kmeans').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalKMeans")

    def test_model_k_bins_discretiser_onehot_dense_uniform(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4], encode='onehot-dense',
                                 strategy='uniform').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseUniform")

    def test_model_k_bins_discretiser_onehot_dense_quantile(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4], encode='onehot-dense',
                                 strategy='quantile').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseQuantile")

    def test_model_k_bins_discretiser_onehot_dense_kmeans(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0],
                      [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=3, encode='onehot-dense',
                                 strategy='kmeans').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseKMeans")

    def test_model_k_bins_discretiser_ordinal_uniform_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0],
                      [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=3, encode='ordinal',
                                 strategy='uniform').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalUniformInt")

    def test_model_k_bins_discretiser_ordinal_quantile_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0],
                      [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4], encode='ordinal',
                                 strategy='quantile').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalQuantileInt")

    def test_model_k_bins_discretiser_ordinal_kmeans_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0],
                      [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=3, encode='ordinal',
                                 strategy='kmeans').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalKMeansInt")

    def test_model_k_bins_discretiser_onehot_dense_uniform_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0],
                      [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4], encode='onehot-dense',
                                 strategy='uniform').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseUniformInt")

    def test_model_k_bins_discretiser_onehot_dense_quantile_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0],
                      [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=[3, 2, 3, 4], encode='onehot-dense',
                                 strategy='quantile').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseQuantileInt")

    def test_model_k_bins_discretiser_onehot_dense_kmeans_int(self):
        X = np.array([[1, 3, 3, -6], [3, -2, 5, 0],
                      [0, 2, 7, -9]])
        model = KBinsDiscretizer(n_bins=3, encode='onehot-dense',
                                 strategy='kmeans').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', Int64TensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.int64), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseKMeansInt")


if __name__ == "__main__":
    unittest.main()
