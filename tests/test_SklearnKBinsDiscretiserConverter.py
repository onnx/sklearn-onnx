"""
Tests scikit-learn's KBinsDiscretiser converter.
"""

import numpy as np
import unittest
try:
    from sklearn.preprocessing import KBinsDiscretizer
except ImportError:
    # available since 0.20
    KBinsDiscretizer = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnKBinsDiscretiser(unittest.TestCase):

    @unittest.skipIf(KBinsDiscretizer is None,
                     reason="KBinsDiscretizer available since 0.20")
    @unittest.skip(reason="Digitize op has not yet been implemented in onnx")
    def test_model_k_bins_discretiser_ordinal_uniform(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(nbins=3, encode='ordinal',
                                 strategy='uniform').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalUniform")

    @unittest.skipIf(KBinsDiscretizer is None,
                     reason="KBinsDiscretizer available since 0.20")
    @unittest.skip(reason="Digitize op has not yet been implemented in onnx")
    def test_model_k_bins_discretiser_ordinal_quantile(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(nbins=[3, 2, 3, 4], encode='ordinal',
                                 strategy='quantile').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalQuantile")

    @unittest.skipIf(KBinsDiscretizer is None,
                     reason="KBinsDiscretizer available since 0.20")
    @unittest.skip(reason="Digitize op has not yet been implemented in onnx")
    def test_model_k_bins_discretiser_ordinal_kmeans(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(nbins=3, encode='ordinal',
                                 strategy='kmeans').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOrdinalKMeans")

    @unittest.skipIf(KBinsDiscretizer is None,
                     reason="KBinsDiscretizer available since 0.20")
    @unittest.skip(reason="Digitize op has not yet been implemented in onnx")
    def test_model_k_bins_discretiser_onehot_dense_uniform(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(nbins=[3, 2, 3, 4], encode='onehot-dense',
                                 strategy='uniform').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseUniform")

    @unittest.skipIf(KBinsDiscretizer is None,
                     reason="KBinsDiscretizer available since 0.20")
    @unittest.skip(reason="Digitize op has not yet been implemented in onnx")
    def test_model_k_bins_discretiser_onehot_dense_quantile(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(nbins=[3, 2, 3, 4], encode='onehot-dense',
                                 strategy='quantile').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseQuantile")

    @unittest.skipIf(KBinsDiscretizer is None,
                     reason="KBinsDiscretizer available since 0.20")
    @unittest.skip(reason="Digitize op has not yet been implemented in onnx")
    def test_model_k_bins_discretiser_onehot_dense_kmeans(self):
        X = np.array([[1.2, 3.2, 1.3, -5.6], [4.3, -3.2, 5.7, 1.0],
                      [0, 3.2, 4.7, -8.9]])
        model = KBinsDiscretizer(nbins=3, encode='onehot-dense',
                                 strategy='kmeans').fit(X)
        model_onnx = convert_sklearn(model, 'scikit-learn KBinsDiscretiser',
                                     [('input', FloatTensorType(X.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), model, model_onnx,
                basename="SklearnKBinsDiscretiserOneHotDenseKMeans")


if __name__ == "__main__":
    unittest.main()
