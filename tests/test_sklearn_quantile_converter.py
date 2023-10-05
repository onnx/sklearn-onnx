# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's polynomial features converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy as np
import onnx
from sklearn.preprocessing import QuantileTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnQuantileTransformer(unittest.TestCase):
    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
        reason="ConstantOfShape not available",
    )
    def test_quantile_transformer(self):
        X = np.empty((100, 2), dtype=np.float32)
        X[:, 0] = np.arange(X.shape[0])
        X[:, 1] = np.arange(X.shape[0]) * 2
        model = QuantileTransformer(n_quantiles=6).fit(X)
        model_onnx = convert_sklearn(
            model, "test", [("input", FloatTensorType([None, X.shape[1]]))]
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnQuantileTransformer",
        )


if __name__ == "__main__":
    unittest.main()
