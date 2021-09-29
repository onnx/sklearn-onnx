# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy
from onnxruntime import InferenceSession
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    RandomTreesEmbedding)
from skl2onnx import to_onnx
from test_utils import TARGET_OPSET, dump_data_and_model


class TestSklearnRandomTreeEmbeddings(unittest.TestCase):

    def check_model(self, model, X, name='X'):
        try:
            sess = InferenceSession(model.SerializeToString())
        except Exception as e:
            raise AssertionError(
                "Unable to load model\n%s" % str(model)) from e
        try:
            return sess.run(None, {name: X[:7]})
        except Exception as e:
            raise AssertionError(
                "Unable to run model X.shape=%r X.dtype=%r\n%s" % (
                    X[:7].shape, X.dtype, str(model))) from e

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_random_trees_embedding(self):
        X, _ = make_regression(
            n_features=5, n_samples=100, n_targets=1, random_state=42,
            n_informative=3)
        X = X.astype(numpy.float32)

        model = RandomTreesEmbedding(
            n_estimators=3, max_depth=2, sparse_output=False).fit(X)
        model.transform(X)
        model_onnx = to_onnx(
            model, X[:1], target_opset=TARGET_OPSET)
        with open("model.onnx", "wb") as f:
            f.write(model_onnx.SerializeToString())
        self.check_model(model_onnx, X)
        dump_data_and_model(
            X.astype(numpy.float32), model, model_onnx,
            basename="SklearnRandomTreesEmbedding")


if __name__ == "__main__":
    unittest.main()
