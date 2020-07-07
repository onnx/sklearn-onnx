# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
try:
    from sklearn.gaussian_process import GaussianProcessClassifier
except ImportError:
    GaussianProcessClassifier = None
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnGaussianProcessClassifier(unittest.TestCase):

    def fit_classification_model(self, gp, n_classes=2):
        data = load_iris()
        X, y = data.data, data.target
        if n_classes == 2:
            y = y % 2
        elif n_classes != 3:
            raise NotImplementedError("n_classes must be 2 or 3")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=3)
        gp.fit(X_train, y_train)
        return gp, X_test.astype(np.float32)

    def common_test_gpc(self, dtype=np.float32, n_classes=2):

        gp = GaussianProcessClassifier()
        gp, X = self.fit_classification_model(gp, n_classes=n_classes)

        # return_cov=False, return_std=False
        if dtype == np.float32:
            cls = FloatTensorType
        else:
            cls = DoubleTensorType
        model_onnx = to_onnx(
            gp, initial_types=[('X', cls([None, None]))],
            target_opset=TARGET_OPSET,
            options={GaussianProcessClassifier: {
                'zipmap': False, 'optim': 'cdist'}})
        self.assertTrue(model_onnx is not None)

        try:
            from mlprodict.onnxrt import OnnxInference
        except ImportError:
            OnnxInference = None
        if OnnxInference is not None:
            # onnx misses solve operator
            oinf = OnnxInference(model_onnx)
            res = oinf.run({'X': X.astype(dtype)})
            assert_almost_equal(res['label'].ravel(), gp.predict(X).ravel())
            assert_almost_equal(res['probabilities'], gp.predict_proba(X),
                                decimal=3)

        dt = 32 if dtype == np.float32 else 64
        dump_data_and_model(
            X.astype(dtype), gp, model_onnx, verbose=False,
            basename="SklearnGaussianProcessRBFT%d%d" % (n_classes, dt),
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('1.5.0')")

    @unittest.skipIf(TARGET_OPSET < 12, reason="einsum")
    @unittest.skipIf(GaussianProcessClassifier is None,
                     reason="scikit-learn is too old")
    def test_gpc_float_bin(self):
        self.common_test_gpc(dtype=np.float32)

    @unittest.skipIf(TARGET_OPSET < 12, reason="einsum")
    @unittest.skipIf(GaussianProcessClassifier is None,
                     reason="scikit-learn is too old")
    def test_gpc_double_bin(self):
        self.common_test_gpc(dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
