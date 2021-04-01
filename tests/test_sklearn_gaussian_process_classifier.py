# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
import scipy
from onnxruntime import InferenceSession, SessionOptions
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
except ImportError:
    OrtFail = RuntimeError
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sklver
try:
    from sklearn.gaussian_process import GaussianProcessClassifier
except ImportError:
    GaussianProcessClassifier = None
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import to_onnx
from skl2onnx.helpers.onnx_helper import change_onnx_domain
from test_utils import dump_data_and_model, TARGET_OPSET


sklver_ = ".".join(sklver.split('.')[:2])


class TestSklearnGaussianProcessClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from ortcustomops import (
                onnx_op, PyCustomOpDef, get_library_path)
        except ImportError:
            return

        @onnx_op(op_type="SolveFloat",
                 inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
                 outputs=[PyCustomOpDef.dt_float])
        def solveopf(a, b):
            # The user custom op implementation here.
            return scipy.linalg.solve(a, b).astype(np.float32)

        @onnx_op(op_type="SolveDouble",
                 inputs=[PyCustomOpDef.dt_double, PyCustomOpDef.dt_double],
                 outputs=[PyCustomOpDef.dt_double])
        def solveopd(a, b):
            # The user custom op implementation here.
            return scipy.linalg.solve(a, b).astype(np.float64)

        cls.path = get_library_path()

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
            sess = InferenceSession(model_onnx.SerializeToString())
        except OrtFail:
            if not hasattr(self, 'path'):
                return
            suffix = 'Double' if dtype == np.float64 else 'Float'
            # Operator Solve is missing
            model_onnx = change_onnx_domain(
                model_onnx, {'Solve': ('Solve%s' % suffix, 'ai.onnx.contrib')})
            so = SessionOptions()
            so.register_custom_ops_library(self.path)
            sess = InferenceSession(model_onnx.SerializeToString(), so)

            res = sess.run(None, {'X': X.astype(dtype)})
            assert_almost_equal(res[0].ravel(), gp.predict(X).ravel())
            assert_almost_equal(res[1], gp.predict_proba(X),
                                decimal=3)
            return

        dt = 32 if dtype == np.float32 else 64
        dump_data_and_model(
            X.astype(dtype), gp, model_onnx, verbose=False,
            basename="SklearnGaussianProcessRBFT%d%d" % (n_classes, dt))

    @unittest.skipIf(TARGET_OPSET < 12, reason="einsum")
    @unittest.skipIf(GaussianProcessClassifier is None,
                     reason="scikit-learn is too old")
    @unittest.skipIf(StrictVersion(sklver_) < StrictVersion("0.22"),
                     reason="not available")
    def test_gpc_float_bin(self):
        self.common_test_gpc(dtype=np.float32)

    @unittest.skipIf(TARGET_OPSET < 12, reason="einsum, reciprocal")
    @unittest.skipIf(GaussianProcessClassifier is None,
                     reason="scikit-learn is too old")
    @unittest.skipIf(StrictVersion(sklver_) < StrictVersion("0.22"),
                     reason="not available")
    def test_gpc_double_bin(self):
        self.common_test_gpc(dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
