# SPDX-License-Identifier: Apache-2.0

"""
Test scikit-learn's IsolationForest.
"""
import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn import __version__ as sklv
try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented
except ImportError:
    NotImplemented = RuntimeError


sklv2 = '.'.join(sklv.split('.')[:2])


class TestSklearnIsolationForest(unittest.TestCase):

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(StrictVersion(sklv2) < StrictVersion('0.22.0'),
                     reason="tree structure is different.")
    def test_isolation_forest(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx,
                            basename="IsolationForest")

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(StrictVersion(sklv2) < StrictVersion('0.22.0'),
                     reason="tree structure is different.")
    def test_isolation_forest_op1(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = isol.fit(data)
        with self.assertRaises(RuntimeError):
            to_onnx(model, data,
                    target_opset={'': TARGET_OPSET, 'ai.onnx.ml': 1})

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(StrictVersion(sklv2) < StrictVersion('0.22.0'),
                     reason="tree structure is different.")
    def test_isolation_forest_rnd(self):
        isol = IsolationForest(n_estimators=2, random_state=0)
        rs = np.random.RandomState(0)
        data = rs.randn(100, 4).astype(np.float32)
        data[-1, 2:] = 99.
        data[-2, :2] = -99.
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx,
                            basename="IsolationForestRnd")


if __name__ == '__main__':
    unittest.main()
