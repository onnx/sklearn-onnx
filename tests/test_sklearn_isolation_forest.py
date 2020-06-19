"""
Test scikit-learn's IsolationForest.
"""
import warnings
import unittest
import numpy as np
try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET
from test_utils.utils_backend import (
    OnnxRuntimeMissingNewOnnxOperatorException)
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented
except ImportError:
    NotImplemented = RuntimeError


class TestSklearnIsolationForest(unittest.TestCase):

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    def test_isolation_forest(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        try:
            dump_data_and_model(data, model, model_onnx,
                                basename="IsolationForest")
        except (OnnxRuntimeMissingNewOnnxOperatorException,
                NotImplemented) as e:
            warnings.warn(str(e))
            return

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    def test_isolation_forest_rnd(self):
        isol = IsolationForest(n_estimators=2, random_state=0)
        rs = np.random.RandomState(0)
        data = rs.randn(100, 4).astype(np.float32)
        data[-1, 2:] = 99.
        data[-2, :2] = -99.
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        try:
            dump_data_and_model(data, model, model_onnx,
                                basename="IsolationForestRnd")
        except (OnnxRuntimeMissingNewOnnxOperatorException,
                NotImplemented) as e:
            warnings.warn(str(e))
            return


if __name__ == '__main__':
    unittest.main()
