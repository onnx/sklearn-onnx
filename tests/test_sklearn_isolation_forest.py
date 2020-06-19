"""
Test scikit-learn's IsolationForest.
"""
import unittest
import numpy as np
from sklearn.ensemble import IsolationForest
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnIsolationForest(unittest.TestCase):

    def test_isolation_forest(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx,
                            basename="IsolationForest")

    def test_isolation_forest_big(self):
        isol = IsolationForest(n_estimators=5, random_state=0)
        data = np.random.rand(100, 4).astype(np.float32)
        data[-1, 2:] = 99.
        data[-2, :2] = -99.
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx,
                            basename="IsolationForest")


if __name__ == '__main__':
    unittest.main()
