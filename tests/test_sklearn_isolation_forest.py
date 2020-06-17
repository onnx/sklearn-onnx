"""
Test scikit-learn's IsolationForest.
"""
import unittest
import numpy as np
from sklearn.ensemble import IsolationForest
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnIsolationForest(unittest.TestCase):

    def test_isolation_forest(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array([[-1.1], [0.3], [0.5], [100]], dtype=np.float32)
        model = isol.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx,
                            basename="IsolationForest")


if __name__ == '__main__':
    unittest.main()
