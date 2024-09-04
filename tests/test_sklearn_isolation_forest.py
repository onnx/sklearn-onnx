# SPDX-License-Identifier: Apache-2.0

"""
Test scikit-learn's IsolationForest.
"""

import logging
import unittest
import packaging.version as pv
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn import __version__ as sklv

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET, TARGET_OPSET_ML

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented
except ImportError:
    NotImplemented = RuntimeError


sklv2 = ".".join(sklv.split(".")[:2])


class TestSklearnIsolationForest(unittest.TestCase):
    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array(
            [[-1.1, -1.2], [0.3, 0.2], [0.5, 0.4], [100.0, 99.0]], dtype=np.float32
        )
        model = isol.fit(data)
        model_onnx = to_onnx(
            model, data, target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML}
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx, basename="IsolationForest")

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_score_samples(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array(
            [[-1.1, -1.2], [0.3, 0.2], [0.5, 0.4], [100.0, 99.0]], dtype=np.float32
        )
        model = isol.fit(data)
        model_onnx = to_onnx(
            model,
            data,
            options={"score_samples": True},
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ["label", "scores", "score_samples"])
        got = sess.run(None, {"X": data})
        self.assertEqual(len(got), 3)
        expected_label = isol.predict(data)
        expected_decif = isol.decision_function(data)
        expected_score = isol.score_samples(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel())
        assert_almost_equal(expected_score, got[2].ravel())

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_op1(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array(
            [[-1.1, -1.2], [0.3, 0.2], [0.5, 0.4], [100.0, 99.0]], dtype=np.float32
        )
        model = isol.fit(data)
        with self.assertRaises(RuntimeError):
            to_onnx(model, data, target_opset={"": TARGET_OPSET, "ai.onnx.ml": 1})

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_rnd(self):
        isol = IsolationForest(n_estimators=2, random_state=0)
        rs = np.random.RandomState(0)
        data = rs.randn(100, 4).astype(np.float32)
        data[-1, 2:] = 99.0
        data[-2, :2] = -99.0
        model = isol.fit(data)
        model_onnx = to_onnx(
            model, data, target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML}
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx, basename="IsolationForestRnd")


if __name__ == "__main__":
    logging.getLogger("skl2onnx").setLevel(logging.ERROR)
    unittest.main()
