# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np

try:
    from onnx.reference import ReferenceEvaluator
except ImportError:
    ReferenceEvaluator = None
from sklearn.tree import DecisionTreeClassifier
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from test_utils import TARGET_OPSET


class TestSklearnClassifiersExtreme(unittest.TestCase):
    def test_one_training_class(self):
        x = np.eye(4, dtype=np.float32)
        y = np.array([5, 5, 5, 5], dtype=np.int64)

        cl = DecisionTreeClassifier()
        cl = cl.fit(x, y)

        expected = [cl.predict(x), cl.predict_proba(x)]
        onx = to_onnx(cl, x, target_opset=TARGET_OPSET, options={"zipmap": False})

        for cls in [
            (
                (lambda onx: ReferenceEvaluator(onx, verbose=0))
                if ReferenceEvaluator is not None
                else None
            ),
            lambda onx: InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            ),
        ]:
            if cls is None:
                continue
            sess = cls(onx)
            res = sess.run(None, {"X": x})
            self.assertEqual(len(res), len(expected))
            for e, g in zip(expected, res):
                self.assertEqual(e.tolist(), g.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
