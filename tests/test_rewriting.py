# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from numpy.testing import assert_almost_equal
from onnx.reference import ReferenceEvaluator


class TestRewriting(unittest.TestCase):
    def test_lp_normalization(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("LpNormalization", ["X"], ["Y1"], axis=-1, p=1),
                    oh.make_node("ReduceSum", ["X", "axis"], ["sa"]),
                    oh.make_node("Div", ["X", "sa"], ["Y2"]),
                ],
                "test",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["a", 2])],
                [
                    oh.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, ["a", 2]),
                    oh.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, ["a", 2]),
                ],
                [onh.from_array(np.array([-1], dtype=np.int64), name="axis")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        ref = ReferenceEvaluator(model, verbose=0)
        feeds = dict(X=np.array([[1, 1], [0, 1], [-1, 2]], dtype=np.float32))
        res = ref.run(None, feeds)
        assert_almost_equal(res[0], res[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
