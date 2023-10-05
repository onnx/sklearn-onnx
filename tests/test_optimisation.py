# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import checker
from onnx import helper
from onnx import TensorProto as tp
from skl2onnx.common.onnx_optimisation_identity import onnx_remove_node_identity
from test_utils import TARGET_OPSET, TARGET_IR, InferenceSessionEx as InferenceSession


class TestOptimisation(unittest.TestCase):
    @unittest.skipIf(TARGET_OPSET <= 14, reason="only verified with opset 15+")
    def test_coptimisation_identity_removal(self):
        # investigation issue #854

        then_branch = helper.make_graph(
            [
                helper.make_node(
                    "Identity", inputs=["identity_one"], outputs=["then_result"]
                )
            ],
            "then_branch",
            [],
            [helper.make_tensor_value_info("then_result", tp.INT64, [1])],
        )

        else_branch = helper.make_graph(
            [
                helper.make_node(
                    "Identity", inputs=["identity_zero"], outputs=["else_result"]
                )
            ],
            "else_branch",
            [],
            [helper.make_tensor_value_info("else_result", tp.INT64, [1])],
        )

        nodes = [
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["one"],
                value=helper.make_tensor(
                    name="", data_type=tp.INT64, dims=[1], vals=[1]
                ),
            ),
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["zero"],
                value=helper.make_tensor(
                    name="", data_type=tp.INT64, dims=[1], vals=[0]
                ),
            ),
            helper.make_node("Identity", inputs=["one"], outputs=["identity_one"]),
            helper.make_node("Identity", inputs=["zero"], outputs=["identity_zero"]),
            helper.make_node(
                "If",
                inputs=["X"],
                outputs=["y"],
                then_branch=then_branch,
                else_branch=else_branch,
            ),
        ]

        g = helper.make_graph(
            nodes,
            "if_test",
            [helper.make_tensor_value_info("X", tp.BOOL, [1])],
            [helper.make_tensor_value_info("y", tp.INT64, [1])],
        )

        # Create the model and check
        m = helper.make_model(
            g,
            opset_imports=[helper.make_opsetid("", TARGET_OPSET)],
            ir_version=TARGET_IR,
        )
        checker.check_model(m)

        sess = InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        optimized_model = onnx_remove_node_identity(m)
        sess_opt = InferenceSession(
            optimized_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        for v in [True, False]:
            x = np.array([v])
            expected = sess.run(None, {"X": x})
            got = sess_opt.run(None, {"X": x})
            assert_almost_equal(expected, got)


if __name__ == "__main__":
    unittest.main()
