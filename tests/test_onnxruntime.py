# SPDX-License-Identifier: Apache-2.0

import os
import unittest
import packaging.version as pv
import numpy as np
from numpy.testing import assert_allclose

try:
    import onnx.reference  # noqa: F401
    from test_utils import ReferenceEvaluatorEx
except ImportError:
    ReferenceEvaluatorEx = None
from onnxruntime import InferenceSession, __version__ as ort_version


class TestOnnxruntime(unittest.TestCase):
    X3_15 = np.array(
        [
            [
                -0.32256478,
                1.7266265,
                0.47051477,
                1.1111994,
                1.9582617,
                -2.1582267,
                -1.9729482,
                -1.5662458,
                1.8967382,
                0.9119621,
                -0.93173814,
                2.9724689,
                -0.7231156,
                0.10379718,
                -1.3578224,
                0.37283298,
                -0.38267845,
                0.23394746,
                -1.6884863,
                0.6374923,
            ],
            [
                -0.53266096,
                -0.767421,
                1.661441,
                0.52790266,
                1.6549803,
                0.5076044,
                -2.9024098,
                0.86126643,
                -1.3819953,
                2.5567708,
                -1.7888857,
                -0.07472081,
                0.24990171,
                -0.87638474,
                -0.14730039,
                1.3493251,
                -0.7835222,
                -0.9997528,
                -0.91080195,
                -3.6515126,
            ],
            [
                -0.8703916,
                0.43145382,
                1.0918913,
                -1.397069,
                -0.48047885,
                3.1278436,
                3.8035386,
                -0.22710086,
                -0.42011356,
                1.4203368,
                0.47596663,
                -0.44953802,
                -0.68278235,
                0.87819546,
                -2.4272032,
                0.08891433,
                0.7960927,
                1.2197107,
                1.7008729,
                1.0122501,
            ],
        ],
        dtype=np.float32,
    )

    @unittest.skipIf(ReferenceEvaluatorEx is None, "onnx too old")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.12.0"),
        reason="ai.opset.ml==3 not implemented",
    )
    def test_tree_ensemble_classifier(self):
        """
        The onnx graph was produced by the following code.

        ::

            node = [n for n in model_onnx.graph.node
                    if n.op_type == "TreeEnsembleClassifier"]
            import numpy
            from onnx import numpy_helper, TensorProto
            from onnx.helper import (
                make_model, make_node, set_model_props, make_tensor,
                make_graph, make_tensor_value_info, make_opsetid)
            from onnx.checker import check_model
            node = node[0]
            X0 = make_tensor_value_info(
                node.input[0], TensorProto.FLOAT, [None, None])
            Y1 = make_tensor_value_info(
                node.output[0], TensorProto.INT64, [None])
            Y2 = make_tensor_value_info(
                node.output[1], TensorProto.FLOAT, [None, None])
            graph = make_graph([node], 'g', [X0], [Y1, Y2])
            opset_imports = [make_opsetid("", 17),
                             make_opsetid('ai.onnx.ml', 1)]
            onnx_model = make_model(graph, opset_imports=opset_imports)
            check_model(onnx_model)
            with open("treecl.onnx", "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(repr(X[:5]))
        """
        X = self.X3_15
        name = os.path.join(os.path.dirname(__file__), "datasets", "treecl.onnx")
        sess = ReferenceEvaluatorEx(name)
        label, proba = sess.run(None, {"input": X})
        sesso = InferenceSession(name, providers=["CPUExecutionProvider"])
        labelo, probao = sesso.run(None, {"input": X})
        assert_allclose(probao, proba, atol=1e-8)
        assert_allclose(labelo, label)

    @unittest.skipIf(ReferenceEvaluatorEx is None, "onnx too old")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.12.0"),
        reason="ai.opset.ml==3 not implemented",
    )
    def test_tree_ensemble_classifier_2(self):
        X = self.X3_15
        name = os.path.join(os.path.dirname(__file__), "datasets", "treecl2.onnx")
        sess = ReferenceEvaluatorEx(name)
        label, proba = sess.run(None, {"input": X})
        sesso = InferenceSession(name, providers=["CPUExecutionProvider"])
        labelo, probao = sesso.run(None, {"input": X})
        assert_allclose(probao, proba, atol=1e-6)
        assert_allclose(labelo, label)

    @unittest.skipIf(ReferenceEvaluatorEx is None, "onnx too old")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.12.0"),
        reason="ai.opset.ml==3 not implemented",
    )
    def test_tree_ensemble_classifier_3(self):
        X = self.X3_15[:, :10]
        name = os.path.join(os.path.dirname(__file__), "datasets", "treecl3.onnx")
        sess = ReferenceEvaluatorEx(name)
        label, proba = sess.run(None, {"input": X})
        sesso = InferenceSession(name, providers=["CPUExecutionProvider"])
        labelo, probao = sesso.run(None, {"input": X})
        assert_allclose(probao, proba, atol=1e-6)
        assert_allclose(labelo, label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
