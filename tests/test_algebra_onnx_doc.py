# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import sys
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from skl2onnx.algebra.onnx_ops import dynamic_class_creation
from skl2onnx.algebra.automation import get_rst_doc_sklearn, get_rst_doc
from test_utils import TARGET_OPSET


class TestAlgebraOnnxDoc(unittest.TestCase):

    def setUp(self):
        self._algebra = dynamic_class_creation()

    def predict_with_onnxruntime(self, model_def, *inputs):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_def.SerializeToString())
        names = [i.name for i in sess.get_inputs()]
        input = {name: input for name, input in zip(names, inputs)}
        res = sess.run(None, input)
        names = [o.name for o in sess.get_outputs()]
        return {name: output for name, output in zip(names, res)}

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    def test_transpose2(self):
        from skl2onnx.algebra.onnx_ops import OnnxTranspose

        node = OnnxTranspose(
            OnnxTranspose(
                'X', perm=[1, 0, 2],
                op_version=TARGET_OPSET),
            perm=[1, 0, 2], output_names=['Y'],
            op_version=TARGET_OPSET)
        X = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)

        model_def = node.to_onnx({'X': X})
        onnx.checker.check_model(model_def)
        res = self.predict_with_onnxruntime(model_def, X)
        assert_almost_equal(res['Y'], X)

    @unittest.skipIf(sys.platform.startswith("win"),
                     reason="onnx schema are incorrect on Windows")
    def test_doc_onnx(self):
        rst = get_rst_doc()
        assert "**Summary**" in rst

    @unittest.skipIf(sys.platform.startswith("win"),
                     reason="onnx schema are incorrect on Windows")
    def test_doc_sklearn(self):
        try:
            rst = get_rst_doc_sklearn()
            assert ".. _l-sklops-OnnxSklearnBernoulliNB:" in rst
        except KeyError as e:
            assert ("SklearnGaussianProcessRegressor" in str(e) or
                    "SklearnGaussianProcessClassifier" in str(e))


if __name__ == "__main__":
    unittest.main()
