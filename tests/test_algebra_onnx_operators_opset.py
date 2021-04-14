# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
import onnxruntime as ort
from skl2onnx.algebra.onnx_ops import OnnxPad  # noqa


class TestOnnxOperatorsOpset(unittest.TestCase):

    @unittest.skipIf(onnx.defs.onnx_opset_version() < 10, "irrelevant")
    def test_pad_opset_10(self):

        pad = OnnxPad('X', output_names=['Y'],
                      mode='constant', value=1.5,
                      pads=[0, 1, 0, 1],
                      op_version=2)

        X = np.array([[0, 1]], dtype=np.float32)
        model_def = pad.to_onnx({'X': X}, target_opset=10)
        onnx.checker.check_model(model_def)

        def predict_with_onnxruntime(model_def, *inputs):
            sess = ort.InferenceSession(model_def.SerializeToString())
            names = [i.name for i in sess.get_inputs()]
            dinputs = {name: input for name, input in zip(names, inputs)}
            res = sess.run(None, dinputs)
            names = [o.name for o in sess.get_outputs()]
            return {name: output for name, output in zip(names, res)}

        Y = predict_with_onnxruntime(model_def, X)
        assert_almost_equal(
            np.array([[1.5, 0., 1., 1.5]], dtype=np.float32), Y['Y'])


if __name__ == "__main__":
    unittest.main()
