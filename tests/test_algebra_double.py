# SPDX-License-Identifier: Apache-2.0

import unittest
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxSub
import onnxruntime
from onnxruntime import InferenceSession
from test_utils import TARGET_OPSET


class TestAlgebraDouble(unittest.TestCase):
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(onnxruntime.__version__) <= pv.Version("0.4.0"),
        reason="Sub(7) not available",
    )
    def test_algebra_converter(self):
        coef = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        intercept = 1
        X_test = numpy.array([[1, -2], [3, -4]], dtype=numpy.float64)

        onnx_fct = OnnxSub(
            OnnxMatMul("X", coef, op_version=TARGET_OPSET),
            numpy.array([intercept], dtype=numpy.float64),
            output_names=["Y"],
            op_version=TARGET_OPSET,
        )
        onnx_model = onnx_fct.to_onnx({"X": X_test}, target_opset=TARGET_OPSET)

        sess = InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_pred = sess.run(None, {"X": X_test})[0]
        assert_almost_equal(ort_pred, numpy.array([[-6.0, -7.0], [-10.0, -11.0]]))


if __name__ == "__main__":
    unittest.main()
