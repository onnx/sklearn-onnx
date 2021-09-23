from distutils.version import StrictVersion
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        InvalidGraph, Fail, InvalidArgument)
except ImportError:
    InvalidGraph = RuntimeError
    InvalidArgument = RuntimeError
    Fail = RuntimeError
from skl2onnx.common.data_types import (
    Complex64TensorType, Complex128TensorType)
from skl2onnx.algebra.onnx_ops import OnnxAdd
from test_utils import TARGET_OPSET


class TestAlgebraComplex(unittest.TestCase):

    @unittest.skipIf(Complex64TensorType is None,
                     reason="not available")
    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion('1.8.0'),
                     reason="not implemented")
    def test_complex(self):
        for dt, var, pr in ((np.complex64, Complex64TensorType, 14),
                            (np.complex128, Complex128TensorType, 15)):
            X = np.array([[1-2j, -12j],
                          [-1-2j, 1+2j]]).astype(dt)

            for opv in range(10, 20):
                if opv > TARGET_OPSET:
                    continue
                with self.subTest(dt=dt, opset=opv):
                    out = OnnxAdd('X', np.array([1+2j], dtype=dt),
                                  output_names=['Y'], op_version=opv)
                    onx = out.to_onnx([('X', var((None, 2)))],
                                      outputs=[('Y', var())],
                                      target_opset=opv)
                    self.assertIn('elem_type: %d' % pr, str(onx))

                    try:
                        ort = InferenceSession(onx.SerializeToString())
                    except InvalidGraph as e:
                        if "Type Error: Type 'tensor(complex" in str(e):
                            continue
                        raise e
                    assert ort is not None
                    got = ort.run(None, {'X': X})[0]
                    assert_almost_equal(X + np.array([1+2j]), got)


if __name__ == "__main__":
    unittest.main()
