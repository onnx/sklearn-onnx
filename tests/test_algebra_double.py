import unittest
from distutils.version import StrictVersion
import onnx
import numpy
from numpy.testing import assert_almost_equal
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxSub
import onnxruntime
from onnxruntime import InferenceSession


class TestAlgebraDouble(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(StrictVersion(onnxruntime.__version__)
                     <= StrictVersion("0.4.0"),
                     reason="Sub(7) not available")
    def test_algebra_converter(self):

        coef = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        intercept = 1
        X_test = numpy.array([[1, -2], [3, -4]], dtype=numpy.float64)

        onnx_fct = OnnxSub(OnnxMatMul('X', coef),
                           numpy.array([intercept], dtype=numpy.float64),
                           output_names=['Y'])
        onnx_model = onnx_fct.to_onnx({'X': X_test}, dtype=numpy.float64)

        sess = InferenceSession(onnx_model.SerializeToString())
        ort_pred = sess.run(None, {'X': X_test})[0]
        assert_almost_equal(ort_pred,
                            numpy.array([[-6., -7.], [-10., -11.]]))


if __name__ == "__main__":
    unittest.main()
