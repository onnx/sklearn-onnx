import unittest
import onnx
import numpy
from numpy.testing import assert_almost_equal
from skl2onnx.algebra.onnx_ops import Abs
from skl2onnx.algebra import OnnxOperator, Symbolic


class TestAlgebraSymbolic(unittest.TestCase):
    
    def test_algebra_abs(self):
    
        op = Abs(Symbolic.Input('I0'))
        onx = op.to_onnx({'I0': numpy.empty((1, 2), dtype=numpy.float32)})
        assert onx is not None
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 1], [-1, -2]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(Y, numpy.abs(X))


if __name__ == "__main__":
    unittest.main()
