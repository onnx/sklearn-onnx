import unittest
import onnx
import numpy
from numpy.testing import assert_almost_equal
from skl2onnx.algebra.onnx_ops import Abs, Normalizer, ArgMin
from skl2onnx.algebra import OnnxOperator


class TestAlgebraSymbolic(unittest.TestCase):
    
    def test_algebra_abs(self):
    
        op = Abs('I0')
        onx = op.to_onnx({'I0': numpy.empty((1, 2), dtype=numpy.float32)})
        assert onx is not None
        
        import onnxruntime as ort
        try:
            sess = ort.InferenceSession(onx.SerializeToString())
        except RuntimeError as e:
            raise RuntimeError("Unable to read\n{}".format(onx)) from e
        X = numpy.array([[0, 1], [-1, -2]])
        try:
            Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        except RuntimeError as e:
            raise RuntimeError("Unable to run\n{}".format(onx)) from e
        assert_almost_equal(Y, numpy.abs(X))

    def test_algebra_normalizer(self):
    
        op = Normalizer('I0', norm='L1', op_version=1)
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        assert "ai.onnx.ml" in sonx
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1], [0, -1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    def test_algebra_argmin(self):
    
        op = ArgMin('I0', op_version=1)
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        if '7' not in sonx:
            raise TypeError("Wrong output type:\n" + sonx)
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)


if __name__ == "__main__":
    unittest.main()
