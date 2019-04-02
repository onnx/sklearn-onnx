import unittest
import warnings
from distutils.version import StrictVersion
import onnx
import numpy
from numpy.testing import assert_almost_equal
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra import OnnxOperator
try:
    from skl2onnx.algebra.onnx_ops import Abs, Normalizer, ArgMin, Split, Identity
except ImportError as e:
    warnings.warn('Unable to test Abs, Normalizer, ArgMin, Split.')
    Abs = None


class TestAlgebraSymbolic(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(Abs is None, reason="Cannot infer operators with current ONNX")
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

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(True or Abs is None,
                     reason="shape inference fails for Normalizer")
    def test_algebra_normalizer(self):
    
        op = Normalizer('I0', norm='L1', op_version=1)
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        assert "ai.onnx.ml" in sonx
        assert "version: 1" in sonx
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1], [0, -1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(Abs is None, reason="Cannot infer operators with current ONNX")
    def test_algebra_normalizer_shape(self):
    
        op = Normalizer('I0', norm='L1', op_version=1, output_names=['O0'])
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)},
                         outputs=[('O0', FloatTensorType((1, 2)))])
        assert onx is not None
        sonx = str(onx)
        assert "ai.onnx.ml" in sonx
        assert "version: 1" in sonx
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1], [0, -1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(Abs is None, reason="Cannot infer operators with current ONNX")
    def test_algebra_argmin(self):
    
        op = ArgMin('I0', op_version=1)
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(Abs is None, reason="Cannot infer operators with current ONNX")
    def test_algebra_normalizer_argmin_named_output(self):
    
        op = ArgMin(Normalizer('I0', norm='L1', output_names=['Y']))
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(Abs is None, reason="Cannot infer operators with current ONNX")
    def test_algebra_normalizer_argmin(self):
    
        op = ArgMin(Normalizer('I0', norm='L1'))
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(Abs is None, reason="Cannot infer operators with current ONNX")
    def test_algebra_split(self):
    
        op = Split('I0', axis=0, output_names=['O1', 'O2'])
        onx = op.to_onnx({'I0': numpy.arange(6, dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.arange(6)
        exp = [numpy.array([0, 1, 2]), numpy.array([3, 4, 5])]
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})
        assert len(Y) == len(exp)
        assert_almost_equal(exp[0], Y[0])
        assert_almost_equal(exp[1], Y[1])


if __name__ == "__main__":
    unittest.main()
