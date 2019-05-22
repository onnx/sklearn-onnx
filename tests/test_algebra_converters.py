import unittest
import warnings
from distutils.version import StrictVersion
import onnx
import numpy
from numpy.testing import assert_almost_equal
from sklearn.preprocessing import StandardScaler
try:
    from skl2onnx.algebra.sklearn_ops import OnnxSklearnStandardScaler
    from skl2onnx import wrap_as_onnx_mixin
except ImportError:
    warnings.warn('Unable to test OnnxSklearnScaler.')
    OnnxSklearnStandardScaler = None


class TestAlgebraConverters(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(OnnxSklearnStandardScaler is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_converter(self):

        X = numpy.array([[1, 2], [2, 3]])
        op = OnnxSklearnStandardScaler()
        op.fit(X)
        onx = op.to_onnx(X.astype(numpy.float32))
        assert onx is not None

        import onnxruntime as ort
        try:
            sess = ort.InferenceSession(onx.SerializeToString())
        except RuntimeError as e:
            raise RuntimeError("Unable to read\n{}".format(onx)) from e
        X = numpy.array([[0, 1], [-1, -2]])
        try:
            Y = sess.run(None, {'X': X.astype(numpy.float32)})[0]
        except RuntimeError as e:
            raise RuntimeError("Unable to run\n{}".format(onx)) from e
        assert_almost_equal(Y, op.transform(X))

        onx1 = str(onx)

        op = wrap_as_onnx_mixin(StandardScaler())
        op = OnnxSklearnStandardScaler()
        op.fit(X)
        onx = op.to_onnx(X.astype(numpy.float32))
        onx2 = str(onx)
        assert 'domain: "ai.onnx.ml"' in onx1
        assert 'domain: "ai.onnx.ml"' in onx2

        try:
            sess = ort.InferenceSession(onx.SerializeToString())
        except RuntimeError as e:
            raise RuntimeError("Unable to read\n{}".format(onx)) from e
        X = numpy.array([[0, 1], [-1, -2]])
        try:
            Y = sess.run(None, {'X': X.astype(numpy.float32)})[0]
        except RuntimeError as e:
            raise RuntimeError("Unable to run\n{}".format(onx)) from e
        assert_almost_equal(Y, op.transform(X))


if __name__ == "__main__":
    unittest.main()
