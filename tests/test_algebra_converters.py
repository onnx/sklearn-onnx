import unittest
import warnings
from distutils.version import StrictVersion
import onnx
import numpy
from numpy.testing import assert_almost_equal
from sklearn.preprocessing import StandardScaler
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxExp, OnnxAdd, OnnxDiv
try:
    from skl2onnx.algebra.sklearn_ops import OnnxSklearnStandardScaler
    from skl2onnx import wrap_as_onnx_mixin
except (ImportError, KeyError):
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

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    def test_algebra_to_onnx(self):
        X = numpy.random.randn(5, 4)
        beta = numpy.array([1, 2, 3, 4]) / 10
        beta32 = beta.astype(numpy.float32)
        onnxExpM = OnnxExp(OnnxMatMul('X', beta32))
        cst = numpy.ones((1, 3), dtype=numpy.float32)
        onnxExpM1 = OnnxAdd(onnxExpM, cst)
        onnxPred = OnnxDiv(onnxExpM, onnxExpM1)
        inputs = {'X': X[:1].astype(numpy.float32)}
        model_onnx = onnxPred.to_onnx(inputs)
        s1 = str(model_onnx)
        model_onnx = onnxPred.to_onnx(inputs)
        s2 = str(model_onnx)
        assert s1 == s2
        nin = list(onnxExpM1.enumerate_initial_types())
        nno = list(onnxExpM1.enumerate_nodes())
        nva = list(onnxExpM1.enumerate_variables())
        self.assertEqual(len(nin), 0)
        self.assertEqual(len(nno), 3)
        self.assertEqual(len(nva), 0)


if __name__ == "__main__":
    unittest.main()
