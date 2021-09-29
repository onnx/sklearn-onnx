# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import onnx
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.preprocessing import StandardScaler
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxExp, OnnxAdd, OnnxDiv
try:
    from skl2onnx.algebra.sklearn_ops import OnnxSklearnStandardScaler
    from skl2onnx import wrap_as_onnx_mixin
except (ImportError, KeyError):
    OnnxSklearnStandardScaler = None
from test_utils import TARGET_OPSET


class TestAlgebraConverters(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(OnnxSklearnStandardScaler is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_converter(self):

        X = numpy.array([[1, 2], [2, 3]])
        op = OnnxSklearnStandardScaler()
        op.fit(X)
        onx = op.to_onnx(X.astype(numpy.float32), target_opset=TARGET_OPSET)
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
        onx = op.to_onnx(X.astype(numpy.float32), target_opset=TARGET_OPSET)
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
        onnxExpM = OnnxExp(
            OnnxMatMul('X', beta32, op_version=TARGET_OPSET),
            op_version=TARGET_OPSET)
        cst = numpy.ones((1, 3), dtype=numpy.float32)
        onnxExpM1 = OnnxAdd(onnxExpM, cst, op_version=TARGET_OPSET)
        onnxPred = OnnxDiv(onnxExpM, onnxExpM1, op_version=TARGET_OPSET)
        inputs = {'X': X[:1].astype(numpy.float32)}
        model_onnx = onnxPred.to_onnx(inputs, target_opset=TARGET_OPSET)
        s1 = str(model_onnx)
        model_onnx = onnxPred.to_onnx(inputs, target_opset=TARGET_OPSET)
        s2 = str(model_onnx)
        assert s1 == s2
        nin = list(onnxExpM1.enumerate_initial_types())
        nno = list(onnxExpM1.enumerate_nodes())
        nva = list(onnxExpM1.enumerate_variables())
        self.assertEqual(len(nin), 0)
        self.assertEqual(len(nno), 3)
        self.assertEqual(len(nva), 0)

    def test_add_12(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'], op_version=12)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)},
                                target_opset=12)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        sess = InferenceSession(model_def.SerializeToString())
        got = sess.run(None, {'X': X})
        exp = idi + X
        self.assertEqual(exp.shape, got[0].shape)
        self.assertEqual(list(exp.ravel()), list(got[0].ravel()))
        self.assertIn("version: 7", str(model_def))


if __name__ == "__main__":
    unittest.main()
