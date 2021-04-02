# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from distutils.version import StrictVersion
import onnx
import numpy
from numpy.random import rand
from numpy.testing import assert_almost_equal
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph, Fail
except ImportError:
    InvalidGraph = RuntimeError
    Fail = RuntimeError
from skl2onnx.common.data_types import FloatTensorType
try:
    from skl2onnx.algebra.onnx_ops import OnnxAbs, OnnxNormalizer, OnnxArgMin
    from skl2onnx.algebra.onnx_ops import OnnxSplit, OnnxScaler
except ImportError:
    warnings.warn(
        'Unable to test OnnxAbs, OnnxNormalizer, OnnxArgMin, OnnxSplit.')
    OnnxAbs = None
from test_utils import TARGET_OPSET


class TestAlgebraSymbolic(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(OnnxAbs is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_abs(self):

        op = OnnxAbs('I0', op_version=TARGET_OPSET)
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

    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
                     reason="not available")
    @unittest.skipIf(OnnxAbs is None,
                     reason="shape inference fails for Normalizer")
    def test_algebra_normalizer(self):
        op = OnnxNormalizer('I0', norm='L1', op_version=1,
                            output_names=['Y'])
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)},
                         outputs=[('Y', FloatTensorType())],
                         target_opset={'': 10})
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

    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
                     reason="not available")
    @unittest.skipIf(OnnxAbs is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_normalizer_shape(self):

        op = OnnxNormalizer('I0', norm='L1', op_version=1, output_names=['O0'])
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)},
                         outputs=[('O0', FloatTensorType((None, 2)))])
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
    @unittest.skipIf(OnnxAbs is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_argmin(self):

        op = OnnxArgMin('I0', op_version=TARGET_OPSET)
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        assert len(sonx) > 0

        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(OnnxAbs is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_normalizer_argmin_named_output(self):

        op = OnnxArgMin(
            OnnxNormalizer('I0', norm='L1', output_names=['Y']),
            op_version=TARGET_OPSET)
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        assert len(sonx) > 0

        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(OnnxAbs is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_normalizer_argmin(self):

        op = OnnxArgMin(
            OnnxNormalizer(
                'I0', norm='L1'),
            op_version=TARGET_OPSET)
        onx = op.to_onnx({'I0': numpy.ones((1, 2), dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        assert len(sonx) > 0

        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.array([[0, 2], [0, -2]])
        exp = numpy.array([[0, 1]])
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})[0]
        assert_almost_equal(exp, Y)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(OnnxAbs is None,
                     reason="Cannot infer operators with current ONNX")
    def test_algebra_split(self):

        op = OnnxSplit('I0', axis=0, output_names=['O1', 'O2'],
                       op_version=TARGET_OPSET)
        onx = op.to_onnx({'I0': numpy.arange(6, dtype=numpy.float32)})
        assert onx is not None
        sonx = str(onx)
        assert len(sonx) > 0

        import onnxruntime as ort
        sess = ort.InferenceSession(onx.SerializeToString())
        X = numpy.arange(6)
        exp = [numpy.array([0, 1, 2]), numpy.array([3, 4, 5])]
        Y = sess.run(None, {'I0': X.astype(numpy.float32)})
        assert len(Y) == len(exp)
        assert_almost_equal(exp[0], Y[0])
        assert_almost_equal(exp[1], Y[1])

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(OnnxAbs is None,
                     reason="Cannot infer operators with current ONNX")
    def test_cascade_scaler(self):

        def generate_onnx_graph(dim, nbnode, input_name='X1'):
            matrices = []
            scale = list(numpy.ones((1, dim)).ravel())
            i1 = input_name
            for i in range(nbnode - 1):
                i2 = list(rand(1, dim).ravel())
                matrices.append(i2)
                node = OnnxScaler(i1, offset=i2, scale=scale)
                i1 = node
            i2 = list(rand(1, dim).ravel())
            matrices.append(i2)
            node = OnnxScaler(
                i1, offset=i2, scale=scale, output_names=['Y'])
            onx = node.to_onnx([(input_name, FloatTensorType((None, dim)))],
                               outputs=[('Y', FloatTensorType((None, dim)))])
            return onx, matrices

        import onnxruntime as ort
        dim = 5
        for nbnode in range(1, 4):
            onx = generate_onnx_graph(dim, nbnode)[0]
            X = rand(1, dim)
            try:
                sess = ort.InferenceSession(onx.SerializeToString())
            except InvalidGraph as e:
                raise AssertionError(
                    "Loading error:\n{}\n{}".format(e, onx)) from e
            try:
                Y = sess.run(None, {'X1': X.astype(numpy.float32)})[0]
            except RuntimeError as e:
                raise RuntimeError("Run error:\n{}\n{}".format(e, onx))
            assert X.shape == Y.shape


if __name__ == "__main__":
    TestAlgebraSymbolic().test_algebra_normalizer()
    unittest.main()
