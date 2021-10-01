# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
from skl2onnx.algebra.onnx_ops import (
    OnnxReduceSumApi11,
    OnnxSplitApi11,
    OnnxSqueezeApi11,
    OnnxUnsqueezeApi11)
from test_utils import TARGET_OPSET


class TestOpset13(unittest.TestCase):

    def test_reduce_sum(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=numpy.float32)

        for opset in range(10, 20):
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                onx = OnnxReduceSumApi11(
                    'X', output_names=['Y'], keepdims=0, op_version=opset)
                model_def = onx.to_onnx(
                    {'X': X.astype(numpy.float32)}, target_opset=opset)
                got = InferenceSession(model_def.SerializeToString()).run(
                    None, {'X': X})
                assert_almost_equal(numpy.sum(X), got[0], decimal=6)
                onx = OnnxReduceSumApi11(
                    'X', output_names=['Y'], axes=[1], op_version=opset)
                model_def = onx.to_onnx(
                    {'X': X.astype(numpy.float32)}, target_opset=opset)
                got = InferenceSession(model_def.SerializeToString()).run(
                    None, {'X': X})
                assert_almost_equal(
                    numpy.sum(X, axis=1, keepdims=True), got[0], decimal=6)

    def test_split(self):
        x = numpy.array([1., 2., 3., 4., 5., 6.]).astype(numpy.float32)
        y = [numpy.array([1., 2.]).astype(numpy.float32),
             numpy.array([3., 4.]).astype(numpy.float32),
             numpy.array([5., 6.]).astype(numpy.float32)]

        for opset in (10, 11, 12, 13):
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                onx = OnnxSplitApi11(
                    'X', axis=0, split=[2, 2, 2],
                    output_names=['Y1', 'Y2', 'Y3'], op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x.astype(numpy.float32)}, target_opset=opset)
                got = InferenceSession(model_def.SerializeToString()).run(
                    None, {'X': x})
                assert_almost_equal(y[0], got[0])
                assert_almost_equal(y[1], got[1])
                assert_almost_equal(y[2], got[2])

    def test_squeeze(self):
        x = numpy.random.randn(20, 1).astype(numpy.float32)
        y = numpy.squeeze(x)
        for opset in range(10, 20):
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                onx = OnnxSqueezeApi11(
                    'X', axes=[1], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x.astype(numpy.float32)}, target_opset=opset)
                got = InferenceSession(model_def.SerializeToString()).run(
                    None, {'X': x})
                assert_almost_equal(y, got[0])

    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.0.0'),
                     reason="onnxruntime too old, onnx too recent")
    def test_unsqueeze(self):
        x = numpy.random.randn(1, 3, 1, 5).astype(numpy.float32)
        y = numpy.expand_dims(x, axis=-2)
        for opset in (10, 11, 12, 13):
            if opset > TARGET_OPSET:
                continue
            with self.subTest(opset=opset):
                onx = OnnxUnsqueezeApi11(
                    'X', axes=[-2], output_names=['Y'], op_version=opset)
                model_def = onx.to_onnx(
                    {'X': x.astype(numpy.float32)}, target_opset=opset)
                got = InferenceSession(model_def.SerializeToString()).run(
                    None, {'X': x})
                assert_almost_equal(y, got[0])


if __name__ == "__main__":
    unittest.main()
