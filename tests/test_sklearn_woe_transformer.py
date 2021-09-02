# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's cast transformer converter.
"""
import unittest
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    ColumnTransformer = None
from skl2onnx.sklapi import WOETransformer
import skl2onnx.sklapi.register  # noqa
from skl2onnx import to_onnx
from test_utils import TARGET_OPSET


class TestSklearnWOETransformerConverter(unittest.TestCase):

    def test_woe_transformer(self):
        x = numpy.array(
            [[0.5, 0.7, 0.9], [0.51, 0.71, 0.91], [0.7, 0.5, 0.92]],
            dtype=numpy.float32)
        woe = WOETransformer(intervals=[
            [(0.5, 0.7, False, False),
             (0.5, 0.7, True, False),
             (0.5, 0.7, False, True),
             (0.5, 0.7, True, True)],
            [(0.9, numpy.inf),
             (-numpy.inf, 0.9)]])
        woe.fit(x)
        self.assertEqual(woe.indices_, [(0, 4), (4, 6), (6, 7)])
        self.assertEqual(woe.n_dims_, 7)
        self.assertEqual(woe.intervals_, [
            [(0.5, 0.7, False, False),
             (0.5, 0.7, True, False),
             (0.5, 0.7, False, True),
             (0.5, 0.7, True, True)],
            [(0.9, numpy.inf, False, True),
             (-numpy.inf, 0.9, False, True)],
            None])
        names = woe.get_feature_names()
        self.assertEqual(
            names,
            [']0.5,0.7[', '[0.5,0.7[', ']0.5,0.7]', '[0.5,0.7]',
             ']0.9,inf]', ']-inf,0.9]', 'X2'])
        x2 = woe.transform(x)
        expected = numpy.array(
            [[0, 1, 0, 1, 0, 1, 0.9],
             [1, 1, 1, 1, 0, 1, 0.91],
             [0, 0, 1, 1, 0, 1, 0.92]],
            dtype=numpy.float32)
        assert_almost_equal(expected, x2)

    def test_woe_transformer_conv_ext(self):
        x = numpy.array(
            [[0.4, 1.4, 2.4, 3.4],
             [0.5, 1.5, 2.5, 3.5],
             [0.6, 1.6, 2.6, 3.6],
             [0.7, 1.7, 2.7, 3.7]],
            dtype=numpy.float32)
        woe = WOETransformer(intervals=[
            [(0.4, 0.6, False, False)],
            [(1.4, 1.6, False, True)],
            [(2.4, 2.6, True, False)],
            [(3.4, 3.6, True, True)]])
        woe.fit(x)
        expected = woe.transform(x)
        onnx_model = to_onnx(woe, x, target_opset=TARGET_OPSET)
        sess = InferenceSession(onnx_model.SerializeToString())
        got = sess.run(None, {'X': x})[0]
        assert_almost_equal(expected, got)

    def test_woe_transformer_conv_ext2(self):
        for inca, incb in [(False, False), (True, True),
                           (False, True), (True, False)]:
            with self.subTest(inca=inca, incb=incb):
                x = numpy.array([[0.45], [0.5], [0.55]], dtype=numpy.float32)
                woe = WOETransformer(intervals=[
                    [(0.4, 0.5, False, inca), (0.5, 0.6, incb, False)]])
                woe.fit(x)
                expected = woe.transform(x)
                onnx_model = to_onnx(
                    woe, x, target_opset=TARGET_OPSET, verbose=0)
                sess = InferenceSession(onnx_model.SerializeToString())
                got = sess.run(None, {'X': x})[0]
                assert_almost_equal(expected, got)

    def test_woe_transformer_conv_ext3(self):
        x = numpy.array(
            [[0.4, 1.4, 2.4, 3.4],
             [0.5, 1.5, 2.5, 3.5],
             [0.6, 1.6, 2.6, 3.6]],
            dtype=numpy.float32)
        woe = WOETransformer(intervals=[
            [(0.4, 0.5, False, False), (0.5, 0.6, False, False)],
            [(1.4, 1.5, False, True), (1.5, 1.6, False, True)],
            [(2.4, 2.5, True, False), (2.5, 2.6, True, False)],
            [(3.4, 3.5, True, True), (3.5, 3.6, True, True)]])
        woe.fit(x)
        expected = woe.transform(x)
        onnx_model = to_onnx(woe, x, target_opset=TARGET_OPSET)
        sess = InferenceSession(onnx_model.SerializeToString())
        got = sess.run(None, {'X': x})[0]
        assert_almost_equal(expected, got)

    def _test_woe_transformer_conv(self):
        x = numpy.array(
            [[0.2, 0.7, 0.9],
             [0.51, 0.71, 0.91],
             [0.7, 1.5, 0.92]],
            dtype=numpy.float32)
        woe = WOETransformer(intervals=[
            [(0.4, 0.6, False, True)],
            [(0.9, numpy.inf), (-numpy.inf, 0.9)]])
        woe.fit(x)
        expected = woe.transform(x)

        onnx_model = to_onnx(woe, x, target_opset=TARGET_OPSET)

        with open("debug.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        sess = InferenceSession(onnx_model.SerializeToString())
        got = sess.run(None, {'X': x})[0]
        assert_almost_equal(expected, got)


if __name__ == "__main__":
    unittest.main()
