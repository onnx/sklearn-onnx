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

    def test_woe_transformer_conv(self):
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

        # from mlprodict.plotting.plotting import onnx_text_plot_tree
        # for node in onnx_model.graph.node:
        #     if node.op_type == 'TreeEnsembleRegressor':
        #         print(onnx_text_plot_tree(node))
        # from mlprodict.onnxrt import OnnxInference
        # oinf = OnnxInference(onnx_model)
        # oinf.run({"X": x}, verbose=10, fLOG=print)

        sess = InferenceSession(onnx_model.SerializeToString())
        got = sess.run(None, {'X': x})[0]
        assert_almost_equal(expected, got)


if __name__ == "__main__":
    unittest.main()
