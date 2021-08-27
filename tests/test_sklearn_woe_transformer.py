# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's cast transformer converter.
"""
import unittest
import numpy
from numpy.testing import assert_almost_equal
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    ColumnTransformer = None
from skl2onnx.sklapi import WOETransformer
# from skl2onnx import to_onnx
# from test_utils import dump_data_and_model, TARGET_OPSET


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


if __name__ == "__main__":
    unittest.main()
