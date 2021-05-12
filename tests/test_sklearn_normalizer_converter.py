# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-normalizer converter.
"""
import unittest
import numpy
from sklearn.preprocessing import Normalizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnNormalizerConverter(unittest.TestCase):
    def test_model_normalizer(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", Int64TensorType([None, 1]))],
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)

    def test_model_normalizer_float(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", FloatTensorType([None, 3]))],
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, 1, 3], [3, 1, 2]], dtype=numpy.float32),
            model,
            model_onnx,
            basename="SklearnNormalizerL2-SkipDim1",
        )

    def test_model_normalizer_float_noshape(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", FloatTensorType([]))],
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, 1, 3], [3, 1, 2]], dtype=numpy.float32),
            model,
            model_onnx,
            basename="SklearnNormalizerL2-SkipDim1",
        )


if __name__ == "__main__":
    unittest.main()
