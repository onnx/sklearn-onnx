# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-normalizer converter.
"""
import unittest
import numpy
from sklearn.preprocessing import Normalizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    Int64TensorType, FloatTensorType, DoubleTensorType)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnNormalizerConverter(unittest.TestCase):
    def test_model_normalizer(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model, "scikit-learn normalizer",
            [("input", Int64TensorType([None, 1]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)

    def test_model_normalizer_blackop(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model, "scikit-learn normalizer",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
            black_op={"Normalizer"})
        self.assertNotIn('op_type: "Normalizer', str(model_onnx))
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model, model_onnx,
            basename="SklearnNormalizerL1BlackOp-SkipDim1")

    def test_model_normalizer_float_l1(self):
        model = Normalizer(norm="l1")
        model_onnx = convert_sklearn(
            model, "scikit-learn normalizer",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model, model_onnx,
            basename="SklearnNormalizerL1-SkipDim1")

    def test_model_normalizer_float_l2(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model, "scikit-learn normalizer",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model, model_onnx,
            basename="SklearnNormalizerL2-SkipDim1")

    def test_model_normalizer_double_l1(self):
        model = Normalizer(norm="l1")
        model_onnx = convert_sklearn(
            model, "scikit-learn normalizer",
            [("input", DoubleTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float64),
            model, model_onnx,
            basename="SklearnNormalizerL1Double-SkipDim1")

    def test_model_normalizer_double_l2(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model, "scikit-learn normalizer",
            [("input", DoubleTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float64),
            model, model_onnx,
            basename="SklearnNormalizerL2Double-SkipDim1")

    def test_model_normalizer_float_noshape(self):
        model = Normalizer(norm="l2")
        model_onnx = convert_sklearn(
            model, "scikit-learn normalizer",
            [("input", FloatTensorType([]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model, model_onnx,
            basename="SklearnNormalizerL2NoShape-SkipDim1")


if __name__ == "__main__":
    unittest.main()
