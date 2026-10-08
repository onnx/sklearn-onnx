# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-normalizer converter.
"""

import unittest
import numpy
from sklearn.preprocessing import Normalizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    Int64TensorType,
    FloatTensorType,
    DoubleTensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnNormalizerConverter(unittest.TestCase):
    def test_model_normalizer(self):
        model = Normalizer(norm="l2")
        x = numpy.random.randn(10, 1).astype(numpy.int64)
        model.fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", Int64TensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)

    def test_model_normalizer_blackop(self):
        model = Normalizer(norm="l2")
        x = numpy.random.randn(10, 3).astype(numpy.float32)
        model.fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
            black_op={"Normalizer"},
        )
        self.assertNotIn('op_type: "Normalizer', str(model_onnx))
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model,
            model_onnx,
            basename="SklearnNormalizerL1BlackOp-SkipDim1",
        )

    def test_model_normalizer_float_l1(self):
        model = Normalizer(norm="l1")
        x = numpy.random.randn(10, 3).astype(numpy.float32)
        model.fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model,
            model_onnx,
            basename="SklearnNormalizerL1-SkipDim1",
        )

    def test_model_normalizer_float_l2(self):
        model = Normalizer(norm="l2")
        x = numpy.random.randn(10, 3).astype(numpy.float32)
        model.fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model,
            model_onnx,
            basename="SklearnNormalizerL2-SkipDim1",
        )

    def test_model_normalizer_double_l1(self):
        model = Normalizer(norm="l1")
        x = numpy.random.randn(10, 3).astype(numpy.float64)
        model.fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", DoubleTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float64),
            model,
            model_onnx,
            basename="SklearnNormalizerL1Double-SkipDim1",
        )

    def test_model_normalizer_double_l2(self):
        model = Normalizer(norm="l2")
        x = numpy.random.randn(10, 3).astype(numpy.float64)
        model.fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", DoubleTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float64),
            model,
            model_onnx,
            basename="SklearnNormalizerL2Double-SkipDim1",
        )

    def test_model_normalizer_max_and_zero_rows(self):
        # norm="max" divides by the maximum absolute value, and a row of
        # zeros stays zeros
        x = numpy.array(
            [[-3, 1, 2], [0, 0, 0], [1, -0.5, 0.25], [-1, -2, -4]],
            dtype=numpy.float64,
        )
        for norm in ["max", "l1", "l2"]:
            for dtype, tensor_type in [
                (numpy.float32, FloatTensorType),
                (numpy.float64, DoubleTensorType),
            ]:
                with self.subTest(norm=norm, dtype=dtype):
                    model = Normalizer(norm=norm).fit(x)
                    model_onnx = convert_sklearn(
                        model,
                        "scikit-learn normalizer",
                        [("input", tensor_type([None, 3]))],
                        target_opset=TARGET_OPSET,
                    )
                    dump_data_and_model(
                        x.astype(dtype),
                        model,
                        model_onnx,
                        basename=f"SklearnNormalizer{norm}{dtype.__name__}Zero",
                    )

    def test_model_normalizer_max_int64(self):
        x = numpy.array([[-3, 1, 2], [0, 0, 0], [1, -5, 4]], dtype=numpy.int64)
        model = Normalizer(norm="max").fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(x, model, model_onnx, basename="SklearnNormalizerMaxInt64")

    def test_model_normalizer_tiny_rows(self):
        # scikit-learn leaves rows whose norm is below 10 * eps unchanged
        for norm, dtype, tensor_type in [
            ("max", numpy.float32, FloatTensorType),
            ("max", numpy.float64, DoubleTensorType),
            ("l1", numpy.float64, DoubleTensorType),
            ("l2", numpy.float64, DoubleTensorType),
        ]:
            eps = numpy.finfo(dtype).eps
            x = numpy.array(
                [[eps, 0, -eps], [1, -0.5, 0.25], [eps * 100, 0, 0]], dtype=dtype
            )
            with self.subTest(norm=norm, dtype=dtype):
                model = Normalizer(norm=norm).fit(x)
                model_onnx = convert_sklearn(
                    model,
                    "scikit-learn normalizer",
                    [("input", tensor_type([None, 3]))],
                    target_opset=TARGET_OPSET,
                )
                dump_data_and_model(
                    x,
                    model,
                    model_onnx,
                    basename=f"SklearnNormalizer{norm}{dtype.__name__}Tiny",
                )

    def test_model_normalizer_float_noshape(self):
        model = Normalizer(norm="l2")
        x = numpy.random.randn(10, 3).astype(numpy.float32)
        model.fit(x)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn normalizer",
            [("input", FloatTensorType([]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(
            numpy.array([[1, -1, 3], [3, 1, 2]], dtype=numpy.float32),
            model,
            model_onnx,
            basename="SklearnNormalizerL2NoShape-SkipDim1",
        )


if __name__ == "__main__":
    unittest.main()
