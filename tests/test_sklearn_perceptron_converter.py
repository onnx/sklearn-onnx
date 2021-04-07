# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's Perceptron converter."""

import unittest
import numpy as np
from sklearn.linear_model import Perceptron
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    TARGET_OPSET
)


class TestPerceptronClassifierConverter(unittest.TestCase):

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_perceptron_binary_class(self):
        model, X = fit_classification_model(
            Perceptron(random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn Perceptron binary classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnPerceptronClassifierBinary-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_perceptron_multi_class(self):
        model, X = fit_classification_model(
            Perceptron(random_state=42), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn Perceptron multi-class classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnPerceptronClassifierMulti-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_perceptron_binary_class_int(self):
        model, X = fit_classification_model(
            Perceptron(random_state=42), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn Perceptron binary classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnPerceptronClassifierBinaryInt-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_perceptron_multi_class_int(self):
        model, X = fit_classification_model(
            Perceptron(random_state=42), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn Perceptron multi-class classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnPerceptronClassifierMultiInt-Out0",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
