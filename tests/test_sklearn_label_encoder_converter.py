"""Tests scikit-LabelEncoder converter"""

import unittest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


class TestSklearnLabelEncoderConverter(unittest.TestCase):
    def test_model_label_encoder(self):
        model = LabelEncoder()
        data = ["str3", "str2", "str0", "str1", "str3"]
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label encoder",
            [("input", StringTensorType([None, 1]))],
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        dump_data_and_model(
            np.array(data),
            model,
            model_onnx,
            basename="SklearnLabelEncoder",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    def test_model_label_encoder_float(self):
        model = LabelEncoder()
        data = np.array([1.2, 3.4, 5.4, 1.2])
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label encoder",
            [("input", StringTensorType([1, 1]))],
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnLabelEncoderFloat",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    def test_model_label_encoder_int(self):
        model = LabelEncoder()
        data = np.array([10, 3, 5, -34, 0])
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label encoder",
            [("input", StringTensorType([1, 1]))],
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnLabelEncoderInt",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )


if __name__ == "__main__":
    unittest.main()
