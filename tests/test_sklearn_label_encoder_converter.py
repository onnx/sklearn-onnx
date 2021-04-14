# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-LabelEncoder converter"""

import unittest
from distutils.version import StrictVersion
import numpy as np
import onnxruntime
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnLabelEncoderConverter(unittest.TestCase):

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.3.0"),
        reason="onnxruntime too old")
    def test_model_label_encoder(self):
        model = LabelEncoder()
        data = ["str3", "str2", "str0", "str1", "str3"]
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label encoder",
            [("input", StringTensorType([None]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        if model_onnx.ir_version >= 7 and TARGET_OPSET < 12:
            raise AssertionError("Incompatbilities")
        dump_data_and_model(
            np.array(data),
            model,
            model_onnx,
            basename="SklearnLabelEncoder",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.3.0"),
        reason="onnxruntime too old")
    def test_model_label_encoder_float(self):
        model = LabelEncoder()
        data = np.array([1.2, 3.4, 5.4, 1.2], dtype=np.float32)
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label encoder",
            [("input", FloatTensorType([None]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        if model_onnx.ir_version >= 7 and TARGET_OPSET < 12:
            raise AssertionError("Incompatbilities")
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnLabelEncoderFloat",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.3.0"),
        reason="onnxruntime too old")
    def test_model_label_encoder_int(self):
        model = LabelEncoder()
        data = np.array([10, 3, 5, -34, 0], dtype=np.int64)
        model.fit(data)
        for op in sorted(set([9, 10, 11, 12, TARGET_OPSET])):
            if op > TARGET_OPSET:
                continue
            with self.subTest(opset=op):
                model_onnx = convert_sklearn(
                    model,
                    "scikit-learn label encoder",
                    [("input", Int64TensorType([None]))],
                    target_opset=op)
                self.assertTrue(model_onnx is not None)
                self.assertTrue(model_onnx.graph.node is not None)
                if model_onnx.ir_version >= 7 and TARGET_OPSET < 12:
                    raise AssertionError("Incompatbilities")
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
