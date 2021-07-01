# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's OrdinalEncoder converter."""
import unittest
from distutils.version import StrictVersion
import numpy as np
import onnx
import onnxruntime
from sklearn import __version__ as sklearn_version
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    pass
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    Int64TensorType,
    StringTensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET


def ordinal_encoder_support():
    # StrictVersion does not work with development versions
    vers = '.'.join(sklearn_version.split('.')[:2])
    if StrictVersion(vers) < StrictVersion("0.20.0"):
        return False
    if StrictVersion(onnxruntime.__version__) < StrictVersion("0.3.0"):
        return False
    return StrictVersion(vers) >= StrictVersion("0.20.0")


class TestSklearnOrdinalEncoderConverter(unittest.TestCase):
    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_model_ordinal_encoder(self):
        model = OrdinalEncoder(dtype=np.int64)
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                        dtype=np.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn ordinal encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOrdinalEncoderInt64-SkipDim1",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_ordinal_encoder_mixed_string_int_drop(self):
        data = [
            ["c0.4", "c0.2", 3],
            ["c1.4", "c1.2", 0],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
        ]
        test = [["c0.2", "c2.2", 1]]
        model = OrdinalEncoder(categories="auto")
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 2])),
            ("input2", Int64TensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "ordinal encoder", inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test,
            model,
            model_onnx,
            basename="SklearnOrdinalEncoderMixedStringIntDrop",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_ordinal_encoder_onecat(self):
        data = [["cat"], ["cat"]]
        model = OrdinalEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(model, "ordinal encoder one string cat",
                                     inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOrdinalEncoderOneStringCat",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_ordinal_encoder_twocats(self):
        data = [["cat2"], ["cat1"]]
        model = OrdinalEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(model, "ordinal encoder two string cats",
                                     inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
            basename="SklearnOrdinalEncoderTwoStringCat",
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_model_ordinal_encoder_cat_list(self):
        model = OrdinalEncoder(categories=[[0, 1, 4, 5],
                                           [1, 2, 3, 5],
                                           [0, 3, 4, 6]])
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                        dtype=np.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn ordinal encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOrdinalEncoderCatList",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )


if __name__ == "__main__":
    unittest.main()
