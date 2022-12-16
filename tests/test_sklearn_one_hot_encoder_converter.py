# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's OneHotEncoder converter."""
import unittest
import packaging.version as pv
import numpy
from onnxruntime import __version__ as ort_version
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    Int32TensorType,
    Int64TensorType,
    StringTensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET


def one_hot_encoder_supports_string():
    # pv.Version does not work with development versions
    vers = '.'.join(sklearn_version.split('.')[:2])
    return pv.Version(vers) >= pv.Version("0.20.0")


def one_hot_encoder_supports_drop():
    # pv.Version does not work with development versions
    vers = '.'.join(sklearn_version.split('.')[:2])
    return pv.Version(vers) >= pv.Version("0.21.0")


class TestSklearnOneHotEncoderConverter(unittest.TestCase):
    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20")
    def test_model_one_hot_encoder(self):
        model = OneHotEncoder(categories='auto')
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn one-hot encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderInt64-SkipDim1")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20")
    def test_model_one_hot_encoder_int32(self):
        model = OneHotEncoder(categories='auto')
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int32)
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn one-hot encoder",
            [("input", Int32TensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        str_model_onnx = str(model_onnx)
        assert "int64_data" in str_model_onnx
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderInt32-SkipDim1")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20")
    def test_model_one_hot_encoder_int32_scaler(self):
        model = make_pipeline(OneHotEncoder(categories='auto', sparse=False),
                              RobustScaler())
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int32)
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn one-hot encoder",
            [("input", Int32TensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        str_model_onnx = str(model_onnx)
        assert "int64_data" in str_model_onnx
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderInt32Scaler-SkipDim1")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21")
    def test_one_hot_encoder_mixed_string_int_drop(self):
        data = [
            ["c0.4", "c0.2", 3],
            ["c1.4", "c1.2", 0],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
        ]
        test = [["c0.2", "c2.2", 1]]
        model = OneHotEncoder(categories="auto", drop=['c0.4', 'c0.2', 3])
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 2])),
            ("input2", Int64TensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test, model, model_onnx, verbose=False,
            basename="SklearnOneHotEncoderMixedStringIntDrop")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19")
    def test_one_hot_encoder_onecat(self):
        data = [["cat"], ["cat"]]
        model = OneHotEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(model, "one-hot encoder one string cat",
                                     inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderOneStringCat")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19")
    def test_one_hot_encoder_twocats(self):
        data = [["cat2"], ["cat1"]]
        model = OneHotEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(model, "one-hot encoder two string cats",
                                     inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderTwoStringCat")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21")
    def test_one_hot_encoder_string_drop_first(self):
        data = [['Male', 'First'], ['Female', 'First'], ['Female', 'Second']]
        test_data = [['Male', 'Second']]
        model = OneHotEncoder(drop='first',
                              categories='auto')
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 1])),
            ("input2", StringTensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test_data, model, model_onnx,
            basename="SklearnOneHotEncoderStringDropFirst")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support this in 0.19")
    def test_model_one_hot_encoder_list_sparse(self):
        model = OneHotEncoder(categories=[[0, 1, 4, 5],
                                          [1, 2, 3, 5],
                                          [0, 3, 4, 6]],
                              sparse=True)
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn one-hot encoder",
            [("input1", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderCatSparse-SkipDim1")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support this in 0.19")
    def test_model_one_hot_encoder_list_dense(self):
        model = OneHotEncoder(categories=[[0, 1, 4, 5],
                                          [1, 2, 3, 5],
                                          [0, 3, 4, 6]],
                              sparse=False)
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn one-hot encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderCatDense-SkipDim1")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21")
    def test_one_hot_encoder_int_drop(self):
        data = [
            [1, 2, 3],
            [4, 1, 0],
            [0, 2, 1],
            [2, 2, 1],
            [0, 4, 0],
            [0, 3, 3],
        ]
        test = numpy.array([[2, 2, 1], [4, 2, 1]], dtype=numpy.int64)
        model = OneHotEncoder(categories="auto", drop=[0, 1, 3],
                              dtype=numpy.float32)
        model.fit(data)
        inputs = [
            ("input1", Int64TensorType([None, 3])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs,
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test, model, model_onnx,
            basename="SklearnOneHotEncoderIntDrop")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21")
    def test_one_hot_encoder_int_drop_first(self):
        data = [
            [1, 2, 3],
            [4, 1, 0],
            [0, 2, 1],
            [2, 2, 1],
            [0, 4, 0],
            [0, 3, 3],
        ]
        test = numpy.array([[2, 2, 1], [1, 3, 3]], dtype=numpy.int64)
        model = OneHotEncoder(categories="auto", drop='first',
                              dtype=numpy.int64)
        model.fit(data)
        inputs = [
            ("input1", Int64TensorType([None, 3])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test, model, model_onnx,
            basename="SklearnOneHotEncoderIntDropFirst")

    @unittest.skipIf(pv.Version(ort_version) <= pv.Version("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21")
    def test_one_hot_encoder_string_drop_first_2(self):
        data = [['Male', 'First'], ['Female', 'First'], ['Female', 'Second']]
        model = OneHotEncoder(drop='first')
        model.fit(data)
        inputs = [
            ("input", StringTensorType([None, 2])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnOneHotEncoderStringDropFirst2")


if __name__ == "__main__":
    unittest.main()
