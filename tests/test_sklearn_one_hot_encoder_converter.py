"""Tests scikit-learn's OneHotEncoder converter."""
import inspect
import unittest
import numpy
from distutils.version import StrictVersion
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    Int64TensorType,
    StringTensorType,
)
from test_utils import dump_data_and_model
from onnxruntime import __version__ as ort_version


def one_hot_encoder_supports_string():
    # StrictVersion does not work with development versions
    vers = '.'.join(sklearn_version.split('.')[:2])
    return StrictVersion(vers) >= StrictVersion("0.20.0")


class TestSklearnOneHotEncoderConverter(unittest.TestCase):
    def test_model_one_hot_encoder(self):
        # categorical_features will be removed in 0.22 (this test
        # will fail by then). FutureWarning: The handling of integer
        # data will change in version 0.22. Currently, the categories
        # are determined based on the range [0, max(values)], while
        # in the future they will be determined based on the unique values.
        # If you want the future behaviour and silence this warning,
        # you can specify "categories='auto'".
        model = OneHotEncoder()
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input", Int64TensorType([None, 3]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOneHotEncoderInt64-SkipDim1",
        )

    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19",
    )
    def test_one_hot_encoder_mixed_string_int(self):
        # categorical_features will be removed in 0.22
        # (this test will fail by then).
        data = [
            ["c0.4", "c0.2", 3],
            ["c1.4", "c1.2", 0],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
        ]
        model = OneHotEncoder(categories="auto")
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 2])),
            ("input2", Int64TensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(model,
                                     "one-hot encoder mixed-type inputs",
                                     inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOneHotEncoderStringInt64",
            verbose=False,
        )

    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19",
    )
    def test_one_hot_encoder_onecat(self):
        # categorical_features will be removed in 0.22
        # (this test will fail by then).
        data = [["cat"], ["cat"]]
        model = OneHotEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(model, "one-hot encoder one string cat",
                                     inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOneHotEncoderOneStringCat",
        )

    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19",
    )
    def test_one_hot_encoder_twocats(self):
        # categorical_features will be removed in 0.22
        # (this test will fail by then).
        data = [["cat2"], ["cat1"]]
        model = OneHotEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(model, "one-hot encoder two string cats",
                                     inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOneHotEncoderTwoStringCat",
        )

    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion("0.4.0"),
                     reason="issues with shapes")
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19",
    )
    def test_one_hot_encoder_one_string_one_int_cat(self):
        # categorical_features will be removed in 0.22
        # (this test will fail by then).
        data = [['Male', 1], ['Female', 3], ['Female', 2]]
        test_data = [['Unknown', 4]]
        sig = inspect.signature(OneHotEncoder)
        if "categorical_features" in sig.parameters:
            # scikit-learn < 0.21
            model = OneHotEncoder(handle_unknown='ignore',
                                  categorical_features='all')
        elif "categories" in sig.parameters:
            # scikit-learn >= 0.22
            model = OneHotEncoder(handle_unknown='ignore',
                                  categories='auto')
        else:
            raise AssertionError("scikit-learn's API has changed.")
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 1])),
            ("input2", Int64TensorType([None, 1]))
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder one string and int categories", inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test_data,
            model,
            model_onnx,
            basename="SklearnOneHotEncoderOneStringOneIntCat",
        )

    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support this in 0.19",
    )
    def test_model_one_hot_encoder_list_sparse(self):
        model = OneHotEncoder(categories=[[0, 1, 4, 5],
                                          [1, 2, 3, 5],
                                          [0, 3, 4, 6]],
                              sparse=True)
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input", Int64TensorType([None, 3]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOneHotEncoderCatSparse-SkipDim1",
        )

    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support this in 0.19",
    )
    def test_model_one_hot_encoder_list_dense(self):
        model = OneHotEncoder(categories=[[0, 1, 4, 5],
                                          [1, 2, 3, 5],
                                          [0, 3, 4, 6]],
                              sparse=False)
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input", Int64TensorType([None, 3]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOneHotEncoderCatDense-SkipDim1",
        )


if __name__ == "__main__":
    unittest.main()
