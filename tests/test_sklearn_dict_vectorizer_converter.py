# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's DictVectorizer converter."""

import unittest
import numpy
from numpy.testing import assert_almost_equal
import onnx.checker
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DictionaryType,
    StringTensorType,
    FloatTensorType,
    Int64TensorType,
    BooleanTensorType,
)
from test_utils import (
    dump_data_and_model,
    TARGET_OPSET,
    InferenceSessionEx as InferenceSession,
)

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph
except ImportError:
    InvalidArgument = None
    InvalidGraph = None


class TestSklearnDictVectorizerConverter(unittest.TestCase):
    def test_model_dict_vectorizer(self):
        model = DictVectorizer()
        data = [{"amy": 1.0, "chin": 200.0}, {"nice": 3.0, "amy": 1.0}]
        model.fit_transform(data)
        model_onnx = convert_sklearn(
            model,
            "dictionary vectorizer",
            [
                (
                    "input",
                    DictionaryType(StringTensorType([1]), FloatTensorType([1])),
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnDictVectorizer-OneOff-SkipDim1"
        )

    def test_model_dict_vectorizer_sort_false(self):
        model = DictVectorizer(sparse=False, sort=False)
        data = [{1: 1.0, 2: 200.0}, {1: 3.0, 3: 1.0}]
        model.fit_transform(data)
        model_onnx = convert_sklearn(
            model,
            "dictionary vectorizer",
            [
                (
                    "input",
                    DictionaryType(Int64TensorType([1]), FloatTensorType([1])),
                )
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnDictVectorizerSortFalse-OneOff-SkipDim1",
        )

    def test_model_dict_vectorizer_issue(self):
        key_value_map = [{1: "A", 2: "B"}, {1: "C", 3: "D"}, {1: "C", 3: "A"}]
        model = DictVectorizer(sparse=False).fit(key_value_map)
        with self.assertRaises(RuntimeError):
            convert_sklearn(
                model,
                "dv",
                [
                    (
                        "input",
                        DictionaryType(Int64TensorType([1]), StringTensorType([1])),
                    )
                ],
                target_opset=TARGET_OPSET,
            )

    def test_model_dict_vectorizer_pipeline_float(self):
        data = [
            {"ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
            {"PREV_ALL_LOWER": 1, "ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
            {"PREV_ALL_LOWER": 1, "ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
            {"PREV_ALL_LOWER": 1, "ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
        ]
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler())
        model.fit(data)
        expected = model.transform(data)
        model_onnx = convert_sklearn(
            model,
            "dv",
            [("input", DictionaryType(StringTensorType([1]), FloatTensorType([1])))],
            target_opset=TARGET_OPSET,
        )
        onnx.checker.check_model(model_onnx)
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        inp = {
            "ALL_LOWER": numpy.array([1], dtype=numpy.float32),
            "NEXT_ALL_LOWER": numpy.array([1], dtype=numpy.float32),
        }
        res = sess.run(None, {"input": inp})
        assert_almost_equal(expected[0].ravel(), res[0].ravel())

    def test_model_dict_vectorizer_pipeline_int(self):
        data = [
            {"ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
            {"PREV_ALL_LOWER": 1, "ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
            {"PREV_ALL_LOWER": 1, "ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
            {"PREV_ALL_LOWER": 1, "ALL_LOWER": 1, "NEXT_ALL_LOWER": 1},
        ]
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler())
        model.fit(data)
        # expected = model.transform(data)
        model_onnx = convert_sklearn(
            model,
            "dv",
            [("input", DictionaryType(StringTensorType([1]), Int64TensorType([1])))],
            target_opset=TARGET_OPSET,
        )
        onnx.checker.check_model(model_onnx)
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        inp = {
            "ALL_LOWER": numpy.array(1, dtype=numpy.int64),
            "NEXT_ALL_LOWER": numpy.array(1, dtype=numpy.int64),
        }
        try:
            got = sess.run(None, {"input": inp})
        except InvalidArgument:
            return
        self.assertTrue(got is not None)
        res = numpy.array(got[0])
        expected = model.transform(data)
        assert_almost_equal(expected[0], res)

    def test_model_dict_vectorizer_pipeline_boolean(self):
        data = [
            {"ALL_LOWER": True, "NEXT_ALL_LOWER": True},
            {"PREV_ALL_LOWER": True, "ALL_LOWER": True, "NEXT_ALL_LOWER": True},
            {"PREV_ALL_LOWER": True, "ALL_LOWER": True, "NEXT_ALL_LOWER": True},
            {"PREV_ALL_LOWER": True, "ALL_LOWER": True, "NEXT_ALL_LOWER": True},
        ]

        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler())
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "dv",
            [("input", DictionaryType(StringTensorType([1]), BooleanTensorType([1])))],
            target_opset=TARGET_OPSET,
        )
        onnx.checker.check_model(model_onnx)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(),
                providers=["CPUExecutionProvider"],
                verbose=0,
            )
        except InvalidGraph:
            return
        got = sess.run(None, {"input": data})
        self.assertTrue(got is not None)


if __name__ == "__main__":
    unittest.main()
