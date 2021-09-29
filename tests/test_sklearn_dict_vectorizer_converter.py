# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's DictVectorizer converter."""

import unittest
from sklearn.feature_extraction import DictVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DictionaryType,
    StringTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnDictVectorizerConverter(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_dict_vectorizer(self):
        model = DictVectorizer()
        data = [{"amy": 1.0, "chin": 200.0}, {"nice": 3.0, "amy": 1.0}]
        model.fit_transform(data)
        model_onnx = convert_sklearn(
            model, "dictionary vectorizer",
            [(
                "input",
                DictionaryType(StringTensorType([1]), FloatTensorType([1])),
            )], target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnDictVectorizer-OneOff-SkipDim1")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_dict_vectorizer_sort_false(self):
        model = DictVectorizer(sparse=False, sort=False)
        data = [{1: 1.0, 2: 200.0}, {1: 3.0, 3: 1.0}]
        model.fit_transform(data)
        model_onnx = convert_sklearn(
            model,
            "dictionary vectorizer",
            [(
                "input",
                DictionaryType(Int64TensorType([1]), FloatTensorType([1])),
            )], target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnDictVectorizerSortFalse-OneOff-SkipDim1")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_dict_vectorizer_issue(self):
        key_value_map = [{1: 'A', 2: 'B'}, {1: 'C', 3: 'D'},
                         {1: 'C', 3: 'A'}]
        model = DictVectorizer(sparse=False).fit(key_value_map)
        with self.assertRaises(RuntimeError):
            convert_sklearn(
                model, 'dv',
                [("input", DictionaryType(Int64TensorType([1]),
                  StringTensorType([1])))],
                target_opset=TARGET_OPSET)


if __name__ == "__main__":
    unittest.main()
