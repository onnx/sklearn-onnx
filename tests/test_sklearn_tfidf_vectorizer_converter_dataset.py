# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's tfidf converter using downloaded data.
"""
import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnTfidfVectorizerDataSet(unittest.TestCase):

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_tfidf_20newsgroups(self):
        data = fetch_20newsgroups()
        X, y = np.array(data.data)[:100], np.array(data.target)[:100]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        model = TfidfVectorizer().fit(X_train)
        onnx_model = convert_sklearn(
            model, 'cv', [('input', StringTensorType(X_test.shape))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X_test, model, onnx_model,
            basename="SklearnTfidfVectorizer20newsgroups")

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_tfidf_20newsgroups_nolowercase(self):
        data = fetch_20newsgroups()
        X, y = np.array(data.data)[:100], np.array(data.target)[:100]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        model = TfidfVectorizer(lowercase=False).fit(X_train)
        onnx_model = convert_sklearn(
            model, 'cv', [('input', StringTensorType(X_test.shape))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X_test, model, onnx_model,
            basename="SklearnTfidfVectorizer20newsgroupsNOLower")


if __name__ == "__main__":
    unittest.main()
