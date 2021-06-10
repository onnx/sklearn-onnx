# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
"""
Tests examples from scikit-learn's documentation.
"""
from distutils.version import StrictVersion
import unittest
import onnx
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import onnxruntime as ort
from skl2onnx.common.data_types import StringTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnTfidfVectorizerSparse(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        # issue with encoding
        reason="https://github.com/onnx/onnx/pull/1734")
    @unittest.skipIf(StrictVersion(ort.__version__) <= StrictVersion("0.2.1"),
                     reason="sparse not supported")
    def test_model_tfidf_transform_bug(self):
        categories = [
            "alt.atheism",
            "soc.religion.christian",
            "comp.graphics",
            "sci.med",
        ]
        twenty_train = fetch_20newsgroups(subset="train",
                                          categories=categories,
                                          shuffle=True,
                                          random_state=0)
        text_clf = Pipeline([("vect", CountVectorizer()),
                             ("tfidf", TfidfTransformer())])
        twenty_train.data[0] = "bruît " + twenty_train.data[0]
        text_clf.fit(twenty_train.data, twenty_train.target)
        model_onnx = convert_sklearn(
            text_clf,
            name="DocClassifierCV-Tfidf",
            initial_types=[("input", StringTensorType([5]))],
            target_opset=TARGET_OPSET
        )
        dump_data_and_model(
            twenty_train.data[5:10],
            text_clf,
            model_onnx,
            basename="SklearnPipelineTfidfTransformer",
            # Operator mul is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " <= StrictVersion('1.5')",
        )


if __name__ == "__main__":
    unittest.main()
