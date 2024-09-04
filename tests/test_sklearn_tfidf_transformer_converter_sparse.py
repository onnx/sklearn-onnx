# SPDX-License-Identifier: Apache-2.0
"""
Tests examples from scikit-learn's documentation.
"""

import packaging.version as pv
import unittest
import sys
import onnx
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import onnxruntime as ort
from skl2onnx.common.data_types import StringTensorType
from skl2onnx import convert_sklearn
from test_utils import dump_data_and_model, TARGET_OPSET

BACKEND = (
    "onnxruntime"
    if pv.Version(onnx.__version__) < pv.Version("1.16.0")
    else "onnx;onnxruntime"
)


class TestSklearnTfidfVectorizerSparse(unittest.TestCase):
    @unittest.skipIf(
        TARGET_OPSET < 9,
        # issue with encoding
        reason="https://github.com/onnx/onnx/pull/1734",
    )
    @unittest.skipIf(TARGET_OPSET < 18, reason="too long")
    @unittest.skipIf(
        pv.Version(ort.__version__) <= pv.Version("0.2.1"),
        reason="sparse not supported",
    )
    @unittest.skipIf(sys.platform != "linux", reason="too long")
    def test_model_tfidf_transform_bug(self):
        categories = [
            "alt.atheism",
            "soc.religion.christian",
            "comp.graphics",
            "sci.med",
        ]
        twenty_train = fetch_20newsgroups(
            subset="train", categories=categories, shuffle=True, random_state=0
        )
        text_clf = Pipeline(
            [("vect", CountVectorizer()), ("tfidf", TfidfTransformer())]
        )
        twenty_train.data[0] = "bruît " + twenty_train.data[0]
        text_clf.fit(twenty_train.data, twenty_train.target)
        model_onnx = convert_sklearn(
            text_clf,
            name="DocClassifierCV-Tfidf",
            initial_types=[("input", StringTensorType([5]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            twenty_train.data[5:10],
            text_clf,
            model_onnx,
            basename="SklearnPipelineTfidfTransformer",
            backend=BACKEND,
        )


if __name__ == "__main__":
    unittest.main()
