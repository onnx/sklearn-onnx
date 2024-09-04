# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's CountVectorizer converter.
"""

import unittest
import sys
import packaging.version as pv
import numpy
import onnx
from sklearn.feature_extraction.text import CountVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnCountVectorizer(unittest.TestCase):
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    def test_model_count_vectorizer11(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 1))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "CountVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx, basename="SklearnCountVectorizer11-OneOff-SklCol"
        )

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer22(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(2, 2))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "CountVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx, basename="SklearnCountVectorizer22-OneOff-SklCol"
        )

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer12(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 2))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "CountVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx, basename="SklearnCountVectorizer12-OneOff-SklCol"
        )

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer13(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 3))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "CountVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx, basename="SklearnCountVectorizer13-OneOff-SklCol"
        )

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer_binary(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = CountVectorizer(binary=True)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "CountVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnCountVectorizerBinary-OneOff-SklCol",
        )

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer11_locale(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 1))
        vect.fit(corpus.ravel())
        locale = "en_US"
        options = {CountVectorizer: {"locale": locale}}
        model_onnx = convert_sklearn(
            vect,
            "CountVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
            options=options,
        )
        self.assertIn('name: "locale"', str(model_onnx))
        self.assertIn(f's: "{locale}"', str(model_onnx))
        self.assertTrue(model_onnx is not None)
        if sys.platform == "win32":
            # Linux fails due to misconfiguration with langage-pack-en.
            dump_data_and_model(
                corpus,
                vect,
                model_onnx,
                basename="SklearnCountVectorizer11Locale-OneOff-SklCol",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
