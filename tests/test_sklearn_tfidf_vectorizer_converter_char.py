# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's tfidf converter.
"""

import unittest
import packaging.version as pv
import numpy
import onnx
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


BACKEND = (
    "onnxruntime"
    if pv.Version(onnx.__version__) < pv.Version("1.16.0")
    else "onnx;onnxruntime"
)


class TestSklearnTfidfVectorizerRegex(unittest.TestCase):
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_short_word(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(
            ngram_range=(1, 1), norm=None, analyzer="word", token_pattern=".{1,2}"
        )
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11CharW2-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer22_short_word(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(
            ngram_range=(1, 2), norm=None, analyzer="word", token_pattern=".{1,5}"
        )
        vect.fit(corpus.ravel())
        try:
            convert_sklearn(
                vect,
                "TfidfVectorizer",
                [("input", StringTensorType([1]))],
                target_opset=TARGET_OPSET,
            )
        except RuntimeError as e:
            assert ("Unable to split n-grams 'e fir st do'") in str(e)

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_char(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None, analyzer="char")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11Char-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(True, reason="expected failure")
    def test_model_tfidf_vectorizer11_char_doublespace(self):
        corpus = numpy.array(
            [
                "This is the first  document.",
                "This document is the second document.",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None, analyzer="char")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11CharSpace-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12_char(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None, analyzer="char")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer12Char-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12_normL1_char(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l1", analyzer="char")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer12L1Char-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12_short_word_spaces(self):
        corpus = numpy.array(
            [
                "This is  the  first document.",
                "This document is the second  document.",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(
            ngram_range=(1, 2), norm=None, analyzer="word", token_pattern=".{1,3}"
        )
        vect.fit(corpus.ravel())
        try:
            model_onnx = convert_sklearn(
                vect,
                "TfidfVectorizer",
                [("input", StringTensorType([None, 1]))],
                target_opset=TARGET_OPSET,
            )
            self.assertTrue(model_onnx is not None)
        except RuntimeError as e:
            if "Unable to split n-grams 't i s t'" not in str(e):
                raise e

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_short_word_spaces(self):
        corpus = numpy.array(
            [
                "This is  the  first document.",
                "This document is the second  document.",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(
            ngram_range=(1, 1), norm=None, analyzer="word", token_pattern=".{1,3}"
        )
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11CharW2-OneOff-SklCol",
            backend=BACKEND,
        )


if __name__ == "__main__":
    unittest.main()
