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
    def get_options(self):
        return {TfidfVectorizer: {"tokenexp": ""}}

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11Regex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_opset(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        for opset in range(8, TARGET_OPSET + 1):
            try:
                model_onnx = convert_sklearn(
                    vect,
                    "TfidfVectorizer",
                    [("input", StringTensorType([1]))],
                    options=self.get_options(),
                    target_opset=opset,
                )
            except RuntimeError as e:
                if "only works for opset" in str(e):
                    continue
                raise e
            self.assertTrue(model_onnx is not None)
            if opset >= 10:
                name = "SklearnTfidfVectorizer11Rx%d-OneOff-SklCol" % opset
                dump_data_and_model(
                    corpus,
                    vect,
                    model_onnx,
                    basename=name,
                    backend=BACKEND,
                )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_word4(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(
            ngram_range=(1, 1), norm=None, token_pattern="[a-zA-Z]{1,4}"
        )
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11Regex4-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_empty_string(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # TfidfVectorizer in onnxruntime fails with empty strings
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11EmptyStringRegex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_out_vocabulary(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        corpus = numpy.array(
            [
                "AZZ ZZ This is the first document.",
                "BZZ ZZ This document is the second document.",
                "ZZZ ZZ And this is the third one.",
                "WZZ ZZ Is this the first document?",
            ]
        ).reshape((4, 1))
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11OutVocabRegex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer22(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(2, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22Regex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12(self):
        corpus = numpy.array(
            [
                "AA AA",
                "AA AA BB",
            ]
        ).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer12SRegex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer122(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer12Regex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12_normL1(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l1")
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
            basename="SklearnTfidfVectorizer12L1Regex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12_normL2(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l2")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer12L2Regex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer13(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 3), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer13Regex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11parenthesis_class(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the (first) document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        extra = {
            TfidfVectorizer: {
                "separators": [" ", "[.]", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"],
                "tokenexp": None,
            }
        }
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=extra,
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisClassRegex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_idparenthesis_id(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the (first) document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())

        extra = {
            id(vect): {
                "sep2": [" ", ".", "?", ",", ";", ":", "!", "(", ")"],
                "regex": None,
            }
        }
        try:
            convert_sklearn(
                vect,
                "TfidfVectorizer",
                [("input", StringTensorType([1]))],
                options=extra,
                target_opset=TARGET_OPSET,
            )
        except (RuntimeError, NameError):
            pass

        extra = {
            id(vect): {
                "separators": [" ", "[.]", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"],
                "tokenexp": None,
            }
        }
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=extra,
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisIdRegex-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer_issue(self):
        corpus = numpy.array(
            [
                "the-first document.",
                "this-is the-third-one.",
                "this-the first-document?",
            ]
        ).reshape((3, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\b[a-z ]+\b")
        vect.fit(corpus.ravel())
        with self.assertRaises(RuntimeError) as e:
            convert_sklearn(
                vect,
                "TfidfVectorizer",
                [("input", StringTensorType([1]))],
                options=self.get_options(),
                target_opset=TARGET_OPSET,
            )
            self.assertIn("More one decomposition in tokens", str(e))
            self.assertIn(
                "Unable to split n-grams 'the first document' into tokens.", str(e)
            )

        corpus = numpy.array(
            [
                "first document.",
                "this-is the-third-one.",
                "the first document",
            ]
        ).reshape((3, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\b[a-z ]+\b")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizerIssue-OneOff-SklCol",
            backend=BACKEND,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
