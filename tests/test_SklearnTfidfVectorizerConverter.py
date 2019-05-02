"""
Tests scikit-learn's tfidf converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnx
from test_utils import dump_data_and_model


class TestSklearnTfidfVectorizer(unittest.TestCase):

    def get_options(self):
        return {TfidfVectorizer: {"regex": None}}

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11_empty_string_case1(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                ' ',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus[:3].ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)

        # TfidfVectorizer in onnxruntime fails with empty strings,
        # which was fixed in version 0.3.0 afterward
        dump_data_and_model(
            corpus[2:], vect, model_onnx,
            basename="SklearnTfidfVectorizer11EmptyStringSepCase1-"
                     "OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11_empty_string_case2(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        # onnxruntime fails with empty strings
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11EmptyString-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11_out_vocabulary(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        corpus = numpy.array([
            "AZZ ZZ This is the first document.",
            "BZZ ZZ This document is the second document.",
            "ZZZ ZZ And this is the third one.",
            "WZZ ZZ Is this the first document?",
        ]).reshape((4, 1))
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11OutVocab-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer22(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(2, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer21(self):
        corpus = numpy.array(["AA AA", "AA AA BB"]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22S-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer12(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer12_normL1(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l1")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22L1-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer12_normL2(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l2")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22L2-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer13(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 3), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "TfidfVectorizer",
                                     [("input", StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer13-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11parenthesis_class(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the (first) document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        extra = {
            TfidfVectorizer: {
                "sep": [" ", ".", "?", ",", ";", ":", "!", "(", ")"]
            }
        }
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1, 1]))],
            options=extra,
        )
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisClass-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11_idparenthesis_id(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the (first) document?",
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())

        extra = {
            id(vect): {
                "sep2": [" ", ".", "?", ",", ";", ":", "!", "(", ")"]
            }
        }
        try:
            convert_sklearn(
                vect,
                "TfidfVectorizer",
                [("input", StringTensorType([1, 1]))],
                options=extra,
            )
        except RuntimeError:
            pass

        extra = {
            id(vect): {
                "sep": [" ", ".", "?", ",", ";", ":", "!", "(", ")"]
            }
        }
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1, 1]))],
            options=extra,
        )
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisId-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')",
        )


if __name__ == "__main__":
    unittest.main()
