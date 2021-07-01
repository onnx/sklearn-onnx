# SPDX-License-Identifier: Apache-2.0

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
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnTfidfVectorizerRegex(unittest.TestCase):

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11_short_word(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None,
                               analyzer='word', token_pattern=".{1,2}")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11CharW2-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')",
            verbose=False)

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer22_short_word(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None,
                               analyzer='word', token_pattern=".{1,5}")
        vect.fit(corpus.ravel())
        try:
            convert_sklearn(vect, 'TfidfVectorizer',
                            [('input', StringTensorType([1]))],
                            target_opset=TARGET_OPSET)
        except RuntimeError as e:
            assert ("Unable to split n-grams ' seco nd do' "
                    "into tokens") in str(e)

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11_char(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None,
                               analyzer='char')
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11Char-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')",
            verbose=False)

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    @unittest.skipIf(True, reason="expected failure")
    def test_model_tfidf_vectorizer11_char_doublespace(self):
        corpus = numpy.array([
            'This is the first  document.',
            'This document is the second document.',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None,
                               analyzer='char')
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11CharSpace-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')",
            verbose=False)

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer12_char(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None,
                               analyzer='char')
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer12Char-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')",
            verbose=False)

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer12_normL1_char(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm='l1', analyzer='char')
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer12L1Char-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer12_short_word_spaces(self):
        corpus = numpy.array([
            'This is  the  first document.',
            'This document is the second  document.',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None,
                               analyzer='word', token_pattern=".{1,3}")
        vect.fit(corpus.ravel())
        try:
            model_onnx = convert_sklearn(
                vect, 'TfidfVectorizer',
                [('input', StringTensorType([None, 1]))],
                target_opset=TARGET_OPSET)
            self.assertTrue(model_onnx is not None)
        except RuntimeError as e:
            if "Unable to split n-grams 'he  sec'" not in str(e):
                raise e

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_tfidf_vectorizer11_short_word_spaces(self):
        corpus = numpy.array([
            'This is  the  first document.',
            'This document is the second  document.',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None,
                               analyzer='word', token_pattern=".{1,3}")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)

        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11CharW2-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')",
            verbose=False)


if __name__ == "__main__":
    unittest.main()
