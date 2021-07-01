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

    def get_options(self):
        return {TfidfVectorizer: {"tokenexp": ""}}

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer11(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11Regex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer11_opset(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        for opset in range(8, TARGET_OPSET + 1):
            try:
                model_onnx = convert_sklearn(
                    vect, 'TfidfVectorizer',
                    [('input', StringTensorType([1]))],
                    options=self.get_options(), target_opset=opset)
            except RuntimeError as e:
                if "only works for opset" in str(e):
                    continue
                raise e
            self.assertTrue(model_onnx is not None)
            if opset >= 10:
                name = "SklearnTfidfVectorizer11Rx%d-OneOff-SklCol" % opset
                dump_data_and_model(
                    corpus, vect, model_onnx, basename=name,
                    allow_failure="StrictVersion(onnxruntime.__version__) <= "
                                  "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer11_word4(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(
            1, 1), norm=None, token_pattern="[a-zA-Z]{1,4}")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11Regex4-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer11_empty_string(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            '',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # TfidfVectorizer in onnxruntime fails with empty strings
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11EmptyStringRegex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) "
                          "<= StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer11_out_vocabulary(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        corpus = numpy.array([
            'AZZ ZZ This is the first document.',
            'BZZ ZZ This document is the second document.',
            'ZZZ ZZ And this is the third one.',
            'WZZ ZZ Is this the first document?',
        ]).reshape((4, 1))
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11OutVocabRegex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer22(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(2, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer22Regex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer12(self):
        corpus = numpy.array([
            'AA AA',
            'AA AA BB',
        ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer12SRegex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer122(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer12Regex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer12_normL1(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm='l1')
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer12L1Regex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer12_normL2(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm='l2')
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer12L2Regex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer13(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 3), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=self.get_options(),
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer13Regex-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer11parenthesis_class(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the (first) document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        extra = {TfidfVectorizer: {'separators': [
            ' ', '[.]', '\\?', ',', ';',
            ':', '\\!', '\\(', '\\)'
        ],
            'tokenexp': None}}
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=extra,
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisClassRegex-"
                     "OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 10.")
    def test_model_tfidf_vectorizer11_idparenthesis_id(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the (first) document?',
        ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())

        extra = {id(vect): {"sep2": [' ', '.', '?', ',', ';', ':',
                                     '!', '(', ')'],
                            'regex': None}}
        try:
            convert_sklearn(vect, 'TfidfVectorizer',
                            [('input', StringTensorType([1]))],
                            options=extra,
                            target_opset=TARGET_OPSET)
        except (RuntimeError, NameError):
            pass

        extra = {id(vect): {"separators": [
            ' ', '[.]', '\\?', ',', ';', ':',
            '\\!', '\\(', '\\)'
        ],
            "tokenexp": None}}
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=extra,
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisIdRegex-"
                     "OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")


if __name__ == "__main__":
    unittest.main()
