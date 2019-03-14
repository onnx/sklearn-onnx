"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


class TestSklearnTfidfVectorizer(unittest.TestCase):

    def get_options(self):
        return {TfidfVectorizer: {"regex": None}}

    def test_model_tfidf_vectorizer11(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the first document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer11Sep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer11_empty_string(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                '',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)

        # TfidfVectorizer in onnxruntime fails with empty strings
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer11EmptyStringSep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.3.0')")

    def test_model_tfidf_vectorizer11_out_vocabulary(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the first document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        corpus = numpy.array([
                'AZZ ZZ This is the first document.',
                'BZZ ZZ This document is the second document.',
                'ZZZ ZZ And this is the third one.',
                'WZZ ZZ Is this the first document?',
                ]).reshape((4, 1))
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer11OutVocabSep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer22(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the first document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(2, 2), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer22Sep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer12(self):
        corpus = numpy.array([
                'AA AA',
                'AA AA BB',
                ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer22SSep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer12(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the first document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer22Sep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer12_normL1(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the first document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm='l1')
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer22L1Sep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer12_normL2(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the first document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm='l2')
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer22L2Sep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer13(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the first document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 3), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=self.get_options())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer13Sep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_tfidf_vectorizer11parenthesis_class(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the (first) document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        extra = {TfidfVectorizer: {'sep': [' ', '.', '?', ',', ';', ':', '!', '(', ')'],
                                   'regex': None}}
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=extra)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx,
                            basename="SklearnTfidfVectorizer11ParenthesisClassSep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.0') or "
                                          "StrictVersion(onnx.__version__) <= StrictVersion('1.3')")

    def test_model_tfidf_vectorizer11_idparenthesis_id(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
                'Is this the (first) document?',
                ]).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        
        extra = {id(vect): {"sep2": [' ', '.', '?', ',', ';', ':', '!', '(', ')'],
                            'regex': None}}
        try:
            convert_sklearn(vect, 'TfidfVectorizer',
                            [('input', StringTensorType([1, 1]))],
                            options=extra)
        except RuntimeError:
            pass
        
        extra = {id(vect): {"sep": [' ', '.', '?', ',', ';', ':', '!', '(', ')'],
                            "regex": None}}
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=extra)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx,
                            basename="SklearnTfidfVectorizer11ParenthesisIdSep-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")


if __name__ == "__main__":
    unittest.main()
