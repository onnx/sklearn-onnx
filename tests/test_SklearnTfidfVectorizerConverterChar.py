"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import warnings
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


class TestSklearnTfidfVectorizerRegex(unittest.TestCase):

    def test_model_tfidf_vectorizer11_short_word(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None,
                               analyzer='word', token_pattern=".{1,2}")
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer11CharW2-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.3.0')",
                            verbose=True)

    def test_model_tfidf_vectorizer11_char(self):
        corpus = numpy.array([
                'This is the first document.',
                'This document is the second document.',
                ]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None,
                               analyzer='char')
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer11Char-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.3.0')",
                            verbose=True)


if __name__ == "__main__":
    unittest.main()
