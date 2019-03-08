"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


class TestSklearnCountVectorizer(unittest.TestCase):

    def test_model_count_vectorizer11(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 1))
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        assert pred is not None
        model_onnx = to_onnx(vect, 'CountVectorizer',
                             [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnCountVectorizer11-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_count_vectorizer22(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(2, 2))
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        assert pred is not None
        model_onnx = to_onnx(vect, 'CountVectorizer',
                             [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnCountVectorizer22-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_count_vectorizer12(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 2))
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        assert pred is not None
        model_onnx = to_onnx(vect, 'CountVectorizer',
                             [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnCountVectorizer12-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_model_count_vectorizer13(self):
        corpus = numpy.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 3))
        vect.fit(corpus.ravel())
        pred = vect.transform(corpus.ravel())
        assert pred is not None
        model_onnx = to_onnx(vect, 'CountVectorizer',
                             [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnCountVectorizer13-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")


if __name__ == "__main__":
    unittest.main()
