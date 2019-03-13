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

    def get_options(self):
        return {TfidfVectorizer: {"regex": ""}}

    def test_re2(self):
        try:
            import re2
        except ImportError:
            warnings.warning("re2 cannot be tested because not installed.")
            return
        text = 'This is the first document.'
        pat = '\\b(\\w\\w+)\\b'
        reg = re2.compile(pat)
        gr = reg.search(text)
        self.assertIsTrue(gr is not None)
        self.assertEqual(gr.groups(), ('This',))

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
        dump_data_and_model(corpus, vect, model_onnx, basename="SklearnTfidfVectorizer11-OneOff-SklCol",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')",
                            verbose=True)


if __name__ == "__main__":
    unittest.main()
