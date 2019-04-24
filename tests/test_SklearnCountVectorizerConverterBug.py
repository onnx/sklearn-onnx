"""
Tests scikit-learn's tfidf converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnx
from test_utils import dump_data_and_model


class TestSklearnCountVectorizerBug(unittest.TestCase):

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_count_vectorizer_custom(self):
        corpus = numpy.array([
            '9999',
            '999 99',
            '1234',
            '1 2 3 4',
            '1 2 3 4+',
        ]).reshape((5, 1))
        vect = CountVectorizer(ngram_range=(1, 1), tokenizer = lambda s: [s])
        vect.fit(corpus.ravel())

        extra = {
            CountVectorizer: {
                "sep": ["ZZZZ"]
            }
        }

        prev = vect.tokenizer
        vect.tokenizer = None
        model_onnx = convert_sklearn(vect, 'CountVectorizer',
                                     [('input', StringTensorType([1, 1]))],
                                     options=extra)
        vect.tokenizer = prev

        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11RegexBug-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')")


if __name__ == "__main__":
    unittest.main()
