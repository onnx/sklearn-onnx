# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's count vectorizer converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnx
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnCountVectorizerBug(unittest.TestCase):

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_count_vectorizer_custom_tokenizer(self):
        corpus = numpy.array([
            '9999',
            '999 99',
            '1234',
            '1 2 3 4',
            '1 2 3 4+',
        ]).reshape((5, 1))
        vect = CountVectorizer(ngram_range=(1, 1),
                               tokenizer=lambda s: [s])
        vect.fit(corpus.ravel())

        extra = {
            CountVectorizer: {
                "separators": ["ZZZZ"]
            }
        }

        prev = vect.tokenizer
        vect.tokenizer = None
        model_onnx = convert_sklearn(vect, 'CountVectorizer',
                                     [('input', StringTensorType([1]))],
                                     options=extra,
                                     target_opset=TARGET_OPSET)
        vect.tokenizer = prev

        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer11CustomTokenizer-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.4.0')")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Requires opset 9.")
    def test_model_count_vectorizer_wrong_ngram(self):
        corpus = numpy.array([
            'A AABBB0',
            'AAABB B1',
            'AA ABBB2',
            'AAAB BB3',
            'AAA BBB4',
        ]).reshape((5, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2),
                               token_pattern=r"(?u)\b\w\w+\b")
        vect.fit(corpus.ravel())

        model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                                     [('input', StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)

        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus, vect, model_onnx,
            basename="SklearnTfidfVectorizer12Wngram-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
                          "StrictVersion('0.3.0')")


if __name__ == "__main__":
    unittest.main()
