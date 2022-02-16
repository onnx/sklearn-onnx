# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's CountVectorizer converter.
"""
import unittest
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnCountVectorizer(unittest.TestCase):

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer11(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 1))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "CountVectorizer",
                                     [("input", StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnCountVectorizer11-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer22(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(2, 2))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "CountVectorizer",
                                     [("input", StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnCountVectorizer22-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer12(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 2))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "CountVectorizer",
                                     [("input", StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnCountVectorizer12-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer13(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = CountVectorizer(ngram_range=(1, 3))
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "CountVectorizer",
                                     [("input", StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnCountVectorizer13-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer_binary(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]).reshape((4, 1))
        vect = CountVectorizer(binary=True)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(vect, "CountVectorizer",
                                     [("input", StringTensorType([1]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnCountVectorizerBinary-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.3.0')",
        )


if __name__ == "__main__":
    unittest.main()
