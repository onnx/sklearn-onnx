# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's count vectorizer converter.
"""

import unittest
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
import onnx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings

try:
    from onnx.reference import ReferenceEvaluator
except ImportError:
    ReferenceEvaluator = None
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model, TARGET_OPSET
from test_utils.reference_implementation_text import Tokenizer


def _skl150() -> bool:
    import sklearn
    import packaging.version as pv

    return pv.Version(sklearn.__version__) >= pv.Version("1.5.0")


class TestSklearnCountVectorizerBug(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @ignore_warnings(category=(UserWarning,))
    def test_model_count_vectorizer_custom_tokenizer(self):
        corpus = numpy.array(
            [
                "9999",
                "999 99",
                "1234",
                "1 2 3 4",
                "1 2 3 4+",
            ]
        ).reshape((5, 1))
        vect = CountVectorizer(ngram_range=(1, 1), tokenizer=lambda s: [s])
        vect.fit(corpus.ravel())

        extra = {CountVectorizer: {"separators": ["ZZZZ"]}}

        prev = vect.tokenizer
        vect.tokenizer = None
        model_onnx = convert_sklearn(
            vect,
            "CountVectorizer",
            [("input", StringTensorType([1]))],
            options=extra,
            target_opset=TARGET_OPSET,
        )
        vect.tokenizer = prev

        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11CustomTokenizer-OneOff-SklCol",
        )

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.16.0"),
        reason="ReferenceEvaluator does not support tfidf with strings",
    )
    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_count_vectorizer_wrong_ngram(self):
        corpus = numpy.array(
            [
                "A AABBB0",
                "AAABB B1",
                "AA ABBB2",
                "AAAB BB3",
                "AAA BBB4",
            ]
        ).reshape((5, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b")
        vect.fit(corpus.ravel())

        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )

        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer12Wngram-OneOff-SklCol",
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        not _skl150(), reason="This issue is solved by using scikit-learn>=1.5.0"
    )
    def test_model_count_vectorizer_short_length(self):
        corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]

        labels = ["a", "b", "c", "d"]

        to_inference = [
            ".",
            "first",
            ". .",
            "document one",
            "first and",
            "document",
            "and",
        ]

        vectorizer = CountVectorizer(
            max_features=5,
            analyzer="word",
            ngram_range=(1, 1),
            encoding="utf8",
            strip_accents=None,
            token_pattern=(
                r"\b[a-zA-Z0-9_]+\b"
                r"|"
                r"[\~\-!.,:;@+&<>*={}\[\]№?()^|/%$#'`\"\\_]"
                r"|"
                r"\d+"
            ),
        )
        vectorizer.fit_transform(corpus)
        classifier = LogisticRegression()

        model = Pipeline(steps=[("vectorizer", vectorizer), ("classifier", classifier)])
        model.fit(corpus, labels)
        expected_probas = model.predict_proba(to_inference)
        expected_labels = model.predict(to_inference)

        onnx_model = convert_sklearn(
            model,
            initial_types=[("X", StringTensorType([None, None]))],
            options={"zipmap": False},
        )

        session = InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        ref = (
            None
            if ReferenceEvaluator is None
            else ReferenceEvaluator(onnx_model, new_ops=[Tokenizer])
        )

        for x in to_inference:
            feeds = {"X": numpy.array([[x]])}
            expected = model.predict([x])
            got = session.run(None, feeds)
            self.assertEqual(expected.tolist(), got[0].tolist())
            if ref is None:
                continue
            got = ref.run(None, feeds)
            self.assertEqual(expected.tolist(), got[0].tolist())

        feeds = {"X": numpy.array([to_inference]).T}
        got = session.run(None, feeds)
        self.assertEqual(expected_labels.tolist(), got[0].tolist())
        assert_almost_equal(expected_probas, got[1])
        if ref is not None:
            got = ref.run(None, feeds)
            self.assertEqual(expected_labels.tolist(), got[0].tolist())
            assert_almost_equal(expected_probas, got[1])


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("skl2onnx")
    logger.setLevel(logging.ERROR)
    unittest.main(verbosity=2)
