import unittest
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
import onnx
from sklearn import __version__ as skl_version, __file__ as skl_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from skl2onnx import to_onnx
from skl2onnx.sklapi import TraceableTfidfVectorizer, TraceableCountVectorizer
from skl2onnx.sklapi.sklearn_text_onnx import register
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model, TARGET_OPSET

BACKEND = (
    "onnxruntime"
    if pv.Version(onnx.__version__) < pv.Version("1.16.0")
    else "onnx;onnxruntime"
)


class TestSklearnText(unittest.TestCase):
    def test_count_vectorizer(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
            mod1 = CountVectorizer(ngram_range=ng)
            mod1.fit(corpus)

            mod2 = TraceableCountVectorizer(ngram_range=ng)
            mod2.fit(corpus)

            pred1 = mod1.transform(corpus)
            pred2 = mod2.transform(corpus)
            assert_almost_equal(pred1.todense(), pred2.todense())

            voc = mod2.vocabulary_
            for k in voc:
                self.assertIsInstance(k, tuple)

    def test_count_vectorizer_regex(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for pattern in ["[a-zA-Z ]{1,4}", "[a-zA-Z]{1,4}"]:
            for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
                mod1 = CountVectorizer(ngram_range=ng, token_pattern=pattern)
                mod1.fit(corpus)

                mod2 = TraceableCountVectorizer(ngram_range=ng, token_pattern=pattern)
                mod2.fit(corpus)

                pred1 = mod1.transform(corpus)
                pred2 = mod2.transform(corpus)
                assert_almost_equal(pred1.todense(), pred2.todense())

                voc = mod2.vocabulary_
                for k in voc:
                    self.assertIsInstance(k, tuple)
                if " ]" in pattern:
                    spaces = 0
                    for k in voc:
                        self.assertIsInstance(k, tuple)
                        for i in k:
                            if " " in i:
                                spaces += 1
                    self.assertGreater(spaces, 1)

    def test_tfidf_vectorizer(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
            mod1 = TfidfVectorizer(ngram_range=ng)
            mod1.fit(corpus)

            mod2 = TraceableTfidfVectorizer(ngram_range=ng)
            mod2.fit(corpus)

            pred1 = mod1.transform(corpus)
            pred2 = mod2.transform(corpus)
            assert_almost_equal(pred1.todense(), pred2.todense())

            voc = mod2.vocabulary_
            for k in voc:
                self.assertIsInstance(k, tuple)

    def test_tfidf_vectorizer_english(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
            with self.subTest(ngram_range=ng):
                mod1 = TfidfVectorizer(ngram_range=ng, stop_words="english")
                mod1.fit(corpus)

                mod2 = TraceableTfidfVectorizer(ngram_range=ng, stop_words="english")
                mod2.fit(corpus)
                if len(mod1.vocabulary_) != len(mod2.vocabulary_):
                    raise AssertionError(
                        f"mod1={mod1.vocabulary_}, mod2={mod2.vocabulary_}"
                    )

                pred1 = mod1.transform(corpus)
                pred2 = mod2.transform(corpus)
                assert_almost_equal(pred1.todense(), pred2.todense())

                voc = mod2.vocabulary_
                for k in voc:
                    self.assertIsInstance(k, tuple)

    def test_count_vectorizer_english2(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for ng in [(1, 1), (1, 2), (1, 3)]:
            with self.subTest(ngram_range=ng):
                mod1 = CountVectorizer(
                    ngram_range=ng,
                    stop_words="english",
                    token_pattern="[\\w_]{2,}",
                    lowercase=True,
                    min_df=2,
                    max_features=100000,
                )
                mod1.fit(corpus)

                mod2 = TraceableCountVectorizer(
                    ngram_range=ng,
                    stop_words="english",
                    token_pattern="[\\w_]{2,}",
                    lowercase=True,
                    min_df=2,
                    max_features=100000,
                )
                mod2.fit(corpus)
                if mod1.token_pattern != mod2.token_pattern:
                    raise AssertionError(
                        f"{mod1.token_pattern!r} != {mod2.token_pattern!r}"
                    )

                if hasattr(mod1, "stop_words_"):
                    if len(mod1.stop_words_) != len(mod2.stop_words_):
                        raise AssertionError(
                            f"{mod1.stop_words_} != {mod2.stop_words_}"
                        )
                if len(mod1.vocabulary_) != len(mod2.vocabulary_):
                    raise AssertionError(
                        f"skl_version={skl_version!r}, "
                        f"skl_file={skl_file!r},\n"
                        f"mod1={mod1.vocabulary_}, mod2={mod2.vocabulary_}"
                    )

                pred1 = mod1.transform(corpus)
                pred2 = mod2.transform(corpus)
                assert_almost_equal(pred1.todense(), pred2.todense())

                voc = mod2.vocabulary_
                for k in voc:
                    self.assertIsInstance(k, tuple)

    def test_tfidf_vectorizer_english2(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for ng in [(1, 1), (1, 2), (1, 3)]:
            with self.subTest(ngram_range=ng):
                mod1 = TfidfVectorizer(
                    ngram_range=ng,
                    stop_words="english",
                    token_pattern="[\\w_]{2,}",
                    lowercase=True,
                    min_df=2,
                    max_features=100000,
                )
                mod1.fit(corpus)

                mod2 = TraceableTfidfVectorizer(
                    ngram_range=ng,
                    stop_words="english",
                    token_pattern="[\\w_]{2,}",
                    lowercase=True,
                    min_df=2,
                    max_features=100000,
                )
                mod2.fit(corpus)
                if mod1.token_pattern != mod2.token_pattern:
                    raise AssertionError(
                        f"{mod1.token_pattern!r} != {mod2.token_pattern!r}"
                    )
                if hasattr(mod1, "stop_words_"):
                    if len(mod1.stop_words_) != len(mod2.stop_words_):
                        raise AssertionError(
                            f"{mod1.stop_words_} != {mod2.stop_words_}"
                        )
                if len(mod1.vocabulary_) != len(mod2.vocabulary_):
                    raise AssertionError(
                        f"skl_version={skl_version!r}, "
                        f"skl_file={skl_file!r},\n"
                        f"mod1={mod1.vocabulary_}, mod2={mod2.vocabulary_}"
                    )

                pred1 = mod1.transform(corpus)
                pred2 = mod2.transform(corpus)
                assert_almost_equal(pred1.todense(), pred2.todense())

                voc = mod2.vocabulary_
                for k in voc:
                    self.assertIsInstance(k, tuple)

    def test_tfidf_vectorizer_regex(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for pattern in ["[a-zA-Z ]{1,4}", "[a-zA-Z]{1,4}"]:
            for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
                mod1 = TfidfVectorizer(ngram_range=ng, token_pattern=pattern)
                mod1.fit(corpus)

                mod2 = TraceableTfidfVectorizer(ngram_range=ng, token_pattern=pattern)
                mod2.fit(corpus)

                pred1 = mod1.transform(corpus)
                pred2 = mod2.transform(corpus)

                if " ]" in pattern:
                    voc = mod2.vocabulary_
                    spaces = 0
                    for k in voc:
                        self.assertIsInstance(k, tuple)
                        for i in k:
                            if " " in i:
                                spaces += 1
                    self.assertGreater(spaces, 1)
                assert_almost_equal(pred1.todense(), pred2.todense())

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer_issue(self):
        register()
        corpus = numpy.array(
            [
                "the-first document.",
                "this-is the-third-one.",
                "this-the first-document?",
            ]
        ).reshape((3, 1))
        vect = TraceableTfidfVectorizer(
            ngram_range=(1, 2), token_pattern=r"\b[a-z ]+\b"
        )
        vect.fit(corpus.ravel())
        model_onnx = to_onnx(
            vect,
            "TfidfVectorizer",
            initial_types=[("input", StringTensorType([1]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizerIssue-OneOff-SklCol",
            backend=BACKEND,
        )


if __name__ == "__main__":
    # TestSklearnText().test_model_tfidf_vectorizer_issue()
    unittest.main()
