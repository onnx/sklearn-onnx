# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's tfidf converter.
"""

import unittest
import copy
import sys
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
import onnx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # Old scikit-learn
    ColumnTransformer = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model,
    TARGET_OPSET,
    InferenceSessionEx as InferenceSession,
)


ort_version = ".".join(ort_version.split(".")[:2])

BACKEND = (
    "onnxruntime"
    if pv.Version(onnx.__version__) < pv.Version("1.16.0")
    else "onnx;onnxruntime"
)


class TestSklearnTfidfVectorizer(unittest.TestCase):
    def get_options(self):
        return {TfidfVectorizer: {"tokenexp": None}}

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.3.0"), reason="Requires opset 9."
    )
    def test_model_tfidf_vectorizer11(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType())],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11-OneOff-SklCol",
            backend=BACKEND,
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": corpus.ravel()})[0]
        assert res.shape == (4, 9)

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.3.0"), reason="Requires opset 9."
    )
    def test_model_tfidf_vectorizer11_nolowercase(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None, lowercase=False)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType())],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11NoL-OneOff-SklCol",
            backend=BACKEND,
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": corpus.ravel()})[0]
        assert res.shape == (4, 11)

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(ColumnTransformer is None, reason="Requires newer scikit-learn")
    def test_model_tfidf_vectorizer11_compose(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        corpus = numpy.hstack([corpus, corpus])
        y = numpy.array([0, 1, 0, 1])
        model = ColumnTransformer(
            [
                ("a", TfidfVectorizer(), 0),
                ("b", TfidfVectorizer(), 1),
            ]
        )
        model.fit(corpus, y)
        model_onnx = convert_sklearn(
            model,
            "TfIdfcomp",
            [("input", StringTensorType([4, 2]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": corpus})[0]
        exp = model.transform(corpus)
        assert_almost_equal(res, exp)

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_empty_string_case1(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                " ",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus[:3].ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        # TfidfVectorizer in onnxruntime fails with empty strings,
        # which was fixed in version 0.3.0 afterward
        dump_data_and_model(
            corpus[2:],
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11EmptyStringSepCase1-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_empty_string_case2(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # onnxruntime fails with empty strings
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11EmptyString-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_out_vocabulary(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        corpus = numpy.array(
            [
                "AZZ ZZ This is the first document.",
                "BZZ ZZ This document is the second document.",
                "ZZZ ZZ And this is the third one.",
                "WZZ ZZ Is this the first document?",
            ]
        ).reshape((4, 1))
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11OutVocab-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer22(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(2, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer21(self):
        corpus = numpy.array(["AA AA", "AA AA BB"]).reshape((2, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22S-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12_normL1(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l1")
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
            basename="SklearnTfidfVectorizer22L1-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer12_normL2(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 2), norm="l2")
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer22L2-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer13(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 3), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer13-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11parenthesis_class(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the (first) document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        extra = {
            TfidfVectorizer: {
                "separators": [" ", "\\.", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"]
            }
        }
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=extra,
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # This test depends on this issue:
        # https://github.com/Microsoft/onnxruntime/issues/957.
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisClass-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer11_idparenthesis_id(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the (first) document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())

        extra = {id(vect): {"sep2": [" ", ".", "?", ",", ";", ":", "!", "(", ")"]}}
        try:
            convert_sklearn(
                vect,
                "TfidfVectorizer",
                [("input", StringTensorType([None, 1]))],
                options=extra,
                target_opset=TARGET_OPSET,
            )
        except (RuntimeError, NameError):
            pass

        extra = {
            id(vect): {
                "separators": [" ", "[.]", "\\?", ",", ";", ":", "\\!", "\\(", "\\)"]
            }
        }
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=extra,
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11ParenthesisId-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    def test_model_tfidf_vectorizer_binary(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(binary=True)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType([1]))],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizerBinary-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.3.0"), reason="Requires opset 9."
    )
    def test_model_tfidf_vectorizer11_64(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType())],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer1164-OneOff-SklCol",
            backend=BACKEND,
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": corpus.ravel()})[0]
        assert res.shape == (4, 9)

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.3.0"), reason="Requires opset 9."
    )
    def test_tfidf_svm(self):
        data = [
            ["schedule a meeting", 0],
            ["schedule a sync with the team", 0],
            ["slot in a meeting", 0],
            ["call ron", 1],
            ["make a phone call", 1],
            ["call in on the phone", 2],
        ]
        docs = [doc for (doc, _) in data]
        labels = [label for (_, label) in data]

        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(docs)
        embeddings = vectorizer.transform(docs)
        dim = embeddings.shape[1]

        clf = SVC()
        clf.fit(embeddings, labels)
        embeddings = numpy.asarray(embeddings.astype(numpy.float32).todense())
        exp = clf.predict(embeddings)

        initial_type = [("input", FloatTensorType([None, dim]))]
        model_onnx = convert_sklearn(
            clf, initial_types=initial_type, target_opset=TARGET_OPSET
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": embeddings})[0]
        assert_almost_equal(exp, res)

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.0.0"), reason="Requires opset 10."
    )
    def test_model_tfidf_vectorizer_nan(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        options = copy.deepcopy(self.get_options())
        options[TfidfVectorizer]["nan"] = True
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType())],
            options=options,
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": corpus.ravel()})[0]
        assert res.shape == (4, 9)
        assert numpy.isnan(res[0, 0])

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_model_tfidf_vectorizer11_custom_vocabulary(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vc = ["first", "second", "third", "document", "this"]
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None, vocabulary=vc)
        vect.fit(corpus.ravel())
        self.assertFalse(hasattr(vect, "stop_words_"))
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType())],
            options=self.get_options(),
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            corpus,
            vect,
            model_onnx,
            basename="SklearnTfidfVectorizer11CustomVocab-OneOff-SklCol",
            backend=BACKEND,
        )

    @unittest.skipIf(TARGET_OPSET < 10, reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.3.0"), reason="Requires opset 9."
    )
    def test_model_tfidf_vectorizer_locale(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        vect = TfidfVectorizer(ngram_range=(1, 1), norm=None)
        vect.fit(corpus.ravel())
        locale = "en_US"
        options = self.get_options()
        options[TfidfVectorizer].update({"locale": locale})
        model_onnx = convert_sklearn(
            vect,
            "TfidfVectorizer",
            [("input", StringTensorType())],
            options=options,
            target_opset=TARGET_OPSET,
        )
        self.assertIn('name: "locale"', str(model_onnx))
        self.assertIn(f's: "{locale}"', str(model_onnx))
        if sys.platform == "win32":
            # Linux fails due to misconfiguration with langage-pack-en.
            dump_data_and_model(
                corpus,
                vect,
                model_onnx,
                basename="SklearnTfidfVectorizer11Locale-OneOff-SklCol",
                backend=BACKEND,
            )

            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            res = sess.run(None, {"input": corpus.ravel()})[0]
            assert res.shape == (4, 9)


if __name__ == "__main__":
    unittest.main()
