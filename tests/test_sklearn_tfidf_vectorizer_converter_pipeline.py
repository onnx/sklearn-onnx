# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's tfidf converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnx
from test_utils import TARGET_OPSET


class TestSklearnTfidfVectorizerPipeline(unittest.TestCase):

    def common_test_model_tfidf_vectorizer_pipeline_cls(
            self, kind=None, verbose=False):
        if kind == 'stop':
            if StrictVersion(ort_version) >= StrictVersion('1.4.0'):
                # regression with stopwords in onnxruntime 1.4+
                stopwords = ['theh']
            else:
                stopwords = ['the', 'and', 'is']
        else:
            stopwords = None
        X_train = numpy.array([
            "This is the first document",
            "This document is the second document.",
            "And this is the third one",
            "Is this the first document?",
        ]).reshape((4, 1))
        y_train = numpy.array([0, 1, 0, 1])

        if kind is None:
            model_pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    stop_words=stopwords, lowercase=True, use_idf=True,
                    ngram_range=(1, 3), max_features=30000)),
            ])
        elif kind == 'cls':
            model_pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    stop_words=stopwords, lowercase=True, use_idf=True,
                    ngram_range=(1, 3), max_features=30000)),
                ('feature_selector', SelectKBest(k=10)),
                ('classifier', SVC(
                    class_weight='balanced', kernel='rbf', gamma='scale',
                    probability=True))
            ])
        elif kind == 'stop':
            model_pipeline = Pipeline([
                ('vectorizer', CountVectorizer(
                    stop_words=stopwords, lowercase=True,
                    ngram_range=(1, 2), max_features=30000)),
            ])
        elif kind == 'reg':
            model_pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    stop_words=stopwords, lowercase=True, use_idf=True,
                    ngram_range=(1, 3), max_features=30000)),
                ('feature_selector', SelectKBest(k=10)),
                ('classifier', SVR(kernel='rbf', gamma='scale'))
            ])
        else:
            raise AssertionError(kind)

        model_pipeline.fit(X_train.ravel(), y_train)
        initial_type = [('input', StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(
            model_pipeline, "cv", initial_types=initial_type,
            options={SVC: {'zipmap': False}},
            target_opset=TARGET_OPSET)

        if kind in (None, 'stop'):
            exp = [model_pipeline.transform(X_train.ravel()).toarray()]
        elif kind == 'cls':
            exp = [model_pipeline.predict(X_train.ravel()),
                   model_pipeline.predict_proba(X_train.ravel())]
        elif kind == 'reg':
            exp = [model_pipeline.predict(X_train.ravel()).reshape((-1, 1))]

        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'input': X_train})
        if verbose:
            voc = model_pipeline.steps[0][-1].vocabulary_
            voc = list(sorted([(v, k) for k, v in voc.items()]))
            for kv in voc:
                print(kv)
        for a, b in zip(exp, got):
            if verbose:
                print(stopwords)
                print(a)
                print(b)
            assert_almost_equal(a, b)

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.5.0"),
        reason="Requires opset 10.")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.0.0"),
        reason="Too old")
    def test_model_tfidf_vectorizer_pipeline(self):
        for kind in [None, 'cls', 'reg']:
            with self.subTest(kind=kind):
                self.common_test_model_tfidf_vectorizer_pipeline_cls(kind)

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.5.0"),
        reason="Requires opset 10.")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.4.0"),
        reason="Wrong handling of stopwods and n-grams")
    def test_model_tfidf_vectorizer_pipeline_stop_words(self):
        for kind in ['stop']:
            with self.subTest(kind=kind):
                self.common_test_model_tfidf_vectorizer_pipeline_cls(
                    kind, verbose=False)


if __name__ == "__main__":
    unittest.main()
