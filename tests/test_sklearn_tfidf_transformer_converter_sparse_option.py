# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
import unittest
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import SparsePCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from onnxruntime import InferenceSession, __version__ as ort_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import TARGET_OPSET


class DensityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return numpy.asarray(X.todense())


class TestSklearnTfidfTransformerConverterSparseOption(unittest.TestCase):
    def common_test_model_tfidf_vectorizer_pipeline_cls(self, verbose=False):
        if pv.Version(ort_version) >= pv.Version("1.4.0"):
            # regression with stopwords in onnxruntime 1.4+
            stopwords = ["theh"]
        else:
            stopwords = ["the", "and", "is"]

        X_train = numpy.array(
            [
                "This is the first document",
                "This document is the second document.",
                "And this is the third one",
                "Is this the first document?",
            ]
        ).reshape((4, 1))
        y_train = numpy.array([0, 1, 0, 1])

        model_pipeline = Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        stop_words=stopwords,
                        lowercase=True,
                        use_idf=True,
                        ngram_range=(1, 3),
                        max_features=30000,
                    ),
                ),
                ("density", DensityTransformer()),
                ("feature_selector", SparsePCA(10, alpha=10)),
                (
                    "classifier",
                    SVC(
                        class_weight="balanced",
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                    ),
                ),
            ]
        )
        model_pipeline.fit(X_train.ravel(), y_train)

        step0 = model_pipeline.steps[0][-1].transform(X_train.ravel())
        assert isinstance(step0, scipy.sparse._csr.csr_matrix)

        if len(model_pipeline.steps) == 2:
            svc_coef = model_pipeline.steps[1][-1].support_vectors_
            assert isinstance(svc_coef, scipy.sparse._csr.csr_matrix)
            if verbose:
                sparsity = (svc_coef == 0).sum() / numpy.prod(svc_coef.shape)
                print(f"sparsity={sparsity}|{svc_coef.shape}")
        else:
            pca_coef = model_pipeline.steps[2][-1].components_
            print(type(pca_coef))
            # assert isinstance(pca_coef, scipy.sparse._csr.csr_matrix)
            if verbose:
                sparsity = (pca_coef == 0).sum() / numpy.prod(pca_coef.shape)
                print(f"sparsity={sparsity}|{pca_coef.shape}")

        initial_type = [("input", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(
            model_pipeline,
            "cv",
            initial_types=initial_type,
            options={SVC: {"zipmap": False}},
            target_opset=TARGET_OPSET,
        )

        exp = [
            model_pipeline.predict(X_train.ravel()),
            model_pipeline.predict_proba(X_train.ravel()),
        ]

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": X_train})
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

    def test_sparse(self):
        self.common_test_model_tfidf_vectorizer_pipeline_cls(__name__ == "__main__")


if __name__ == "__main__":
    unittest.main()
