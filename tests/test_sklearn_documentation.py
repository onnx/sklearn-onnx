"""
Tests examples from scikit-learn documentation.
"""
import unittest
from distutils.version import StrictVersion
import numpy as np
import onnx

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

try:
    from sklearn.compose import ColumnTransformer
except ModuleNotFoundError:
    ColumnTransformer = None

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.
    Takes a sequence of strings and produces a dict of sequences. Keys are
    `subject` and `body`.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # construct object dtype array with two columns
        # first column = 'subject' and second column = 'body'
        features = np.empty(shape=(len(posts), 2), dtype=object)
        for i, text in enumerate(posts):
            headers, _, bod = text.partition("\n\n")
            bod = strip_newsgroup_footer(bod)
            bod = strip_newsgroup_quoting(bod)
            features[i, 1] = bod

            prefix = "Subject:"
            sub = ""
            for line in headers.split("\n"):
                if line.startswith(prefix):
                    sub = line[len(prefix):]
                    break
            features[i, 0] = sub

        return features


class TestSklearnDocumentation(unittest.TestCase):
    "Test example from the documentation of scikit-learn."

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Encoding issue fixed in a later version")
    def test_pipeline_tfidf(self):
        categories = ["alt.atheism", "talk.religion.misc"]
        train = fetch_20newsgroups(random_state=1,
                                   subset="train",
                                   categories=categories)
        train_data = SubjectBodyExtractor().fit_transform(train.data)
        tfi = TfidfVectorizer(min_df=30)
        tdata = train_data[:300, :1]
        tfi.fit(tdata.ravel())
        extra = {
            TfidfVectorizer: {
                "sep": [" ", ".", "?", ",", ";", ":", "!", "(", ")"]
            }
        }
        model_onnx = convert_sklearn(
            tfi,
            "tfidf",
            initial_types=[("input", StringTensorType([1, 1]))],
            options=extra,
        )
        dump_data_and_model(
            tdata[:5],
            tfi,
            model_onnx,
            basename="SklearnDocumentationTfIdf-OneOff-SklCol",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')",
        )

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer introduced in 0.20",
    )
    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
        reason="Encoding issue fixed in a later version")
    def test_pipeline_tfidf_pipeline_minmax(self):
        categories = ["alt.atheism", "talk.religion.misc"]
        train = fetch_20newsgroups(random_state=1,
                                   subset="train",
                                   categories=categories)
        train_data = SubjectBodyExtractor().fit_transform(train.data)
        pipeline = Pipeline([(
            "union",
            ColumnTransformer(
                [
                    ("subject", TfidfVectorizer(min_df=50), 0),
                    ("body", TfidfVectorizer(min_df=40), 1),
                ],
                transformer_weights={"subject": 0.8},
            ),
        )])
        pipeline.fit(train_data[:300])
        extra = {
            TfidfVectorizer: {
                "sep": [
                    " ",
                    ".",
                    "?",
                    ",",
                    ";",
                    ":",
                    "!",
                    "(",
                    ")",
                    "\n",
                    '"',
                    "'",
                    "-",
                    "[",
                    "]",
                    "@",
                ]
            }
        }
        model_onnx = convert_sklearn(
            pipeline,
            "tfidf",
            initial_types=[("input", StringTensorType([1, 2]))],
            options=extra,
        )
        test_data = np.array([
            ["Albert Einstein", "Not relatively."],
            ["Alan turing", "Not automatically."],
        ])
        dump_data_and_model(
            test_data,
            pipeline,
            model_onnx,
            verbose=False,
            basename="SklearnDocumentationTfIdfUnion1-OneOff-Dec2",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')",
        )


if __name__ == "__main__":
    unittest.main()
