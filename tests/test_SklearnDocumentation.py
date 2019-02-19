"""
Tests pipeline within pipelines.
"""
import unittest
import onnx
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

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
            headers, _, bod = text.partition('\n\n')
            bod = strip_newsgroup_footer(bod)
            bod = strip_newsgroup_quoting(bod)
            features[i, 1] = bod

            prefix = 'Subject:'
            sub = ''
            for line in headers.split('\n'):
                if line.startswith(prefix):
                    sub = line[len(prefix):]
                    break
            features[i, 0] = sub

        return features
        

class TestSklearnDocumentation(unittest.TestCase):
    "Test example from the documentation of scikit-learn."

    def test_pipeline_pca_pipeline_minmax(self):        
        categories = ['alt.atheism', 'talk.religion.misc']
        train = fetch_20newsgroups(random_state=1,
                                   subset='train',
                                   categories=categories,
                                   )
        
        train_data = SubjectBodyExtractor().fit_transform(train.data)
        
        pipeline = Pipeline([
            ('union', ColumnTransformer(
                [
                    ('subject', TfidfVectorizer(min_df=50), 0),

                    ('body_bow', Pipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('best', TruncatedSVD(n_components=50)),
                    ]), 1),
                ],

                transformer_weights={
                    'subject': 0.8,
                    'body_bow': 0.5,
                }
            )),

            ('svc', LinearSVC()),
        ])
        
        pipeline.fit(train_data, train.target)
        
        model_onnx = convert_sklearn(pipeline, "tfidf",
                                     initial_types=[("input", StringTensorType([1, 2]))])        
        dump_data_and_model(data, pipeline, model_onnx,
                            basename="SklearnDocumentationTfIdf-OneOff")


if __name__ == "__main__":
    unittest.main()
