"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


class TestSklearnTfidfVectorizerSparse(unittest.TestCase):

    def test_model_tfidf_transform_bug(self):        
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                          shuffle=True, random_state=42)
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])

        text_clf.fit(twenty_train.data, twenty_train.target)
        model_onnx = convert_sklearn(text_clf, name='DocClassifierCV-Tfidf',
                               initial_types=[('input', StringTensorType())])
        dump_data_and_model(twenty_train.data[:10], text_clf, model_onnx,
                            basename="SklearnPipelineTfidfTransformer",
                            # Operator mul is not implemented in onnxruntime
                            allow_failure="StrictVersion(onnx.__version__) <= StrictVersion('1.4.1')")

if __name__ == "__main__":
    unittest.main()
