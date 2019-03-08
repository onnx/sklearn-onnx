# coding: utf-8
"""
Tests examples from scikit-learn's documentation.
"""
from distutils.version import StrictVersion
import unittest
import onnx
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from skl2onnx.common.data_types import StringTensorType
from skl2onnx import to_onnx
from test_utils import dump_data_and_model


class TestSklearnTfidfVectorizerSparse(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.4.1'),
                     # issue with encoding
                     reason="https://github.com/onnx/onnx/pull/1734")
    def test_model_tfidf_transform_bug(self):
        categories = ['alt.atheism', 'soc.religion.christian',
                      'comp.graphics', 'sci.med']
        twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                          shuffle=True, random_state=0)
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer())])
        twenty_train.data[0] = "bruît " + twenty_train.data[0]
        text_clf.fit(twenty_train.data, twenty_train.target)
        model_onnx = to_onnx(text_clf, name='DocClassifierCV-Tfidf',
                             initial_types=[('input', StringTensorType())])
        dump_data_and_model(twenty_train.data[:10], text_clf, model_onnx,
                            basename="SklearnPipelineTfidfTransformer",
                            # Operator mul is not implemented in onnxruntime
                            allow_failure="StrictVersion(onnx.__version__) <= StrictVersion('1.4.1')")


if __name__ == "__main__":
    unittest.main()
