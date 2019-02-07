import unittest
import numpy
import pandas
from distutils.version import StrictVersion
import onnx
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from test_utils import dump_data_and_model



class TestSklearnPipeline(unittest.TestCase):

    def test_pipeline_pca_pipeline_0(self):
        
        model = Pipeline(memory=None,
                    steps=[('PCA', PCA(copy=True, iterated_power='auto',
                            n_components=0.15842105263157896, random_state=None,
                            svd_solver='auto', tol=0.0, whiten=False)), 
                           ('Pipeline', Pipeline(memory=None,
                                steps=[('MinMax scaler', MinMaxScaler(copy=True, feature_range=(0, 3.7209871159509307)))]))])

        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(model, 'pipelinepipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelinePcaPipelineMinMax")
        
    def test_pipeline_pca_pipeline_none_lin(self):
        
        model = Pipeline(memory=None,
                    steps=[('PCA', PCA(copy=True, iterated_power='auto',
                            n_components=0.15842105263157896, random_state=None,
                            svd_solver='auto', tol=0.0, whiten=False)), 
                           ('Pipeline', Pipeline(memory=None,
                                steps=[('MinMax scaler', MinMaxScaler(copy=True, feature_range=(0, 3.7209871159509307))),
                                       ('logreg', LogisticRegression())]))])

        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(model, 'pipelinepipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelinePcaPipelineMinMaxLogReg")
        
    def test_pipeline_pca_pipeline(self):
        
        model = Pipeline(memory=None,
                    steps=[('PCA', PCA(copy=True, iterated_power='auto',
                            n_components=2, random_state=None,
                            svd_solver='auto', tol=0.0, whiten=False)), 
                           ('Pipeline', Pipeline(memory=None,
                                steps=[('MinMax scaler', MinMaxScaler(copy=True, feature_range=(0, 3.7209871159509307))),
                                       ('MultinomialNB', MultinomialNB(alpha=0.7368421052631579, class_prior=None, fit_prior=True))]))])

        data = numpy.array([[0, 0, 0], [0, 0, 0.1], [1, 1, 1.1], [1, 1.1, 1]], dtype=numpy.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(model, 'pipelinepipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelinePcaPipelineMinMaxNB2")
        
    def test_pipeline_pca_pipeline_none(self):
        
        model = Pipeline(memory=None,
                    steps=[('PCA', PCA(copy=True, iterated_power='auto',
                            n_components=0.15842105263157896, random_state=None,
                            svd_solver='auto', tol=0.0, whiten=False)), 
                           ('Pipeline', Pipeline(memory=None,
                                steps=[('MinMax scaler', MinMaxScaler(copy=True, feature_range=(0, 3.7209871159509307))),
                                       ('MultinomialNB', MultinomialNB(alpha=0.7368421052631579, class_prior=None, fit_prior=True))]))])

        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(model, 'pipelinepipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelinePcaPipelineMinMaxNB")
        

if __name__ == "__main__":
    unittest.main()
