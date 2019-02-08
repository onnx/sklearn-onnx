"""
Tests pipeline within pipelines.
"""
import unittest
import numpy
import onnx
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnPipelineWithinPipeline(unittest.TestCase):

    def test_pipeline_pca_pipeline_minmax(self):        
        model = Pipeline(memory=None,
                    steps=[('PCA', PCA(copy=True, iterated_power='auto',
                            n_components=0.15842105263157896, random_state=None,
                            svd_solver='auto', tol=0.0, whiten=False)), 
                           ('Pipeline', Pipeline(memory=None,
                                steps=[('MinMax scaler', MinMaxScaler(copy=True, feature_range=(0, 3.7209871159509307)))]))])

        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(model, 'pipelinewithinpipeline', [('input', FloatTensorType([1, 2]))])
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
        model_onnx = convert_sklearn(model, 'pipelinewithinpipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelinePcaPipelineMinMaxLogReg")
        
    @unittest.skip(reason="Type 'tensor(float)' of input parameter (input) of operator (ConstantOfShape) "
                          "in node (ConstantOfShape) is invalid.")
    def test_pipeline_pca_pipeline_multinomial(self):
        
        model = Pipeline(memory=None,
                    steps=[('PCA', PCA(copy=True, iterated_power='auto',
                            n_components=2, random_state=None,
                            svd_solver='auto', tol=0.0, whiten=False)), 
                           ('Pipeline', Pipeline(memory=None,
                                steps=[('MinMax scaler', MinMaxScaler(copy=True, feature_range=(0, 3.7209871159509307))),
                                       ('MultinomialNB', MultinomialNB(alpha=0.7368421052631579,
                                                                      class_prior=None, fit_prior=True))]))])

        data = numpy.array([[0, 0, 0], [0, 0, 0.1], [1, 1, 1.1], [1, 1.1, 1]], dtype=numpy.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(model, 'pipelinewithinpipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelinePcaPipelineMinMaxNB2")
        
    @unittest.skip(reason="Type 'tensor(float)' of input parameter (input) of operator (ConstantOfShape) "
                          "in node (ConstantOfShape) is invalid.")
    def test_pipeline_pca_pipeline_multinomial_none(self):
        
        model = Pipeline(memory=None,
                    steps=[('PCA', PCA(copy=True, iterated_power='auto',
                            n_components=0.15842105263157896, random_state=None,
                            svd_solver='auto', tol=0.0, whiten=False)), 
                           ('Pipeline', Pipeline(memory=None,
                                steps=[('MinMax scaler', MinMaxScaler(copy=True, feature_range=(0, 3.7209871159509307))),
                                       ('MultinomialNB', MultinomialNB(alpha=0.7368421052631579,
                                                                      class_prior=None, fit_prior=True))]))])

        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(model, 'pipelinewithinpipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelinePcaPipelineMinMaxNBNone")
        

if __name__ == "__main__":
    unittest.main()
