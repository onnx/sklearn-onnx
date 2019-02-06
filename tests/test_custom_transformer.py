"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy
import inspect
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.operator_converters.LinearClassifier import convert_sklearn_linear_classifier
from skl2onnx.common._registration import get_shape_calculator
from skl2onnx._parse import _get_sklearn_operator_name

from test_utils import dump_data_and_model


class PredictableTSNE(BaseEstimator, TransformerMixin):

    def __init__(self, transformer=None, estimator=None,
                 normalize=True, keep_tsne_outputs=False, **kwargs):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)
        if estimator is None:
            estimator = KNeighborsRegressor()
        if transformer is None:
            transformer = TSNE()
        self.estimator = estimator
        self.transformer = transformer
        self.keep_tsne_outputs = keep_tsne_outputs
        if not hasattr(transformer, "fit_transform"):
            raise AttributeError(
                "transformer {} does not have a 'fit_transform' method.".format(type(transformer)))
        if not hasattr(estimator, "predict"):
            raise AttributeError(
                "estimator {} does not have a 'predict' method.".format(type(estimator)))
        self.normalize = normalize
        if kwargs:
            self.set_params(**kwargs)

    def fit(self, X, y, sample_weight=None):
        params = dict(y=y, sample_weight=sample_weight)

        self.transformer_ = clone(self.transformer)

        sig = inspect.signature(self.transformer.fit_transform)
        pars = {}
        for p in ['sample_weight', 'y']:
            if p in sig.parameters and p in params:
                pars[p] = params[p]
        target = self.transformer_.fit_transform(X, **pars)

        sig = inspect.signature(self.estimator.fit)
        if 'sample_weight' in sig.parameters:
            self.estimator_ = clone(self.estimator).fit(
                X, target, sample_weight=sample_weight)
        else:
            self.estimator_ = clone(self.estimator).fit(X, target)
        mean = target.mean(axis=0)
        var = target.std(axis=0)
        self.mean_ = mean
        self.inv_std_ = 1. / var
        exp = (target - mean) * self.inv_std_
        got = (self.estimator_.predict(X) - mean) * self.inv_std_
        self.loss_ = mean_squared_error(exp, got)
        if self.keep_tsne_outputs:
            self.tsne_outputs_ = exp if self.normalize else target
        return self

    def transform(self, X):
        pred = self.estimator_.predict(X)
        if self.normalize:
            pred -= self.mean_
            pred *= self.inv_std_
        return pred

    def get_params(self, deep=True):
        res = {}
        for k, v in self.transformer.get_params().items():
            res["t_" + k] = v
        for k, v in self.estimator.get_params().items():
            res["e_" + k] = v
        return res

    def set_params(self, **values):
        pt, pe, pn = {}, {}, {}
        for k, v in values.items():
            if k.startswith('e_'):
                pe[k[2:]] = v
            elif k.startswith('t_'):
                pt[k[2:]] = v
            elif k.startswith('n_'):
                pn[k[2:]] = v
            else:
                raise ValueError("Unexpected parameter name '{0}'".format(k))
        self.transformer.set_params(**pt)
        self.estimator.set_params(**pe)


def predictable_tsne_shape_calculator(operator):    
    input = operator.inputs[0]
    output = operator.outputs[0]
    op = operator.raw_operator    
    N = input.type.shape[0]
    C = op.estimator_._y.shape[1]    
    operator.outputs[0].type = FloatTensorType([N, C])
        
        
def predictable_tsne_converter(scope, operator, container):
    input = operator.inputs[0]
    output = operator.outputs[0]
    op = operator.raw_operator    
    model = op.estimator_
    alias = _get_sklearn_operator_name(type(model))
    knn_op = scope.declare_local_operator(alias, model)
    knn_op.inputs = operator.inputs
    knn_output = scope.declare_local_variable('knn_output', FloatTensorType())
    knn_op.outputs.append(knn_output)
    shape_calc = get_shape_calculator(alias)
    shape_calc(knn_op)
    name = scope.get_unique_operator_name('Scaler')
    attrs = dict(name=name,
                 scale=op.inv_std_.ravel().astype(float),
                 offset=op.mean_.ravel().astype(float))

    container.add_node('Scaler', [knn_output.onnx_name], [output.full_name],
                       op_domain='ai.onnx.ml', **attrs)


class TestOtherLibrariesInPipeline(unittest.TestCase):

    def test_custom_pipeline_scaler(self):

        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20]
        yd = digits.target[:20]
        imgs = digits.images
        n_samples, n_features = Xd.shape

        ptsne_knn = PredictableTSNE()
        ptsne_knn.fit(Xd, yd)

        update_registered_converter(PredictableTSNE, 'CustomPredictableTSNE',                                    
                                    predictable_tsne_shape_calculator,
                                    predictable_tsne_converter)

        model_onnx = convert_sklearn(ptsne_knn, 'predictable_tsne',
                                     [('input', FloatTensorType([1, Xd.shape[1]]))])
        
        dump_data_and_model(Xd.astype(numpy.float32)[:7], ptsne_knn, model_onnx,
                            basename="CustomTransformerTSNEkNN-OneOffArray", verbose=True)
        

if __name__ == "__main__":
    unittest.main()
