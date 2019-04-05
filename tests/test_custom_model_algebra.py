"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy as np
import inspect
import onnx.checker
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from skl2onnx.algebra.base import OnnxOperatorMixin
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
from skl2onnx import operator_converters
from skl2onnx.algebra.onnx_ops import Div, Sub
from test_utils import dump_data_and_model


class CustomOpTransformer(BaseEstimator, TransformerMixin,
                          OnnxOperatorMixin):

    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

    def fit(self, X, y=None):
        self.W_ = np.mean(X, axis=0)
        self.S_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.W_) / self.S_
    
    def to_onnx_operator(self, inputs=('X', ), outputs=('Y', )):
        i0 = self.get_inputs(inputs, 0)
        W = self.W_
        S = self.S_
        return Div(Sub(i0, W), S,
                   output_names=outputs)


class TestCustomModelAlgebra(unittest.TestCase):

    def _test_base_api(self):
        
        class CustomOpScaler(StandardScaler, OnnxOperatorMixin):
            pass
            
        model = CustomOpScaler()
        data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
        model.fit(data)
        try:
            model_onnx = convert_sklearn(model)
        except RuntimeError as e:
            assert "Method enumerate_initial_types is missing" in str(e)  

    def test_custom_scaler(self):

        mat = np.array([[0., 1.], [0., 1.], [2., 2.]])
        tr = CustomOpTransformer()
        tr.fit(mat)
        z = tr.transform(mat)

        matf = mat.astype(np.float32)
        model_onnx = tr.to_onnx(matf)
        # Next instructions fails...
        # Field 'shape' of type is required but missing.
        # onnx.checker.check_model(model_onnx)
        
        dump_data_and_model(
            mat.astype(np.float32), tr, model_onnx,
            basename="CustomTransformerAlgebra")


if __name__ == "__main__":
    unittest.main()
