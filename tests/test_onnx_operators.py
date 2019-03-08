import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.base import BaseEstimator, TransformerMixin
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.algebra import OP, Sub
from skl2onnx.common.data_types import FloatTensorType



class TestOnnxOperators(unittest.TestCase):
    
    def test_gemm(self):
        
        class CustomOpTransformer(BaseEstimator, TransformerMixin):

            def __init__(self):
                pass
                
            def fit(self, X, y=None):
                self.W = np.mean(X, axis=0)
                return self
                
            def transform(self, X):
                return X - self.W

        mat = np.array([[0., 1.], [1., 2.], [3., 4.]])
        tr = CustomOpTransformer()
        tr.fit(mat)
        z = tr.transform(mat)
        
        def conv(scope, operator, container):
            W = operator.raw_operator.W
            op = Sub(operator.inputs[0], W, outputs=operator.outputs)
            op.add(scope, operator, container)
            
        def shape(operator):
            N = operator.inputs[0].type.shape[0]
            W = operator.raw_operator.W
            operator.outputs[0].type.shape = [N, W.shape[0]]
        
        model_onnx = convert_sklearn(tr, 'a-sub', [('input', FloatTensorType([1, 2]))],
                                     custom_shape_calculators={CustomOpTransformer: shape},
                                     custom_conversion_functions={CustomOpTransformer: conv})
        
        sess = InferenceSession(model_onnx.SerializeToString())
        z2 = sess.run(None, {'input': mat.astype(np.float32)})[0]
        assert_almost_equal(z, z2)
        

if __name__ == "__main__":
    unittest.main()
