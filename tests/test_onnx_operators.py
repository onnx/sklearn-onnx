import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.utils.extmath import row_norms
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.algebra.onnx_ops import Sub, Div
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import ReduceSumSquare, Gemm, Add, ArgMin, Sqrt
from test_utils import dump_data_and_model


class TestOnnxOperators(unittest.TestCase):

    def test_sub(self):

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
            op = Sub(operator.inputs[0], W, output_names=operator.outputs)
            op.add_to(scope, container)

        def shape(operator):
            N = operator.inputs[0].type.shape[0]
            W = operator.raw_operator.W
            operator.outputs[0].type.shape = [N, W.shape[0]]

        model_onnx = convert_sklearn(tr, 'a-sub', [('input', FloatTensorType([1, 2]))],
                                     custom_shape_calculators={
                                         CustomOpTransformer: shape},
                                     custom_conversion_functions={CustomOpTransformer: conv})

        sess = InferenceSession(model_onnx.SerializeToString())
        z2 = sess.run(None, {'input': mat.astype(np.float32)})[0]
        assert_almost_equal(z, z2)

    def test_sub_div(self):

        class CustomOpTransformer(BaseEstimator, TransformerMixin):

            def __init__(self):
                pass

            def fit(self, X, y=None):
                self.W = np.mean(X, axis=0)
                self.S = np.std(X, axis=0)
                return self

            def transform(self, X):
                return (X - self.W) / self.S

        mat = np.array([[0., 1.], [0., 1.], [2., 2.]])
        tr = CustomOpTransformer()
        tr.fit(mat)
        z = tr.transform(mat)

        def conv(scope, operator, container):
            W = operator.raw_operator.W
            S = operator.raw_operator.S
            X = operator.inputs[0]
            out = operator.outputs
            op = Div(Sub(X, W), S, output_names=out)
            op.add_to(scope, container)

        def shape(operator):
            N = operator.inputs[0].type.shape[0]
            W = operator.raw_operator.W
            operator.outputs[0].type.shape = [N, W.shape[0]]

        model_onnx = convert_sklearn(tr, 'a-sub-div', [('input', FloatTensorType([1, 2]))],
                                     custom_shape_calculators={
                                         CustomOpTransformer: shape},
                                     custom_conversion_functions={CustomOpTransformer: conv})

        sess = InferenceSession(model_onnx.SerializeToString())
        z2 = sess.run(None, {'input': mat.astype(np.float32)})[0]
        assert_almost_equal(z, z2)
        
    def test_sub_kmeans(self):

        def conv(scope, operator, container):
            X = operator.inputs[0]
            out = operator.outputs
            op = operator.raw_operator

            C = op.cluster_centers_
            C2 = row_norms(C, squared=True)

            N = X.type.shape[0]
            zeros = np.zeros((N, ))

            rs = ReduceSumSquare(X, axes=[1], keepdims=1)
            z = Add(rs, Gemm(X, C, zeros, alpha=-2., transB=1))
            y2 = Add(C2, z)
            l = ArgMin(y2, axis=1, keepdims=0, output_names=out[:1])
            y2s = Sqrt(y2, output_names=out[1:])

            l.add_to(scope, container)
            y2s.add_to(scope, container)

        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)

        model_onnx = convert_sklearn(model, 'a-kmeans', [('input', FloatTensorType([1, X.shape[1]]))],
                                     custom_conversion_functions={KMeans: conv})

        dump_data_and_model(X.astype(np.float32)[40:60], model, model_onnx,
                            basename="SklearnKMeansCustom-Dec4")


if __name__ == "__main__":
    unittest.main()
