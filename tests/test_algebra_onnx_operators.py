import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.utils.extmath import row_norms
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxDiv
from skl2onnx.algebra.onnx_ops import OnnxReduceSumSquare, OnnxGemm
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxArgMin, OnnxSqrt
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
            op = OnnxSub(operator.inputs[0], W, output_names=operator.outputs)
            op.add_to(scope, container)

        def shape(operator):
            N = operator.inputs[0].type.shape[0]
            W = operator.raw_operator.W
            operator.outputs[0].type.shape = [N, W.shape[0]]

        model_onnx = convert_sklearn(
            tr, 'a-sub', [('input', FloatTensorType([1, 2]))],
            custom_shape_calculators={CustomOpTransformer: shape},
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
            op = OnnxDiv(OnnxSub(X, W), S, output_names=out)
            op.add_to(scope, container)

        def shape(operator):
            N = operator.inputs[0].type.shape[0]
            W = operator.raw_operator.W
            operator.outputs[0].type.shape = [N, W.shape[0]]

        model_onnx = convert_sklearn(
            tr, 'a-sub-div', [('input', FloatTensorType([1, 2]))],
            custom_shape_calculators={CustomOpTransformer: shape},
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

            rs = OnnxReduceSumSquare(X, axes=[1], keepdims=1)
            z = OnnxAdd(rs, OnnxGemm(X, C, zeros, alpha=-2., transB=1))
            y2 = OnnxAdd(C2, z)
            lo = OnnxArgMin(y2, axis=1, keepdims=0, output_names=out[:1])
            y2s = OnnxSqrt(y2, output_names=out[1:])

            lo.add_to(scope, container)
            y2s.add_to(scope, container)

        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model, 'a-kmeans', [('input', FloatTensorType([1, X.shape[1]]))],
            custom_conversion_functions={KMeans: conv})

        dump_data_and_model(X.astype(np.float32)[40:60], model, model_onnx,
                            basename="SklearnKMeansCustom-Dec4")

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    def test_cascade_add(self):

        def generate_onnx_graph(dim, nbnode, input_name='X1'):
            i1 = input_name
            for i in range(nbnode - 1):
                i2 = (np.ones((1, dim)) * nbnode * 10).astype(np.float32)
                node = OnnxAdd(i1, i2)
                i1 = node
            i2 = (np.ones((1, dim)) * nbnode * 10).astype(np.float32)
            node = OnnxAdd(i1, i2, output_names=['Y'])
            onx = node.to_onnx([(input_name, FloatTensorType((1, dim)))],
                               outputs=[('Y', FloatTensorType())])
            return onx

        exp = [np.array([[11., 11., 11., 11., 11.]]),
               np.array([[42., 42., 42., 42., 42.]]),
               np.array([[93., 93., 93., 93., 93.]])]
        for nbnode in (1, 2, 3):
            onx = generate_onnx_graph(5, nbnode)
            as_string = onx.SerializeToString()
            ort = InferenceSession(as_string)
            X = (np.ones((1, 5)) * nbnode).astype(np.float32)
            res_out = ort.run(None, {'X1': X})
            assert len(res_out) == 1
            res = res_out[0]
            assert_almost_equal(exp[nbnode - 1], res)

        dim = 10
        onx = generate_onnx_graph(dim, 300)
        as_string = onx.SerializeToString()
        ort = InferenceSession(as_string)
        X = (np.ones((1, dim)) * nbnode).astype(np.float32)
        res_out = ort.run(None, {'X1': X})
        assert len(res_out) == 1
        res = res_out[0]
        assert res.shape[1] == dim


if __name__ == "__main__":
    unittest.main()
