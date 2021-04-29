import pickle
from io import BytesIO
import unittest
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from skl2onnx import update_registered_converter, to_onnx
from skl2onnx.algebra.onnx_operator import OnnxSubOperator
from test_utils import TARGET_OPSET


class DecorrelateTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, alpha=0.):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.alpha = alpha

    def fit(self, X, y=None, sample_weights=None):
        self.pca_ = PCA(X.shape[1])
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def decorrelate_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.pca_.components_.shape[1]])
    operator.outputs[0].type = output_type


def decorrelate_transformer_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    # We tell in ONNX language how to compute the unique output.
    # op_version=opv tells which opset is requested
    subop = OnnxSubOperator(op.pca_, X, op_version=opv)
    Y = OnnxIdentity(subop, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


class TestOnnxDeprecation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        update_registered_converter(
            DecorrelateTransformer, "SklearnDecorrelateTransformer",
            decorrelate_transformer_shape_calculator,
            decorrelate_transformer_converter)

    def test_decorrelate_transformer(self):
        data = load_iris()
        X = data.data

        dec = DecorrelateTransformer()
        dec.fit(X)
        pred = dec.transform(X)
        cov = pred.T @ pred
        for i in range(cov.shape[0]):
            cov[i, i] = 1.
        assert_almost_equal(np.identity(4), cov)

        st = BytesIO()
        pickle.dump(dec, st)
        dec2 = pickle.load(BytesIO(st.getvalue()))
        assert_almost_equal(dec.transform(X), dec2.transform(X))

    def test_sub_operator(self):

        data = load_iris()
        X = data.data

        dec = DecorrelateTransformer()
        dec.fit(X)

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            onx = to_onnx(dec, X.astype(np.float32),
                          target_opset=TARGET_OPSET)
        mes = None
        for w in ws:
            if (w.category == DeprecationWarning and
                    'numpy' not in str(w.message).lower()):
                mes = w.message
        self.assertTrue(mes is not None)
        self.assertIn('will be removed', str(mes))

        sess = InferenceSession(onx.SerializeToString())

        exp = dec.transform(X.astype(np.float32))
        got = sess.run(None, {'X': X.astype(np.float32)})[0]

        def diff(p1, p2):
            p1 = p1.ravel()
            p2 = p2.ravel()
            d = np.abs(p2 - p1)
            return d.max(), (d / np.abs(p1)).max()

        res = diff(exp, got)
        self.assertLess(res[0], 1e-5)


if __name__ == "__main__":
    unittest.main()
