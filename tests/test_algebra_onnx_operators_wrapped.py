# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx import update_registered_converter
from onnxruntime import __version__ as ortv
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


class DecorrelateTransformer2(TransformerMixin, BaseEstimator):

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


def decorrelate_transformer_convertor(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    subop = OnnxSubEstimator(op.pca_, X, op_version=opv)
    Y = OnnxIdentity(subop, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


def decorrelate_transformer_convertor2(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    Y = OnnxSubEstimator(op.pca_, X, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


class TestOnnxOperatorsWrapped(unittest.TestCase):

    @unittest.skipIf(StrictVersion(ortv) < StrictVersion('0.5.0'),
                     reason="onnxruntime too old")
    def test_sub(self):

        data = load_iris()
        X = data.data
        dec = DecorrelateTransformer()
        dec.fit(X)

        update_registered_converter(
            DecorrelateTransformer, "SklearnDecorrelateTransformer",
            decorrelate_transformer_shape_calculator,
            decorrelate_transformer_convertor)

        onx = to_onnx(dec, X.astype(np.float32), target_opset=TARGET_OPSET)
        self.assertIn('output: "variable"', str(onx))
        sess = InferenceSession(onx.SerializeToString())
        exp = dec.transform(X.astype(np.float32))
        got = sess.run(None, {'X': X.astype(np.float32)})[0]
        assert_almost_equal(got, exp, decimal=4)

    @unittest.skipIf(StrictVersion(ortv) < StrictVersion('0.5.0'),
                     reason="onnxruntime too old")
    def test_sub_double(self):

        data = load_iris()
        X = data.data
        dec = DecorrelateTransformer()
        dec.fit(X)

        update_registered_converter(
            DecorrelateTransformer, "SklearnDecorrelateTransformer",
            decorrelate_transformer_shape_calculator,
            decorrelate_transformer_convertor)

        onx = to_onnx(dec, X.astype(np.float64), target_opset=TARGET_OPSET)
        self.assertIn('output: "variable"', str(onx))
        sess = InferenceSession(onx.SerializeToString())
        exp = dec.transform(X.astype(np.float64))
        got = sess.run(None, {'X': X.astype(np.float64)})[0]
        assert_almost_equal(got, exp, decimal=4)

    @unittest.skipIf(StrictVersion(ortv) < StrictVersion('0.5.0'),
                     reason="onnxruntime too old")
    def test_sub_output(self):

        data = load_iris()
        X = data.data
        dec = DecorrelateTransformer2()
        dec.fit(X)

        update_registered_converter(
            DecorrelateTransformer2, "SklearnDecorrelateTransformer2",
            decorrelate_transformer_shape_calculator,
            decorrelate_transformer_convertor2)

        onx = to_onnx(dec, X.astype(np.float32), target_opset=TARGET_OPSET)
        self.assertIn('output: "variable"', str(onx))
        sess = InferenceSession(onx.SerializeToString())
        exp = dec.transform(X.astype(np.float32))
        got = sess.run(None, {'X': X.astype(np.float32)})[0]
        assert_almost_equal(got, exp, decimal=4)

    @unittest.skipIf(StrictVersion(ortv) < StrictVersion('0.5.0'),
                     reason="onnxruntime too old")
    def test_sub_output_double(self):

        data = load_iris()
        X = data.data
        dec = DecorrelateTransformer2()
        dec.fit(X)

        update_registered_converter(
            DecorrelateTransformer2, "SklearnDecorrelateTransformer2",
            decorrelate_transformer_shape_calculator,
            decorrelate_transformer_convertor2)

        onx = to_onnx(dec, X.astype(np.float64), target_opset=TARGET_OPSET)
        self.assertIn('output: "variable"', str(onx))
        sess = InferenceSession(onx.SerializeToString())
        exp = dec.transform(X.astype(np.float64))
        got = sess.run(None, {'X': X.astype(np.float64)})[0]
        assert_almost_equal(got, exp, decimal=4)


if __name__ == "__main__":
    unittest.main()
