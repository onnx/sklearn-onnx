import unittest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from skl2onnx import convert_sklearn, to_onnx, wrap_as_onnx_mixin
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxDiv
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
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

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = operator.inputs[0].type
        return shape_calculator

    def to_onnx_operator(self, inputs=None, outputs=('Y', )):
        if inputs is None:
            raise RuntimeError("inputs should contain one name")
        i0 = self.get_inputs(inputs, 0)
        W = self.W_
        S = self.S_
        return OnnxDiv(OnnxSub(i0, W), S,
                       output_names=outputs)


class TestOnnxOperatorMixinSyntax(unittest.TestCase):

    def test_way1_convert_sklean(self):

        X = np.arange(20).reshape(10, 2)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        onx = convert_sklearn(
            tr, initial_types=[('X', FloatTensorType((None, X.shape[1])))])

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinWay1ConvertSklearn")

    def test_way2_to_onnx(self):

        X = np.arange(20).reshape(10, 2)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        onx = to_onnx(tr, X.astype(np.float32))

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinWay2ToOnnx")

    def test_way3_mixin(self):

        X = np.arange(20).reshape(10, 2)
        tr = KMeans(n_clusters=2)
        tr.fit(X)

        try:
            tr_mixin = wrap_as_onnx_mixin(tr)
        except KeyError as e:
            assert "SklearnGaussianProcessRegressor" in str(e)
            return

        try:
            onx = tr_mixin.to_onnx()
        except RuntimeError as e:
            assert "Method enumerate_initial_types" in str(e)
        onx = tr_mixin.to_onnx(X.astype(np.float32))

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinWay3OnnxMixin")

    def test_way4_mixin_fit(self):

        X = np.arange(20).reshape(10, 2)
        try:
            tr = wrap_as_onnx_mixin(KMeans(n_clusters=2))
        except KeyError as e:
            assert "SklearnGaussianProcessRegressor" in str(e)
            return
        tr.fit(X)

        onx = tr.to_onnx(X.astype(np.float32))

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinWay4OnnxMixin2")

    def test_pipe_way1_convert_sklean(self):

        X = np.arange(20).reshape(10, 2)
        tr = make_pipeline(CustomOpTransformer(), KMeans(n_clusters=2))
        tr.fit(X)

        onx = convert_sklearn(
            tr, initial_types=[('X', FloatTensorType((None, X.shape[1])))])

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinPipeWay1ConvertSklearn")

    def test_pipe_way2_to_onnx(self):

        X = np.arange(20).reshape(10, 2)
        tr = make_pipeline(CustomOpTransformer(), KMeans(n_clusters=2))
        tr.fit(X)

        onx = to_onnx(tr, X.astype(np.float32))

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinPipeWay2ToOnnx")

    def test_pipe_way3_mixin(self):

        X = np.arange(20).reshape(10, 2)
        tr = make_pipeline(CustomOpTransformer(), KMeans(n_clusters=2))
        tr.fit(X)

        try:
            tr_mixin = wrap_as_onnx_mixin(tr)
        except KeyError as e:
            assert "SklearnGaussianProcessRegressor" in str(e)
            return

        try:
            onx = tr_mixin.to_onnx()
        except RuntimeError as e:
            assert "Method enumerate_initial_types" in str(e)
        onx = tr_mixin.to_onnx(X.astype(np.float32))

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinPipeWay3OnnxMixin")

    def test_pipe_way4_mixin_fit(self):

        X = np.arange(20).reshape(10, 2)
        try:
            tr = wrap_as_onnx_mixin(make_pipeline(
                CustomOpTransformer(), KMeans(n_clusters=2)))
        except KeyError as e:
            assert "SklearnGaussianProcessRegressor" in str(e)
            return

        tr.fit(X)

        onx = tr.to_onnx(X.astype(np.float32))

        dump_data_and_model(
            X.astype(np.float32), tr, onx,
            basename="MixinPipeWay4OnnxMixin2")


if __name__ == "__main__":
    unittest.main()
