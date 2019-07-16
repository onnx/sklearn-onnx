"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy as np
from distutils.version import StrictVersion
import onnx
import onnx.checker
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx import convert_sklearn
from skl2onnx.convert import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxDiv, OnnxSub
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

    def to_onnx_operator(self, inputs=None, outputs=('Y', )):
        if inputs is None:
            raise RuntimeError("inputs should contain one name")
        i0 = self.get_inputs(inputs, 0)
        W = self.W_
        S = self.S_
        # case if there are multiple output nodes

        return OnnxDiv(OnnxSub(i0, W), S,
                       output_names=outputs)


class CustomOpTransformerShape(CustomOpTransformer):
    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = FloatTensorType(
                shape=operator.inputs[0].type.shape)
        return shape_calculator


class CustomOpScaler(StandardScaler, OnnxOperatorMixin):
    pass


class TestCustomModelAlgebra(unittest.TestCase):

    def test_base_api(self):
        model = CustomOpScaler()
        data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
        model.fit(data)
        try:
            model_onnx = convert_sklearn(model)
            assert model_onnx is not None
        except RuntimeError as e:
            assert "Method enumerate_initial_types is missing" in str(e)

    def test_custom_scaler(self):
        mat = np.array([[0., 1.], [0., 1.], [2., 2.]])
        tr = CustomOpTransformerShape()
        tr.fit(mat)
        z = tr.transform(mat)
        assert z is not None

        matf = mat.astype(np.float32)
        model_onnx = tr.to_onnx(matf)
        # Next instructions fails...
        # Field 'shape' of type is required but missing.
        # onnx.checker.check_model(model_onnx)

        dump_data_and_model(
            mat.astype(np.float32), tr, model_onnx,
            basename="CustomTransformerAlgebra")

    def test_custom_scaler_pipeline_right(self):
        pipe = make_pipeline(StandardScaler(), CustomOpTransformerShape())
        mat = np.array([[0., 1.], [0., 1.], [2., 2.]])
        pipe.fit(mat)
        z = pipe.transform(mat)
        assert z is not None

        matf = mat.astype(np.float32)
        model_onnx = to_onnx(pipe, matf)
        # Next instructions fails...
        # Field 'shape' of type is required but missing.
        # onnx.checker.check_model(model_onnx)

        # use assert_consistent_outputs
        # calls dump_data_and_model
        dump_data_and_model(
            mat.astype(np.float32), pipe, model_onnx, verbose=True,
            basename="CustomTransformerPipelineRightAlgebra")

    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.3.0"),
                     reason="not available")
    def test_custom_scaler_pipeline_left(self):
        pipe = make_pipeline(CustomOpTransformer(), StandardScaler())
        mat = np.array([[0., 1.], [0., 1.], [2., 2.]])
        pipe.fit(mat)
        z = pipe.transform(mat)

        matf = mat.astype(np.float32)

        try:
            model_onnx = to_onnx(pipe, matf)
        except RuntimeError as e:
            assert "cannot be infered" in str(e)

        pipe = make_pipeline(CustomOpTransformerShape(), StandardScaler())
        mat = np.array([[0., 1.], [0., 1.], [2., 2.]])
        pipe.fit(mat)
        z = pipe.transform(mat)
        assert z is not None

        matf = mat.astype(np.float32)

        model_onnx = to_onnx(pipe, matf)

        # Next instructions fails...
        # Field 'shape' of type is required but missing.
        # onnx.checker.check_model(model_onnx)

        dump_data_and_model(
            mat.astype(np.float32), pipe, model_onnx,
            basename="CustomTransformerPipelineLeftAlgebra")


if __name__ == "__main__":
    unittest.main()
