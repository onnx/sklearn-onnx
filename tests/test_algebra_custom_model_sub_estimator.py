# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import logging
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
except ImportError:
    InvalidArgument = RuntimeError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from test_utils import TARGET_OPSET


class CustomOpTransformer1(BaseEstimator, TransformerMixin,
                           OnnxOperatorMixin):

    def __init__(self, op_version=None):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        OnnxOperatorMixin.__init__(self)
        self.op_version = op_version

    def fit(self, X, y=None):
        self.norm_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        return self.norm_.transform(X)

    def to_onnx_operator(self, inputs=None, outputs=('Y', ),
                         target_opset=None, **kwargs):
        if inputs is None:
            raise RuntimeError("inputs should contain one name")
        opv = target_opset or self.op_version
        i0 = self.get_inputs(inputs, 0)
        out = OnnxSubEstimator(self.norm_, i0, op_version=opv)
        return OnnxIdentity(out, op_version=self.op_version,
                            output_names=outputs)

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = FloatTensorType(
                shape=operator.inputs[0].type.shape)
        return shape_calculator


class CustomOpTransformer2(BaseEstimator, TransformerMixin,
                           OnnxOperatorMixin):

    def __init__(self, op_version=None):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        OnnxOperatorMixin.__init__(self)
        self.op_version = op_version

    def fit(self, X, y=None):
        self.norm_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        return self.norm_.transform(X)

    def to_onnx_operator(self, inputs=None, outputs=('Y', ),
                         target_opset=None, **kwargs):
        if inputs is None:
            raise RuntimeError("inputs should contain one name")
        opv = target_opset or self.op_version
        i0 = self.get_inputs(inputs, 0)
        out = OnnxSubEstimator(self.norm_, i0, op_version=opv,
                               output_names=outputs)
        return out

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = FloatTensorType(
                shape=operator.inputs[0].type.shape)
        return shape_calculator


class CustomOpTransformer3(BaseEstimator, TransformerMixin,
                           OnnxOperatorMixin):

    def __init__(self, op_version=None):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        OnnxOperatorMixin.__init__(self)
        self.op_version = op_version

    def fit(self, X, y=None):
        self.norm_ = LogisticRegression().fit(X, y)
        return self

    def transform(self, X):
        return self.norm_.predict_proba(X)

    def to_onnx_operator(self, inputs=None, outputs=('Y', ),
                         target_opset=None, **kwargs):
        if inputs is None:
            raise RuntimeError("inputs should contain one name")
        opv = target_opset or self.op_version
        i0 = self.get_inputs(inputs, 0)
        out = OnnxSubEstimator(self.norm_, i0, op_version=opv,
                               options={'zipmap': False})
        return OnnxIdentity(
            out[1], output_names=outputs, op_version=self.op_version)

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = FloatTensorType(
                shape=operator.inputs[0].type.shape)
        return shape_calculator


class CustomOpTransformer4(BaseEstimator, TransformerMixin,
                           OnnxOperatorMixin):

    def __init__(self, op_version=None):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        OnnxOperatorMixin.__init__(self)
        self.op_version = op_version

    def fit(self, X, y=None):
        self.norm_ = LogisticRegression().fit(X, y)
        return self

    def transform(self, X):
        return self.norm_.predict_proba(X)

    def to_onnx_operator(self, inputs=None, outputs=('Y', ),
                         target_opset=None, **kwargs):
        if inputs is None:
            raise RuntimeError("inputs should contain one name")
        opv = target_opset or self.op_version
        i0 = self.get_inputs(inputs, 0)
        out = OnnxSubEstimator(self.norm_, i0, op_version=opv)
        return OnnxIdentity(
            out[1], output_names=outputs, op_version=opv)

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = FloatTensorType(
                shape=operator.inputs[0].type.shape)
        return shape_calculator


class TestCustomModelAlgebraSubEstimator(unittest.TestCase):

    def setUp(self, log=False):
        self.log = logging.getLogger('skl2onnx')
        if log:
            self.log.setLevel(logging.DEBUG)
            logging.basicConfig(level=logging.DEBUG)

    def check_transform(self, obj, X):
        self.log.debug("[check_transform------] type(obj)=%r" % type(obj))
        expected = obj.transform(X)
        onx = to_onnx(obj, X, target_opset=TARGET_OPSET)
        try:
            sess = InferenceSession(onx.SerializeToString())
        except InvalidArgument as e:
            raise AssertionError(
                "Issue %r with\n%s" % (e, str(onx))) from e
        got = sess.run(None, {'X': X})[0]
        assert_almost_equal(expected, got)

    def test_custom_scaler_1(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = CustomOpTransformer1(op_version=TARGET_OPSET)
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_2(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = CustomOpTransformer2(op_version=TARGET_OPSET)
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_3(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        y = np.array([0, 0, 1], dtype=np.int64)
        tr = CustomOpTransformer3(op_version=TARGET_OPSET)
        tr.fit(X, y)
        self.check_transform(tr, X)

    def test_custom_scaler_4(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        y = np.array([0, 0, 1], dtype=np.int64)
        tr = CustomOpTransformer4(op_version=TARGET_OPSET)
        tr.fit(X, y)
        self.check_transform(tr, X)


if __name__ == "__main__":
    # cl = TestCustomModelAlgebraSubEstimator()
    # cl.setUp(log=False)
    # cl.test_custom_scaler_2()
    unittest.main()
