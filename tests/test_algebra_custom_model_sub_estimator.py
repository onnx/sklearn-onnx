# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import logging
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
except ImportError:
    InvalidArgument = RuntimeError
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes)
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx.algebra.onnx_ops import (
    OnnxArgMax,
    OnnxConcat,
    OnnxIdentity,
    OnnxReshape,
    OnnxSoftmax)
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


class CustomOpTransformer1w(BaseEstimator, TransformerMixin,
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

    def to_onnx_operator(self, inputs=None, outputs=('Y', )):
        if inputs is None:
            raise RuntimeError("inputs should contain one name")
        opv = self.op_version
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


class Custom2OpTransformer1(BaseEstimator, TransformerMixin):

    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

    def fit(self, X, y=None):
        self.norm_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        return self.norm_.transform(X)


def custom_shape_calculator(operator):
    operator.outputs[0].type = FloatTensorType(
        shape=operator.inputs[0].type.shape)


def custom_transformer_converter1(scope, operator, container):
    i0 = operator.inputs[0]
    outputs = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    out = OnnxSubEstimator(op.norm_, i0, op_version=opv)
    final = OnnxIdentity(out, op_version=opv,
                         output_names=outputs)
    final.add_to(scope, container)


class Custom2OpTransformer1w(Custom2OpTransformer1):
    pass


def custom_transformer_converter1w(scope, operator, container):
    i0 = operator.inputs[0]
    outputs = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    out = OnnxSubEstimator(op.norm_, i0, op_version=opv)
    final = OnnxIdentity(out, op_version=opv,
                         output_names=outputs)
    final.add_to(scope, container)


class Custom2OpTransformer1ww(Custom2OpTransformer1):
    pass


def custom_transformer_converter1ww(scope, operator, container):
    i0 = operator.inputs[0]
    outputs = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    idin = OnnxIdentity(i0, op_version=opv)
    out = OnnxSubEstimator(op.norm_, idin, op_version=opv)
    final = OnnxIdentity(out, op_version=opv,
                         output_names=outputs)
    final.add_to(scope, container)


class Custom2OpTransformer2(Custom2OpTransformer1):
    pass


def custom_transformer_converter2(scope, operator, container):
    i0 = operator.inputs[0]
    outputs = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    out = OnnxSubEstimator(op.norm_, i0, op_version=opv,
                           output_names=outputs)
    out.add_to(scope, container)


class Custom2OpTransformer3(Custom2OpTransformer1):

    def fit(self, X, y=None):
        self.norm_ = LogisticRegression().fit(X, y)
        return self

    def transform(self, X):
        return self.norm_.predict_proba(X)


def custom_transformer_converter3(scope, operator, container):
    i0 = operator.inputs[0]
    outputs = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    out = OnnxSubEstimator(op.norm_, i0, op_version=opv,
                           options={'zipmap': False})
    final = OnnxIdentity(
        out[1], output_names=outputs, op_version=opv)
    final.add_to(scope, container)


class Custom2OpTransformer4(Custom2OpTransformer3):
    pass


def custom_transformer_converter4(scope, operator, container):
    i0 = operator.inputs[0]
    outputs = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    out = OnnxSubEstimator(op.norm_, i0, op_version=opv)
    final = OnnxIdentity(
        out[1], output_names=outputs, op_version=opv)
    final.add_to(scope, container)


class CustomOpClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)

    def fit(self, X, y=None):
        ncl = len(set(y))
        ycl = np.array(list(sorted(set(y))))
        self.estimators_ = []
        for i in range(0, ncl):
            yl = y.copy()
            yl[y != ycl[i]] = 0
            yl[y == ycl[i]] = 1
            logreg = LinearSVC().fit(X, yl)
            self.estimators_.append(logreg)
        self.classes_ = ycl
        return self

    def predict_proba(self, X):
        scores = np.zeros((X.shape[0], len(self.estimators_)), dtype=X.dtype)
        for i, est in enumerate(self.estimators_):
            scores[:, i] = est.decision_function(X).ravel()
        exp = np.exp(scores)
        exp /= exp.sum(axis=1, keepdims=1)
        return exp

    def predict(self, X):
        dec = self.predict_proba(X)
        return np.argmax(dec, axis=1)


def custom_classifier_converter(scope, operator, container):
    op = operator.raw_operator
    X = operator.inputs[0]
    outputs = operator.outputs
    opv = container.target_opset
    y_list = [
        OnnxReshape(
            OnnxSubEstimator(est, X, op_version=opv)[1],
            np.array([-1, 1], dtype=np.int64), op_version=opv)
        for est in op.estimators_]
    y_matrix = OnnxConcat(*y_list, axis=1, op_version=opv)
    probs = OnnxSoftmax(y_matrix, axis=1, op_version=opv,
                        output_names=[outputs[1]])
    probs.add_to(scope, container)
    labels = OnnxArgMax(probs, axis=1, keepdims=0, op_version=opv,
                        output_names=[outputs[0]])
    labels.add_to(scope, container)


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
        assert_almost_equal(expected, got, decimal=5)

    def check_classifier(self, obj, X):
        self.log.debug("[check_classifier------] type(obj)=%r" % type(obj))
        expected_labels = obj.predict(X)
        expected_probas = obj.predict_proba(X)
        onx = to_onnx(obj, X, target_opset=TARGET_OPSET,
                      options={id(obj): {'zipmap': False}})
        try:
            sess = InferenceSession(onx.SerializeToString())
        except InvalidArgument as e:
            raise AssertionError(
                "Issue %r with\n%s" % (e, str(onx))) from e
        got = sess.run(None, {'X': X})
        assert_almost_equal(expected_probas, got[1], decimal=5)
        assert_almost_equal(expected_labels, got[0])

    def test_custom_scaler_1(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = CustomOpTransformer1(op_version=TARGET_OPSET)
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_1_classic(self):
        update_registered_converter(
            Custom2OpTransformer1, 'Custom2OpTransformer1',
            custom_shape_calculator,
            custom_transformer_converter1)
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = Custom2OpTransformer1()
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_1w(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = CustomOpTransformer1w(op_version=TARGET_OPSET)
        tr.fit(X)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.check_transform(tr, X)
            self.assertEqual(len(w), 1)
            assert isinstance(w[0].message, DeprecationWarning)
            self.assertIn("to_onnx_operator", str(w[0].message))

    def test_custom_scaler_1w_classic(self):
        update_registered_converter(
            Custom2OpTransformer1w, 'Custom2OpTransformer1w',
            custom_shape_calculator,
            custom_transformer_converter1w)
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = Custom2OpTransformer1w()
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_1ww_classic(self):
        update_registered_converter(
            Custom2OpTransformer1ww, 'Custom2OpTransformer1ww',
            custom_shape_calculator,
            custom_transformer_converter1ww)
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = Custom2OpTransformer1ww()
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_2(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = CustomOpTransformer2(op_version=TARGET_OPSET)
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_2_classic(self):
        update_registered_converter(
            Custom2OpTransformer2, 'Custom2OpTransformer2',
            custom_shape_calculator,
            custom_transformer_converter2)
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = Custom2OpTransformer2()
        tr.fit(X)
        self.check_transform(tr, X)

    def test_custom_scaler_3(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        y = np.array([0, 0, 1], dtype=np.int64)
        tr = CustomOpTransformer3(op_version=TARGET_OPSET)
        tr.fit(X, y)
        self.check_transform(tr, X)

    def test_custom_scaler_3_classic(self):
        update_registered_converter(
            Custom2OpTransformer3, 'Custom2OpTransformer3',
            custom_shape_calculator,
            custom_transformer_converter3)
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        y = np.array([0, 0, 1], dtype=np.int64)
        tr = Custom2OpTransformer3()
        tr.fit(X, y)
        self.check_transform(tr, X)

    def test_custom_scaler_4(self):
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        y = np.array([0, 0, 1], dtype=np.int64)
        tr = CustomOpTransformer4(op_version=TARGET_OPSET)
        tr.fit(X, y)
        self.check_transform(tr, X)

    def test_custom_scaler_4_classic(self):
        update_registered_converter(
            Custom2OpTransformer4, 'Custom2OpTransformer4',
            custom_shape_calculator,
            custom_transformer_converter4)
        X = np.array([[0., 1.], [0., 1.], [2., 2.]], dtype=np.float32)
        tr = Custom2OpTransformer1()
        tr.fit(X)
        self.check_transform(tr, X)

    @ignore_warnings(category=ConvergenceWarning)
    def test_custom_classifier(self):
        update_registered_converter(
            CustomOpClassifier, 'CustomOpClassifier',
            calculate_linear_classifier_output_shapes,
            custom_classifier_converter,
            options={'zipmap': [False, True],
                     'nocl': [False, True]})
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(np.float32)
        cls = CustomOpClassifier()
        cls.fit(X, y)
        self.check_classifier(cls, X)


if __name__ == "__main__":
    # cl = TestCustomModelAlgebraSubEstimator()
    # cl.setUp(log=False)
    # cl.test_custom_scaler_2()
    unittest.main()
