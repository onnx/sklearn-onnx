# SPDX-License-Identifier: Apache-2.0

import unittest
import inspect
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.base import (
    BaseEstimator, ClassifierMixin, clone, TransformerMixin)
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from onnxruntime import InferenceSession, __version__ as ort_version
from skl2onnx.algebra.onnx_ops import (
    OnnxIdentity, OnnxCast, OnnxReduceMax, OnnxGreater,
    OnnxExp)
from skl2onnx import update_registered_converter
from skl2onnx import to_onnx, get_model_alias
from skl2onnx.proto import onnx_proto
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType)
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from test_utils import TARGET_OPSET


class ValidatorClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator=None, threshold=0.75):
        ClassifierMixin.__init__(self)
        BaseEstimator.__init__(self)
        if estimator is None:
            estimator = LogisticRegression(solver='liblinear')
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None):
        sig = inspect.signature(self.estimator.fit)
        if 'sample_weight' in sig.parameters:
            self.estimator_ = clone(self.estimator).fit(
                X, y, sample_weight=sample_weight)
        else:
            self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def validate(self, X):
        pred = self.predict_proba(X)
        mx = pred.max(axis=1)
        return (mx >= self.threshold) * 1


def validator_classifier_shape_calculator(operator):

    input = operator.inputs[0]  # inputs in ONNX graph
    outputs = operator.outputs  # outputs in ONNX graph
    op = operator.raw_operator  # scikit-learn model (mmust be fitted)
    if len(outputs) != 3:
        raise RuntimeError("3 outputs expected not {}.".format(len(outputs)))

    N = input.type.shape[0]                 # number of observations
    C = op.estimator_.classes_.shape[0]     # dimension of outputs

    outputs[0].type = Int64TensorType([N])     # label
    outputs[1].type = FloatTensorType([N, C])  # probabilities
    outputs[2].type = Int64TensorType([C])     # validation


def validator_classifier_converter(scope, operator, container):
    input = operator.inputs[0]      # input in ONNX graph
    outputs = operator.outputs      # outputs in ONNX graph
    op = operator.raw_operator      # scikit-learn model (mmust be fitted)
    opv = container.target_opset

    # We reuse existing converter and declare it as local
    # operator.
    model = op.estimator_
    onnx_op = OnnxSubEstimator(model, input, op_version=opv,
                               options={'zipmap': False})

    rmax = OnnxReduceMax(onnx_op[1], axes=[1], keepdims=0, op_version=opv)
    great = OnnxGreater(rmax, np.array([op.threshold], dtype=np.float32),
                        op_version=opv)
    valid = OnnxCast(great, to=onnx_proto.TensorProto.INT64,
                     op_version=opv)

    r1 = OnnxIdentity(onnx_op[0], output_names=[outputs[0].full_name],
                      op_version=opv)
    r2 = OnnxIdentity(onnx_op[1], output_names=[outputs[1].full_name],
                      op_version=opv)
    r3 = OnnxIdentity(valid, output_names=[outputs[2].full_name],
                      op_version=opv)

    r1.add_to(scope, container)
    r2.add_to(scope, container)
    r3.add_to(scope, container)


def validator_classifier_parser(scope, model, inputs, custom_parsers=None):
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)

    # inputs
    this_operator.inputs.append(inputs[0])

    # outputs
    val_label = scope.declare_local_variable('val_label', Int64TensorType())
    val_prob = scope.declare_local_variable('val_prob', FloatTensorType())
    val_val = scope.declare_local_variable('val_val', Int64TensorType())
    this_operator.outputs.append(val_label)
    this_operator.outputs.append(val_prob)
    this_operator.outputs.append(val_val)

    # ends
    return this_operator.outputs


def dummy1_parser(scope, model, inputs):
    pass


def dummy2_parser(scope, model, input, custom_parsers):
    pass


def dummy_val_2(op, c):
    pass


def dummy_conv_1(scope, op, cont):
    pass


def dummy_conv_2(scope, operator):
    pass


class MinMaxScalerTwo(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.est1_ = MinMaxScaler().fit(X)
        x2 = np.exp(self.est1_.transform(X))
        self.est2_ = MaxAbsScaler().fit(x2)
        return self

    def transform(self, X):
        x2 = np.exp(self.est1_.transform(X))
        return self.est2_.transform(x2)


def subsub_mmtwo_parser(scope, model, inputs, custom_parsers=None):
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs.append(inputs[0])
    cls_type = inputs[0].type.__class__
    val = scope.declare_local_variable('variable', cls_type())
    this_operator.outputs.append(val)
    return this_operator.outputs


def subsub_mmtwo_shape_calculator(operator):
    operator.outputs[0].type.shape = operator.inputs[0].type.shape


def subsub_mmtwo_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    x2 = OnnxSubEstimator(op.est1_, X, op_version=opv)
    x2.set_onnx_name_prefix('AAA')
    x2_exp = OnnxExp(x2, op_version=opv)
    x3 = OnnxSubEstimator(op.est2_, x2_exp, op_version=opv)
    x3.set_onnx_name_prefix('BBB')
    final = OnnxIdentity(x3, op_version=opv, output_names=out[:1])
    final.add_to(scope, container)


class TestOnnxOperatorSubEstimator(unittest.TestCase):

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.0"),
        reason="Cast not available.")
    def test_sub_estimator_exc(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = ValidatorClassifier()
        model.fit(X_train, y_train)

        # parser

        try:
            update_registered_converter(
                ValidatorClassifier, 'CustomValidatorClassifier',
                validator_classifier_shape_calculator,
                validator_classifier_converter,
                parser=dummy1_parser)
            raise AssertionError("exception not raised")
        except TypeError:
            pass

        try:
            update_registered_converter(
                ValidatorClassifier, 'CustomValidatorClassifier',
                validator_classifier_shape_calculator,
                validator_classifier_converter,
                parser=dummy1_parser)
            raise AssertionError("exception not raised")
        except TypeError:
            pass

        # shape

        try:
            update_registered_converter(
                ValidatorClassifier, 'CustomValidatorClassifier',
                dummy_val_2,
                validator_classifier_converter,
                parser=validator_classifier_parser)
            raise AssertionError("exception not raised")
        except TypeError:
            pass

        # conv

        try:
            update_registered_converter(
                ValidatorClassifier, 'CustomValidatorClassifier',
                validator_classifier_shape_calculator,
                dummy_conv_1,
                parser=validator_classifier_parser)
            raise AssertionError("exception not raised")
        except NameError:
            pass

        try:
            update_registered_converter(
                ValidatorClassifier, 'CustomValidatorClassifier',
                validator_classifier_shape_calculator,
                dummy_conv_2,
                parser=validator_classifier_parser)
            raise AssertionError("exception not raised")
        except TypeError:
            pass

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.0"),
        reason="Cast not available.")
    def test_sub_estimator(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = ValidatorClassifier()
        model.fit(X_train, y_train)

        update_registered_converter(
            ValidatorClassifier, 'CustomValidatorClassifier',
            validator_classifier_shape_calculator,
            validator_classifier_converter,
            parser=validator_classifier_parser)

        X32 = X_test[:5].astype(np.float32)
        model_onnx = to_onnx(
            model, X32, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': X32})
        assert_almost_equal(model.predict(X32), res[0])
        assert_almost_equal(model.predict_proba(X32), res[1], decimal=4)
        assert_almost_equal(model.validate(X32), res[2])

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.0"),
        reason="Cast not available.")
    def test_sub_sub_estimator(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = MinMaxScalerTwo()
        model.fit(X_train, y_train)

        update_registered_converter(
            MinMaxScalerTwo, "SubSubDummy",
            subsub_mmtwo_shape_calculator,
            subsub_mmtwo_converter,
            parser=subsub_mmtwo_parser)

        X32 = X_test[:5].astype(np.float32)
        model_onnx = to_onnx(
            model, X32, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': X32})
        assert_almost_equal(model.transform(X32), res[0], decimal=5)


if __name__ == "__main__":
    unittest.main()
