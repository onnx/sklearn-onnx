# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MaxAbsScaler
from skl2onnx import update_registered_converter, to_onnx, get_model_alias
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common.utils import check_input_and_output_numbers
from skl2onnx.algebra.onnx_ops import OnnxCast, OnnxIdentity
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx.sklapi import WOETransformer
import skl2onnx.sklapi.register  # noqa
from test_utils import TARGET_OPSET


class OrdinalWOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X, y, sample_weight=None):
        self.encoder_ = OrdinalEncoder().fit(X)
        tr = self.encoder_.transform(X)
        maxi = (tr.max(axis=1) + 1).astype(np.int64)
        intervals = [[(i-1, i, False, True) for i in range(0, m)]
                     for m in maxi]
        weights = [[10 * j + i for i in range(len(inter))]
                   for j, inter in enumerate(intervals)]
        self.woe_ = WOETransformer(intervals, onehot=False, weights=weights)
        self.woe_.fit(tr)
        return self

    def transform(self, X):
        tr = self.encoder_.transform(X)
        return self.woe_.transform(tr)


def ordwoe_encoder_parser(
        scope, model, inputs, custom_parsers=None):
    if len(inputs) != 1:
        raise RuntimeError(
            "Unexpected number of inputs: %d != 1." % len(inputs))
    if inputs[0].type is None:
        raise RuntimeError(
            "Unexpected type: %r." % (inputs[0], ))
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs.append(inputs[0])
    this_operator.outputs.append(
        scope.declare_local_variable('catwoe', FloatTensorType()))
    return this_operator.outputs


def ordwoe_encoder_shape_calculator(operator):
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=1)
    input_dim = operator.inputs[0].get_first_dimension()
    shape = operator.inputs[0].type.shape
    second_dim = None if len(shape) != 2 else shape[1]
    output_type = FloatTensorType([input_dim, second_dim])
    operator.outputs[0].type = output_type


def ordwoe_encoder_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    X = operator.inputs[0]

    sub = OnnxSubEstimator(op.encoder_, X, op_version=opv)
    cast = OnnxCast(sub, op_version=opv, to=np.float32)
    cat = OnnxSubEstimator(op.woe_, cast, op_version=opv,
                           input_types=[Int64TensorType()])
    idcat = OnnxIdentity(cat, output_names=operator.outputs[:1],
                         op_version=opv)
    idcat.add_to(scope, container)


class TestCustomTransformerOrdWOE(unittest.TestCase):

    def test_pipeline(self):
        data = load_iris()
        X = data.data.astype(np.float32)
        pipe = make_pipeline(StandardScaler(), MaxAbsScaler())
        pipe.fit(X)
        expected = pipe.transform(X)
        onx = to_onnx(pipe, X, target_opset=TARGET_OPSET)
        sess = InferenceSession(onx.SerializeToString())
        got = sess.run(None, {'X': X})[0]
        assert_almost_equal(expected, got)

    @unittest.skipIf(TARGET_OPSET < 12, reason="opset>=12 is required")
    def test_custom_ordinal_woe(self):

        update_registered_converter(
            OrdinalWOETransformer, "OrdinalWOETransformer",
            ordwoe_encoder_shape_calculator,
            ordwoe_encoder_converter,
            parser=ordwoe_encoder_parser)

        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(np.int64)[:, :2]
        y = (y == 2).astype(np.int64)

        ordwoe = OrdinalWOETransformer()
        ordwoe.fit(X, y)
        expected = ordwoe.transform(X)

        onx = to_onnx(ordwoe, X, target_opset=TARGET_OPSET)
        sess = InferenceSession(onx.SerializeToString())
        got = sess.run(None, {'X': X})[0]
        assert_almost_equal(expected, got)


if __name__ == "__main__":
    unittest.main()
