# SPDX-License-Identifier: Apache-2.0

"""
Tests topology.
"""
import unittest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn import datasets

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from test_utils import TARGET_OPSET


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X, y, sample_weight=None):
        return self

    def transform(self, X):
        return X


class identity(IdentityTransformer):

    def __init__(self):
        IdentityTransformer.__init__(self)


def dummy_shape_calculator(operator):
    op_input = operator.inputs[0]
    operator.outputs[0].type.shape = op_input.type.shape


def dummy_converter(scope, operator, container):
    X = operator.inputs[0]
    out = operator.outputs

    id1 = OnnxIdentity(X, op_version=TARGET_OPSET)
    id2 = OnnxIdentity(id1, output_names=out[:1],
                       op_version=TARGET_OPSET)
    id2.add_to(scope, container)


class TestTopologyPrune(unittest.TestCase):

    def test_dummy_identity(self):

        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20]
        yd = digits.target[:20]
        n_samples, n_features = Xd.shape

        idtr = make_pipeline(IdentityTransformer(), identity())
        idtr.fit(Xd, yd)

        update_registered_converter(IdentityTransformer, "IdentityTransformer",
                                    dummy_shape_calculator, dummy_converter)
        update_registered_converter(identity, "identity",
                                    dummy_shape_calculator, dummy_converter)

        model_onnx = convert_sklearn(
            idtr, "idtr",
            [("input", FloatTensorType([None, Xd.shape[1]]))],
            target_opset=TARGET_OPSET)

        idnode = [node for node in model_onnx.graph.node
                  if node.op_type == "Identity"]
        self.assertEqual(len(idnode), 1)


if __name__ == "__main__":
    unittest.main()
