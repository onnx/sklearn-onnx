# SPDX-License-Identifier: Apache-2.0

"""
Tests topology.
"""
import unittest
import numpy
from onnxruntime import InferenceSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn import datasets
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxIdentity, OnnxAdd
from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
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

    @ignore_warnings(category=DeprecationWarning)
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

    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_subgraphs1(self):
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(
            OnnxIdentity('input', op_version=TARGET_OPSET),
            'input', op_version=TARGET_OPSET)
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=TARGET_OPSET)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'],
                            op_version=TARGET_OPSET)

        model_def = cop2.to_onnx(
            {'input': FloatTensorType([None, None])},
            outputs=[('cdist', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        self.assertEqual(len(res), 1)

    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_subgraphs2(self):
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(
            OnnxIdentity('input', op_version=TARGET_OPSET),
            'input', op_version=TARGET_OPSET)
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=TARGET_OPSET)
        id1 = [id(a) for a in cdist.onx_op.graph_algebra['body']]
        cdist2 = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=TARGET_OPSET)
        id2 = [id(a) for a in cdist2.onx_op.graph_algebra['body']]
        self.assertNotEqual(id1, id2)
        cop2 = OnnxAdd(cdist, cdist2, output_names=['cdist'],
                       op_version=TARGET_OPSET)

        model_def = cop2.to_onnx(
            {'input': FloatTensorType([None, None])},
            outputs=[('cdist', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        self.assertEqual(len(res), 1)


if __name__ == "__main__":
    # import logging
    # log = logging.getLogger('skl2onnx')
    # log.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestTopologyPrune().test_onnx_subgraphs2()
    unittest.main()
