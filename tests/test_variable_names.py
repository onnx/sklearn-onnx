# SPDX-License-Identifier: Apache-2.0

import unittest
import copy
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.pipeline import Pipeline
from onnxruntime import InferenceSession
from skl2onnx import get_model_alias, update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from skl2onnx import convert_sklearn
from onnxconverter_common.data_types import FloatTensorType
from test_utils import TARGET_OPSET


class Passthrough:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def parser(scope, model, inputs, custom_parsers=None):
    alias = get_model_alias(type(model))
    operator = scope.declare_local_operator(alias, model)
    operator.inputs = inputs
    for op_input in inputs:
        op_output = scope.declare_local_variable(
            op_input.raw_name, copy.deepcopy(op_input.type))
        operator.outputs.append(op_output)
    return operator.outputs


def shape_calculator(operator):
    op_input_map = {op_input.raw_name: op_input
                    for op_input in operator.inputs}
    for op_output in operator.outputs:
        op_output.type.shape = op_input_map[op_output.raw_name].type.shape


def converter(scope, operator, container):
    op_input_map = {op_input.raw_name: op_input
                    for op_input in operator.inputs}
    for op_output in operator.outputs:
        op_input = op_input_map[op_output.raw_name]
        OnnxIdentity(
            op_input,
            output_names=[op_output],
            op_version=container.target_opset,
        ).add_to(scope, container)


class TestVariableNames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        update_registered_converter(
            Passthrough, "Passthrough",
            shape_calculator, converter,
            parser=parser)

    def test_variable_names(self):
        pipeline = Pipeline([("passthrough", Passthrough())])
        initial_types = [("input", FloatTensorType([None, 2]))]
        model_onnx = convert_sklearn(pipeline, initial_types=initial_types,
                                     target_opset=TARGET_OPSET,
                                     verbose=0)
        self.assertIn('Identity', str(model_onnx))
        x = np.array([0, 1, 1, 0], dtype=np.float32).reshape((-1, 2))
        sess = InferenceSession(model_onnx.SerializeToString())
        name = sess.get_inputs()[0].name
        got = sess.run(None, {name: x})
        assert_almost_equal(x, got[0])

    def test_variable_names_distinct(self):
        pipeline = Pipeline([("passthrough", Passthrough())])
        initial_types = [("INPUTA", FloatTensorType([None, 2]))]
        final_types = [("OUTPUTA", FloatTensorType([None, 2]))]
        model_onnx = convert_sklearn(pipeline, initial_types=initial_types,
                                     target_opset=TARGET_OPSET,
                                     final_types=final_types,
                                     verbose=0)
        x = np.array([0, 1, 1, 0], dtype=np.float32).reshape((-1, 2))
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'INPUTA': x})
        assert_almost_equal(x, got[0])

    def test_variable_names_output(self):
        pipeline = Pipeline([("passthrough", Passthrough())])
        initial_types = [("input", FloatTensorType([None, 2]))]
        final_types = initial_types
        with self.assertRaises(RuntimeError):
            convert_sklearn(pipeline, initial_types=initial_types,
                            target_opset=TARGET_OPSET,
                            final_types=final_types)


if __name__ == "__main__":
    unittest.main()
