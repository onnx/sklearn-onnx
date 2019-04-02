# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from onnx.onnx_ml_pb2 import TensorProto
from ..proto import onnx_proto
from ..common._topology import Variable


class GraphState:

    def __init__(self, inputs, outputs,
                 operator_name,
                 scope, container, converter,
                 **attrs):
        self.inputs = inputs
        self.scope = scope
        self.operator_name = operator_name
        self.container = container
        self.converter = converter
        self.expected_outputs = outputs
        self.computed_outputs = None
        self.attrs = attrs
        if isinstance(self.inputs, tuple):
            raise TypeError("inputs must be a list or a string or a Variable.")
        elif not isinstance(self.inputs, list):
            self.inputs = [self.inputs]
        if self.expected_outputs is None:
            raise ValueError("expected_outputs must be named.")
        if not isinstance(self.expected_outputs, list):
            self.expected_outputs = [self.expected_outputs]

    @property
    def outputs(self):
        self.run()
        return self.computed_outputs

    def _get_var_name(self, var, output):
        if isinstance(var, Variable):
            return var.full_name
        elif isinstance(var, np.ndarray):
            return self._add_constant(var)
        elif hasattr(var, 'ConstantValue'):            
            return self._add_constant(var.ConstantValue)
        elif hasattr(var, 'add_to'):
            var.add_to(self.scope, self.container)
            outputs = var.outputs
            if isinstance(outputs, list):
                if len(outputs) == 1:
                    var = outputs[0]
                    if isinstance(var, Variable):
                        return var.full_name
                    elif isinstance(var, str):
                        return var
            raise RuntimeError("Unexpected type {}".format(outputs))
        elif hasattr(var, 'name') and isinstance(var.name, str) and var.name:
            return var.name
        elif isinstance(var, str):
            return var
        else:
            raise RuntimeError("Unexpected type: {0}".format(type(var)))

    def _add_constant(self, cst):
        if isinstance(cst, np.ndarray):
            shape = cst.shape
            name = self.scope.get_unique_variable_name(
                self.operator_name + 'cst')
            if cst.dtype in (np.float32, np.float64):
                ty = onnx_proto.TensorProto.FLOAT
            else:
                raise NotImplementedError(
                    "Unable to guess ONNX type from type {}.".format(
                        cst.dtype))
            self.container.add_initializer(
                name, ty, shape, cst.astype(np.float64).flatten())
            return name
        elif isinstance(cst, TensorProto):
            name = self.scope.get_unique_variable_name(
                self.operator_name + 'cst')
            self.container.add_initializer(name, None, None, cst)
            return name
        else:
            raise NotImplementedError(
                "Unable to add a constant of type {}.".format(type(cst)))

    def _get_output_name(self, output):
        if isinstance(output, Variable):
            return output.full_name
        elif isinstance(output, str):
            return output
        else:
            raise NotImplementedError(
                "Unexpected type {}".format(type(output)))

    def run(self):

        if self.computed_outputs is None:
            if self.expected_outputs is None:
                self.expected_outputs = [self._get_var_name(
                    o, True) for o in self.expected_outputs]
            inputs = [self._get_var_name(i, False) for i in self.inputs]
            inputs = [i for i in inputs if i is not None]
            name = self.scope.get_unique_operator_name(self.operator_name)
            outputs = [self._get_output_name(o) for o in self.expected_outputs]
            self.container.add_node(self.operator_name, inputs, outputs,
                                    name=name, **self.attrs)
            self.computed_outputs = self.expected_outputs
