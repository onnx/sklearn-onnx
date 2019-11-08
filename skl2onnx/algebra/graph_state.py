# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from onnx import AttributeProto
from ..proto import onnx_proto, TensorProto
from ..common._topology import Variable


class GraphState:

    def __init__(self, inputs, outputs,
                 operator_name, scope,
                 container, converter,
                 onnx_prefix_name=None,
                 **attrs):
        self.inputs = inputs
        self.scope = scope
        self.operator_name = operator_name
        self.container = container
        self.converter = converter
        self.expected_outputs = outputs
        self.computed_outputs = None
        self.onnx_prefix_name = onnx_prefix_name
        self.attrs = attrs
        if isinstance(self.inputs, tuple):
            raise TypeError("Parameter inputs must be a list or a string or a "
                            "Variable not tuple.")
        elif not isinstance(self.inputs, list):
            self.inputs = [self.inputs]
        if self.expected_outputs is None:
            raise ValueError("Parameter outputs must not be empty.")
        if not isinstance(self.expected_outputs, list):
            self.expected_outputs = [self.expected_outputs]

    @property
    def onnx_prefix(self):
        if self.onnx_prefix_name is None:
            return self.operator_name
        else:
            return self.onnx_prefix_name + "_" + self.operator_name

    @property
    def outputs(self):
        self.run()
        return self.computed_outputs

    def _get_var_name(self, var, unused, operator=None):
        if isinstance(var, Variable):
            return var.full_name
        elif isinstance(var, (np.ndarray, np.bool, np.int64,
                              np.float32, np.float64, np.bool)):
            return self._add_constant(var)
        elif hasattr(var, 'ConstantValue'):
            return self._add_constant(var.ConstantValue, var.ImplicitCast)
        elif hasattr(var, 'add_to'):
            var.add_to(self.scope, self.container, operator=operator)
            outputs = var.outputs
            if isinstance(outputs, list):
                if len(outputs) == 1:
                    var = outputs[0]
                    if isinstance(var, Variable):
                        return var.full_name
                    elif isinstance(var, str):
                        return var
            raise RuntimeError("Unexpected output type {}".format(outputs))
        elif hasattr(var, 'name') and isinstance(var.name, str) and var.name:
            return var.name
        elif isinstance(var, str):
            return var
        else:
            raise RuntimeError("Unexpected type for parameter 'var': {0}."
                               "".format(type(var)))

    def _add_constant(self, cst, can_cast=True):
        if isinstance(cst, np.ndarray):
            shape = cst.shape
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            if cst.dtype == np.float32:
                ty = onnx_proto.TensorProto.FLOAT
                astype = np.float64
            elif cst.dtype == np.float64:
                ty = onnx_proto.TensorProto.DOUBLE
                astype = np.float64
            elif cst.dtype == np.int64:
                ty = onnx_proto.TensorProto.INT64
                astype = np.int64
            elif cst.dtype == np.int32:
                ty = onnx_proto.TensorProto.INT32
                astype = np.int64
            elif cst.dtype == np.bool:
                ty = onnx_proto.TensorProto.BOOL
                astype = np.bool
            else:
                st = str(cst.dtype).lower()
                if st.startswith('u') or st.startswith("<u"):
                    ty = onnx_proto.TensorProto.STRING
                    astype = None
                    cst = np.array([s.encode('utf-8') for s in cst])
                else:
                    raise NotImplementedError(
                        "Unable to guess ONNX type from type {}. "
                        "You may raise an issue at https://github.com/onnx/"
                        "sklearn-onnx/issues.".format(
                            cst.dtype))
            if astype is not None:
                cst = cst.astype(astype)
            self.container.add_initializer(
                name, ty, shape, cst.flatten(),
                can_cast=can_cast)
            return name
        elif isinstance(cst, TensorProto):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            self.container.add_initializer(name, None, None, cst)
            return name
        elif isinstance(cst, np.int64):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = AttributeProto.INT
            self.container.add_initializer(name, ty, None, cst)
            return name
        elif isinstance(cst, np.bool):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = AttributeProto.INT
            self.container.add_initializer(name, ty, None, cst)
            return name
        elif isinstance(cst, np.float64):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = AttributeProto.DOUBLE
            self.container.add_initializer(name, ty, None, float(cst))
            return name
        elif isinstance(cst, np.float32):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = AttributeProto.FLOAT
            self.container.add_initializer(name, ty, None, float(cst))
            return name
        else:
            raise NotImplementedError(
                "Unable to add a constant of type {}. "
                "You may raise an issue at https://github.com/onnx/"
                "sklearn-onnx/issues.".format(type(cst)))

    def _get_output_name(self, output):
        if isinstance(output, Variable):
            return output.full_name
        elif isinstance(output, str):
            return output
        elif isinstance(output, tuple):
            return output[0]
        else:
            raise NotImplementedError(
                "Unexpected output type {} [{}]. "
                "You may raise an issue at https://github.com/onnx/"
                "sklearn-onnx/issues.".format(type(output), output))

    def run(self, operator=None):
        if self.computed_outputs is None:
            if self.expected_outputs is None:
                eoli = [self._get_var_name(o, True, operator=operator)
                        for o in self.expected_outputs]
                self.expected_outputs = eoli
            inputs = []
            for i in self.inputs:
                v = self._get_var_name(i, False, operator=operator)
                if v is not None:
                    inputs.append(v)
            name = self.scope.get_unique_operator_name(self.onnx_prefix)
            outputs = [self._get_output_name(o)
                       for o in self.expected_outputs]
            self.container.add_node(self.operator_name, inputs, outputs,
                                    name=name, **self.attrs)
            self.computed_outputs = self.expected_outputs
