# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from scipy.sparse import coo_matrix
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
        if hasattr(operator_name, 'fit'):
            from .. import get_model_alias
            self.operator_instance = operator_name
            self.is_model = True
            self.operator_name = get_model_alias(type(operator_name))
        else:
            self.operator_name = operator_name
            self.is_model = False
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
        if self.expected_outputs is None and not self.is_model:
            raise ValueError("Parameter outputs must not be empty.")
        if (not isinstance(self.expected_outputs, list) and
                self.expected_outputs is not None):
            self.expected_outputs = [self.expected_outputs]

    @property
    def onnx_prefix(self):
        if self.onnx_prefix_name is None:
            return self.operator_name
        return self.onnx_prefix_name + "_" + self.operator_name

    @property
    def outputs(self):
        self.run()
        return self.computed_outputs

    def _get_var_name(self, var, unused, operator=None):
        if isinstance(var, Variable):
            return var.full_name
        if isinstance(var, (np.ndarray, np.bool, np.int64,
                            np.float32, np.float64, np.bool,
                            np.int8, np.uint8)):
            return self._add_constant(var)
        elif hasattr(var, 'ConstantValue'):
            return self._add_constant(var.ConstantValue)
        elif hasattr(var, 'add_to'):
            var.add_to(self.scope, self.container, operator=operator)
            outputs = var.outputs
            if isinstance(outputs, list):
                if len(outputs) == 1:
                    var = outputs[0]
                    if isinstance(var, Variable):
                        return var.full_name
                    if isinstance(var, str):
                        return var
            raise RuntimeError("Unexpected output type {}".format(outputs))
        if hasattr(var, 'name') and isinstance(var.name, str) and var.name:
            return var.name
        if isinstance(var, str):
            return var
        raise RuntimeError("Unexpected type for parameter 'var': {0}."
                           "".format(type(var)))

    def _add_constant(self, cst):

        def _ty_astype(cst):
            dtype = cst.dtype
            if dtype == np.float32:
                ty = onnx_proto.TensorProto.FLOAT
                astype = np.float64
            elif dtype == np.float64:
                ty = onnx_proto.TensorProto.DOUBLE
                astype = np.float64
            elif dtype == np.int64:
                ty = onnx_proto.TensorProto.INT64
                astype = np.int64
            elif dtype == np.int32:
                ty = onnx_proto.TensorProto.INT32
                astype = np.int32
            elif dtype == np.int8:
                ty = onnx_proto.TensorProto.INT8
                astype = np.int8
            elif dtype == np.uint8:
                ty = onnx_proto.TensorProto.UINT8
                astype = np.uint8
            elif dtype == np.bool:
                ty = onnx_proto.TensorProto.BOOL
                astype = np.bool
            else:
                st = str(dtype).lower()
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
            return cst, ty, astype

        if isinstance(cst, np.ndarray):
            shape = cst.shape
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            cst, ty, astype = _ty_astype(cst)
            if astype is not None:
                cst = cst.astype(astype)
            self.container.add_initializer(
                name, ty, shape, cst.flatten())
            return name
        if isinstance(cst, coo_matrix):
            shape = cst.shape
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            cst, ty, astype = _ty_astype(cst)
            self.container.add_initializer(
                name, ty, shape, cst.astype(astype))
            return name
        if isinstance(cst, TensorProto):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            self.container.add_initializer(name, None, None, cst)
            return name
        if isinstance(cst, np.int64):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.INT64
            self.container.add_initializer(name, ty, None, cst)
            return name
        if isinstance(cst, np.bool):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.BOOL
            self.container.add_initializer(name, ty, None, cst)
            return name
        if isinstance(cst, np.float64):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.DOUBLE
            self.container.add_initializer(name, ty, None, float(cst))
            return name
        if isinstance(cst, np.float32):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.FLOAT
            self.container.add_initializer(name, ty, None, float(cst))
            return name
        if isinstance(cst, np.int8):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.INT8
            self.container.add_initializer(name, ty, None, cst)
            return name
        if isinstance(cst, np.uint8):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.UINT8
            self.container.add_initializer(name, ty, None, cst)
            return name
        raise NotImplementedError(
            "Unable to add a constant of type {}. "
            "You may raise an issue at https://github.com/onnx/"
            "sklearn-onnx/issues.".format(type(cst)))

    def _get_output_name(self, output):
        if isinstance(output, Variable):
            return output.full_name
        if isinstance(output, str):
            return output
        if isinstance(output, tuple):
            return output[0]
        raise NotImplementedError(
            "Unexpected output type {} [{}]. "
            "You may raise an issue at https://github.com/onnx/"
            "sklearn-onnx/issues.".format(type(output), output))

    def run(self, operator=None):
        if self.computed_outputs is None:
            if self.expected_outputs is not None:
                eoli = [self._get_var_name(o, True, operator=operator)
                        for o in self.expected_outputs]
                self.expected_outputs = eoli
            inputs = []
            for i in self.inputs:
                v = self._get_var_name(i, False, operator=operator)
                if v is not None:
                    inputs.append(v)
            name = self.scope.get_unique_operator_name(self.onnx_prefix)
            if self.is_model:
                # a model is converted into a subgraph
                sub_op = self.scope.declare_local_operator(
                    self.operator_name, self.operator_instance)
                sub_op.inputs = self.inputs
                if self.expected_outputs is None:
                    # output are not defined, we need to call a parser.
                    from .._parse import _parse_sklearn
                    self.expected_outputs = _parse_sklearn(
                        self.scope, self.operator_instance, self.inputs)
                    if (self.expected_outputs is None or
                            None in self.expected_outputs):
                        raise RuntimeError(
                            "Wrong result when parsing model {}.".format(
                                type(self.operator_instance)))
                    self.expected_outputs = [
                        Variable(v.raw_name,
                                 self.scope.get_unique_variable_name(
                                    v.raw_name),
                                 self.scope, v.type)
                        for v in self.expected_outputs]
                sub_op.outputs = self.expected_outputs
            else:
                # only one node is added
                outputs = [self._get_output_name(o)
                           for o in self.expected_outputs]
                self.container.add_node(self.operator_name, inputs, outputs,
                                        name=name, **self.attrs)
            self.computed_outputs = self.expected_outputs
