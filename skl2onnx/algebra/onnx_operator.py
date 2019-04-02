# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
import onnx
from ..proto import TensorProto
from ..common.data_types import FloatTensorType, Int64TensorType
from ..common.data_types import StringTensorType
from ..common.data_types import Int32TensorType, DoubleTensorType
from ..common.data_types import BoolTensorType
from ..common._topology import Variable
from ..proto import get_opset_number_from_onnx, onnx_proto
from ..helpers.onnx_helper import infer_outputs
from .graph_state import GraphState
from .symbolic_op_tranformer import SymbolicOpTransformer


class OnnxOperator:
    """
    Ancestor to every *ONNX* operator exposed in
    :mod:`onnx_ops` and :mod:`onnx_ops_ml`. These files
    are automatically generated by unit test
    *test_onnx_operators_parse_spec*
    Every instance is supposed to be included in
    a graph as a node.

    :param inputs: list of inputs expected by the operator
    :param op_version: to select a specific version of the operator
    :param output_names: used defined names for the outputs
    :param kwargs: additional parameters of the operator
    """

    class UnscopedVariable:
        def __init__(self, name):
            self.name = name

        def __eq__(self, name):
            if isinstance(name, str):
                return name == self.name
            elif isinstance(name, OnnxOperator.UnscopedVariable):
                return self.name == name.name
            else:
                raise TypeError('Unsupported type for comparison {}'.format(
                    type(name)))

    class ConstantVariable:
        def __init__(self, value):
            self.value = value

        @property
        def ConstantValue(self):
            return self.value

    def __init__(self, *inputs, op_version=None, output_names=None, **kwargs):
        self.state = None
        self.op_version = op_version or get_opset_number_from_onnx()
        self.kwargs = kwargs

        # check inputs
        if len(inputs) == 0:
            if self.input_range[0] == self.input_range[1]:
                self.inputs = [_[0] for _ in self.__class__.expected_inputs]
            else:
                # The number of inputs may vary.
                self.inputs = None
        else:
            self.inputs = []
            for inp in inputs:
                if isinstance(inp, str):
                    self.inputs.append(OnnxOperator.UnscopedVariable(inp))
                elif isinstance(inp, (OnnxOperator, Variable)):
                    self.inputs.append(inp)
                elif isinstance(inp, (np.ndarray, TensorProto)):
                    self.inputs.append(OnnxOperator.ConstantVariable(inp))
                else:
                    raise TypeError("Unable to interpret the "
                                    "input name for type {}.".format(
                                        type(inp)))

        if self.inputs is not None:
            if len(self.inputs) < self.input_range[0] or \
                    len(self.inputs) > self.input_range[1]:
                raise RuntimeError("Operator '{}' expects a number of inputs "
                                   "in [{}, {}] not {}".format(
                                        self.__class__.__name__,
                                        *self.input_range,
                                        len(self.inputs)))

        # check output
        if hasattr(output_names, 'outputs') and \
                output_names.outputs is not None:
            self.output_names = [out.full_name
                                 for out in output_names.outputs]
        else:
            self.output_names = output_names
        if self.output_names:
            for i in range(len(self.output_names)):
                name = self.output_names[i]
                if isinstance(name, Variable):
                    self.output_names[i] = name.onnx_name
                elif not isinstance(name, str):
                    raise TypeError("output_names must be a list of strings "
                                    "and element {} is {}".format(
                                        i, type(name)))

    def get_output(self, i):
        """
        Returns the ith output.
        """
        if hasattr(self, 'output_names_'):
            return self.output_names_[i]
        if self.output_names and i < len(self.output_names) and \
                self.output_names[i]:
            return self.output_names[i]
        if i < len(self.__class__.expected_outputs):
            return self.__class__.expected_outputs[i][0]
        else:
            return "O%d" % i

    def update_name(self, i, name):
        """
        Updates the name of a variable after it was scoped.
        """
        if hasattr(self, 'output_names_') and i < len(self.output_names_):
            if self.output_names_[i] != name:
                raise RuntimeError("Inconsistent, cannot "
                                   "changed variable name "
                                   "after it was used: "
                                   "'{}' != '{}'".format(
                                       self.output_names_[i],
                                       name))
        if self.output_names is None:
            self.output_names = []
        while len(self.output_names) <= i:
            self.output_names.append(None)
        self.output_names[i] = name

    def add_to(self, scope, container):
        """
        Adds outputs to the container if not already added,
        registered the outputs if the node is not final.

        :param scope: scope
        :param container: container
        """
        if self.state is None:
            if self.kwargs.get('op_version', '') is None:
                kwargs = self.kwargs.copy()
                del kwargs['op_version']
            else:
                kwargs = self.kwargs

            if hasattr(self, 'output_names_'):
                outputs = self.output_names_
            elif self.output_names:
                outputs = self.output_names
                self.output_names_ = outputs
            else:
                outputs = []
                for name in self.__class__.expected_outputs:
                    name = scope.get_unique_variable_name(name[0])
                    outputs.append(name)
                self.output_names_ = outputs

            self.state = GraphState(self.inputs, self.output_names_,
                                    self.__class__.__name__,
                                    scope, container, None,
                                    op_version=self.op_version,
                                    op_domain=self.__class__.domain,
                                    **self.kwargs)
            self.state.run()

    @property
    def outputs(self):
        """
        Returns the outputs of the node.
        """
        if self.state is None:
            raise RuntimeError("Method add was not called.")
        return self.state.outputs

    def parse_inputs(self, inputs):
        """
        Checks the given inputs follow the constraints
        defined by the operator. Done at parsing time.
        """
        if isinstance(inputs, list):
            inputs = {v.onnx_name: v for v in inputs}
        if not isinstance(inputs, dict):
            raise TypeError(
                "inputs must be a dictionary, not a {}".format(type(inputs)))
        if self.input_range is not None:
            if len(inputs) < self.input_range[0] or \
                    len(inputs) > self.input_range[1]:
                raise RuntimeError("Operator '{}' expects a number of inputs "
                                   "in [{}, {}] not {}".format(
                                       self.__class__.__name__,
                                       *self.input_range,
                                       len(inputs)))

        res = []
        for k, value in inputs.items():
            if self.__class__.input_range[1] == 2147483647:
                # infinity is allowed
                exp = self.__class__.expected_inputs[0]
                res.append(('I%d' % len(res), self.guess_type(exp[1], value)))
            elif isinstance(value, (OnnxOperator, Variable, str)):
                res.append((value, self.guess_type(None, value)))
            elif isinstance(value, np.ndarray):
                res.append((k, self.guess_type(None, value)))
            elif isinstance(value, TensorProto):
                res.append((k, self.guess_type(None, value)))
            else:
                raise TypeError("Unexpected input type: {}".format(
                    type(value)))
        return res

    def get_schema_nb_output(self, inputs):
        """
        Infers the number of outputs given the inputs.
        Used by the parser.
        """
        return len(self.__class__.expected_outputs)

    def get_typed_outputs(self, inputs, outputs):
        """
        Infers the output shapes and type given the inputs.
        Used by *set_shape*.
        """
        outputs = infer_outputs(self.__class__.__name__, inputs,
                                outputs, **self.kwargs)
        return outputs

    def _guess_typed_outputs(self, inputs):
        """
        Infers the output shapes and type given the inputs.
        Used by *set_shape*.
        """
        if self.output_range[0] == self.output_range[1]:
            nb = self.output_range[0]
        else:
            nb = min(
                max(len(inputs), self.output_range[0]), self.output_range[1])
        res = []
        if inputs:
            for i in range(nb):
                exp = self.expected_outputs[i]
                inp = inputs[i] if i < len(inputs) else None
                if exp[1]:
                    res.append(Variable(exp[0], exp[0], None, exp[1]))
                elif inp:
                    res.append(Variable(exp[0], exp[0], None, inp.type))
                else:
                    res.append(Variable(exp[0], exp[0]))
        else:
            for i in range(nb):
                exp = self.expected_outputs[i]
                if exp[1]:
                    res.append(Variable(exp[0], exp[0], None, exp[1]))
                else:
                    res.append(Variable(exp[0], exp[0]))
        return res

    def guess_type(self, expected_type, given_type):
        """
        Returns the proper type of an input.
        """
        if isinstance(given_type, np.ndarray):
            if given_type.dtype == np.float32:
                return FloatTensorType(given_type.shape)
            elif given_type.dtype == np.int64:
                return Int64TensorType(given_type.shape)
            elif given_type.dtype == np.str:
                return StringTensorType(given_type.shape)
            else:
                raise NotImplementedError(
                    "Unsupported type '{}'".format(given_type.dtype))
        elif isinstance(given_type, (FloatTensorType, Int64TensorType,
                                     StringTensorType)):
            return given_type
        elif isinstance(given_type, Variable):
            return given_type.type
        elif isinstance(given_type, onnx.onnx_ml_pb2.TensorProto):
            if given_type.data_type == onnx_proto.TensorProto.FLOAT:
                return FloatTensorType(given_type.dims)
            elif given_type.data_type == onnx_proto.TensorProto.DOUBLE:
                return DoubleTensorType(given_type.dims)
            elif given_type.data_type == onnx_proto.TensorProto.STRING:
                return StringTensorType(given_type.dims)
            elif given_type.data_type == onnx_proto.TensorProto.INT64:
                return Int64TensorType(given_type.dims)
            elif given_type.data_type == onnx_proto.TensorProto.INT32:
                return Int32TensorType(given_type.dims)
            elif given_type.data_type == onnx_proto.TensorProto.BOOL:
                return BoolTensorType(given_type.dims)
            else:
                raise NotImplementedError("Unsupported type '{}' "
                                          "data_type={}".format(
                                              type(given_type),
                                              given_type.data_type))
        else:
            raise NotImplementedError(
                "Unsupported type '{}'".format(type(given_type)))

    def to_onnx(self, inputs=None):
        """
        Converts this operator into an ONNX graph.

        :param inputs: specific inputs (as a dictionary) or
            default inputs if not specified
        """
        from .. import convert_sklearn
        if inputs:
            inputs = self.parse_inputs(inputs)
        else:
            inputs = self.expected_inputs
        for name, typ in inputs:
            if typ in (None, ''):
                raise RuntimeError("Type input '{}' for operator '{}' "
                                   "is unknown. You should specify "
                                   "input types.".format(
                                       name, self.__class__.__name__))
        new_cl = type('SymbolicOpTransformer' + self.__class__.__name__,
                      (SymbolicOpTransformer,), {})
        tr = new_cl(self)
        onx = convert_sklearn(tr, 'onnx-op', inputs,
                              target_opset=self.op_version,
                              custom_parsers={new_cl: tr.parse},
                              custom_shape_calculators={new_cl: tr.set_shape},
                              custom_conversion_functions={new_cl: tr.convert})
        return onx
