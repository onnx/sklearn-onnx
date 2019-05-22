# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from ..proto import TensorProto, helper
from ..common._topology import Variable, Scope
from ..common._container import ModelComponentContainer
from ..common import utils
from ..proto import get_opset_number_from_onnx, onnx_proto
from ..helpers.onnx_helper import infer_outputs
from .graph_state import GraphState
from .type_helper import _guess_type


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
    :param domain: to overwrite the default domain
    :param kwargs: additional parameters of the operator
    """
    class OnnxOperatorVariable:

        def __init__(self, index, name=None):
            self.index = index
            self.name = name

        def __repr__(self):
            return "OnnxOperatorVariable('%s')" % self.name

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

        def __repr__(self):
            return "UnscopedVariable('%s')" % self.name

    class ConstantVariable:
        def __init__(self, value):
            self.value = value

        @property
        def ConstantValue(self):
            return self.value

    def __init__(self, *inputs, op_version=None, output_names=None,
                 domain=None, **kwargs):
        self.state = None
        self.op_version = op_version or get_opset_number_from_onnx()
        self.domain = domain
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
                elif isinstance(inp, OnnxOperator.OnnxOperatorVariable):
                    self.inputs.append(inp)
                else:
                    raise TypeError("Unable to interpret the "
                                    "input name for type {}.".format(
                                        type(inp)))

        if self.inputs is not None:
            if (len(self.inputs) < self.input_range[0] or
                    len(self.inputs) > self.input_range[1]):
                raise RuntimeError("Operator '{}' expects a number of inputs "
                                   "in [{}, {}] not {}".format(
                                       self.operator_name,
                                       *self.input_range,
                                       len(self.inputs)))

        # check output
        if (hasattr(output_names, 'outputs') and
                output_names.outputs is not None):
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
        if (self.output_names and i < len(self.output_names) and
                self.output_names[i]):
            return self.output_names[i]
        if i < len(self.__class__.expected_outputs):
            return self.__class__.expected_outputs[i][0]
        elif i < self.__class__.output_range[1]:
            if i > 1000:
                raise IndexError("You should redesign your operator.")
            return "O%d" % i
        else:
            raise IndexError("Output %d does not exist." % i)

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

    def add_to(self, scope, container, operator=None):
        """
        Adds outputs to the container if not already added,
        registered the outputs if the node is not final.

        :param scope: scope
        :param container: container
        :param operator: overwrite inputs
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

            domain = self.domain
            if domain is None:
                domain = self.__class__.domain
            inputs = []
            for input in self.inputs:
                if isinstance(input, OnnxOperator.OnnxOperatorVariable):
                    if operator is None:
                        raise RuntimeError("A placeholder cannot be replaced "
                                           "as an operator is not specified.")
                    if len(operator.inputs) == 0:
                        raise RuntimeError("No input variable in {}.".format(
                            operator))
                    # The inputs must be looked into the graph.
                    for i in operator.inputs:
                        if i.raw_name == input.name:
                            inputs.append(i)
                            break
                    else:
                        vars = ', '.join(map(lambda o: "'%s'" % o.raw_name,
                                             operator.inputs))
                        raise RuntimeError("Unable to find variable "
                                           "{} in {}.".format(input, vars))
                else:
                    inputs.append(input)
            self.state = GraphState(inputs, self.output_names_,
                                    self.operator_name,
                                    scope, container, None,
                                    op_version=self.op_version,
                                    op_domain=domain,
                                    **self.kwargs)
            self.state.run(operator=operator)

    @property
    def outputs(self):
        """
        Returns the outputs of the node.
        """
        if self.state is None:
            raise RuntimeError("Method add_to was not called.")
        return self.state.outputs

    def to_onnx(self, inputs=None, outputs=None):
        """
        Converts this operator into an ONNX graph.

        :param inputs: specific inputs (as a dictionary) or
            default inputs if not specified
        :param outputs: specific outputs
        """
        if inputs is None:
            raise NotImplementedError("inputs must be specified.")
        if isinstance(inputs, dict):
            inputs = [(k, v) for k, v in inputs.items()]
        new_inputs = []
        for obj in inputs:
            if isinstance(obj, Variable):
                new_inputs.append((obj.onnx_name, obj.type))
            elif isinstance(obj, tuple) and len(obj) == 2:
                ty = _guess_type(obj[1])
                new_inputs.append((obj[0], ty))
            else:
                raise TypeError("Unexpected type {}.".format(type(obj)))
        inputs = new_inputs
        for name, typ in inputs:
            if typ is None:
                raise RuntimeError("Type input '{}' for operator '{}' "
                                   "is unknown. You should specify "
                                   "input types.".format(
                                       name, self.__class__.__name__))

        target_opset = get_opset_number_from_onnx()
        container = ModelComponentContainer(target_opset)
        if container.target_opset < 9:
            raise RuntimeError("The operator cannot be converted into ONNX."
                               " It requires ONNX op_set >= 9.")
        model_name = self.__class__.__name__
        scope = Scope(model_name, target_opset=target_opset,
                      variable_name_set=set(_[0] for _ in inputs))
        for inp in inputs:
            container.add_input(Variable(inp[0], inp[0],
                                         scope=scope, type=inp[1]))
        self.add_to(scope, container)

        # infer shapes
        if outputs:
            shapes = []
            for o in outputs:
                if isinstance(o, Variable):
                    shapes.append(o)
                elif isinstance(o, tuple):
                    shapes.append(Variable(o[0], o[0], None, o[1]))
                else:
                    raise TypeError("Outputs must be Variable or "
                                    "tuple(name, type).")
        else:
            shapes = infer_outputs(container, container.inputs)

            if self.output_names:
                shapes = [shape for shape in shapes
                          if shape.onnx_name in self.output_names]

        # add the output to the container
        for shape in shapes:
            container.add_output(shape)

        # convert the graph
        graph = helper.make_graph(
            container.nodes, model_name, container.inputs,
            container.outputs, container.initializers)
        onnx_model = helper.make_model(graph)

        # domains
        domains = {}
        version = get_opset_number_from_onnx()
        for n in container.nodes:
            domains[n.domain] = max(domains.get(n.domain, version),
                                    getattr(n, 'op_version', version))
        for i, (k, v) in enumerate(domains.items()):
            if i == 0 and len(onnx_model.opset_import) == 1:
                op_set = onnx_model.opset_import[0]
            else:
                op_set = onnx_model.opset_import.add()
            op_set.domain = k
            op_set.version = domains.get(k, version)

        # metadata
        onnx_model.ir_version = onnx_proto.IR_VERSION
        onnx_model.producer_name = utils.get_producer()
        onnx_model.producer_version = utils.get_producer_version()
        onnx_model.domain = utils.get_domain()
        onnx_model.model_version = utils.get_model_version()

        return onnx_model

    def enumerate_nodes(self):
        """
        Iterates on all nodes of the graph.
        """
        yield self
        for input in self.inputs:
            if isinstance(input, OnnxOperator):
                for i in input:
                    yield i

    def enumerate_variables(self):
        """
        Iterates on all nodes of the graph to find variables.
        Returns an iterator `(node, i)` which means
        `node.inputs[i]` is a variable.
        """
        for node in self.enumerate_nodes():
            if self.inputs:
                for i, input in enumerate(self.inputs):
                    if isinstance(input, (OnnxOperator.UnscopedVariable,
                                          Variable)):
                        yield (node, i)

    def enumerate_initial_types(self):
        """
        Retrieves iniatial types of the implemented functions.
        It goes through the graph and returns the name and types
        of all variables not computed by an intemediate node.

        :return: list of `(name, type)`
        """
        for node, i in self.enumerate_variables():
            input = node.inputes[i]
            if isinstance(input, Variable):
                yield (input.name, input.type)
            elif isinstance(input, OnnxOperator.UnscopedVariable):
                name = input.name
                typ = node.__class__.expected_inputs[i]
                yield (name, typ)
