# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from scipy.sparse import coo_matrix
from onnxconverter_common.onnx_ops import apply_identity
from ..proto import TensorProto
from ..common._topology import (
    Variable, Scope, _update_domain_version,
    _get_main_opset_version, OPSET_TO_IR_VERSION)
from ..common._container import ModelComponentContainer
from ..common import utils
from .._supported_operators import sklearn_operator_name_map
from ..proto import get_latest_tested_opset_version, onnx_proto
from ..proto.onnx_helper_modified import make_graph, make_model
from ..helpers.onnx_helper import infer_outputs
from .graph_state import GraphState
from .type_helper import _guess_type


class OnnxOperatorItem:
    """
    Accessor to one of the output returned by a *OnnxOperator*.

    :param onx_op: OnnxOperator
    :param index: integer
    """
    def __init__(self, onx_op, index):
        if not isinstance(index, int):
            raise TypeError("index must be an integer.")
        self.onx_op = onx_op
        self.index = index

    def __str__(self):
        """
        usual
        """
        return "%s[%d]" % (str(self.onx_op), self.index)

    def get_latest_tested_opset_version(self):
        """
        Returns ``get_latest_tested_opset_version()``
        of the wrapped *OnnxOperator* instance.
        """
        return self.onx_op.get_latest_tested_opset_version()

    def add_to(self, scope, container, operator=None):
        """
        Adds outputs to the container if not already added,
        registered the outputs if the node is not final.

        :param scope: scope
        :param container: container
        :param operator: overwrite inputs
        """
        self.onx_op.add_to(scope, container, operator=operator)

    def get_output(self, i=0):
        """
        Returns the output.
        """
        if i != 0:
            raise IndexError("Can only return the first item.")
        return self.onx_op.get_output(self.index)

    @property
    def outputs(self):
        """
        Returns the outputs of the node.
        """
        if self.onx_op is None:
            raise RuntimeError(
                "self.onx_op cannot be None, type(self)={}".format(
                    type(self)))
        if self.index is None:
            raise RuntimeError(
                "self.index cannot be None, type(self)={}".format(
                    type(self)))
        outputs = self.onx_op.outputs
        if outputs is None:
            raise RuntimeError(
                "self.onx_op.outputs cannot be None, "
                "type(self)={}, type(self.onx_op)={}, "
                "type(self.onx_op.state)={}".format(
                    type(self), type(self.onx_op), type(self.onx_op.state)))
        return outputs[self.index:self.index + 1]


class OnnxSubOperator:
    """
    Includes a sub operator in the ONNX graph.
    """

    def __init__(self, op, inputs, output_names=None, op_version=None,
                 options=None):
        self.op = op
        self.output_names = output_names
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.inputs = inputs
        self.op_version = op_version
        self.options = options

    def add_to(self, scope, container, operator=None):
        """
        Adds outputs to the container if not already added,
        registered the outputs if the node is not final.

        :param scope: scope
        :param container: container
        :param operator: overwrite inputs
        """
        if operator is not None:
            raise RuntimeError(
                "operator must be None, the operator to convert "
                "is specified in member 'op'.")
        try:
            op_type = sklearn_operator_name_map[type(self.op)]
        except KeyError:
            raise RuntimeError(
                "Unable to find a converter for model of type '{}'."
                "".format(self.op.__class__.__name__))

        this_operator = scope.declare_local_operator(op_type, self.op)
        this_operator.inputs = self.inputs
        if self.output_names is None:
            output = scope.declare_local_variable('sub_%s' % op_type)
            this_operator.outputs.append(output)
            self.outputs = [output]
        else:
            self.outputs = []
            for v in self.output_names:
                if isinstance(v, Variable):
                    output = scope.declare_local_variable(
                        '%s_%s' % (v.onnx_name, op_type))
                    apply_identity(
                        scope, output.onnx_name, v.onnx_name, container)
                elif isinstance(v, str):
                    output = scope.declare_local_variable(v)
            self.outputs.append(output)
            this_operator.outputs.extend(self.outputs)


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

        def __str__(self):
            """
            usual
            """
            return "Cst({})".format(self.value)

    def find_schema(self, op_version):
        """
        Checks if there is an existing schema for a
        specific version.

        :param op_version: requested version
        :return: schema
        """
        if not hasattr(self.__class__, 'past_version'):
            raise RuntimeError("Missing attribute 'past_version', there is "
                               "no other available schema.")
        found = None
        for v in self.past_version.values():
            if v.since_version > op_version:
                continue
            if found is None or v.since_version > found.since_version:
                found = v
        if found is None:
            raise RuntimeError(
                "Operator '{}': requested version {} < "
                "{} schema version.".format(
                    self.__class__.__name__,
                    op_version, self.since_version))
        return found

    def __init__(self, *inputs, op_version=None, output_names=None,
                 domain=None, **kwargs):

        if (output_names is None and
                self.__class__.__name__ in {"OnnxScan"}):
            raise NotImplementedError(
                "The class cannot infer the number of variables "
                "for node '{}' yet. output_names must be specified"
                ".".format(self.__class__.__name__))

        if op_version is None:
            if domain == '':
                self.op_version = get_latest_tested_opset_version()
            else:
                self.op_version = None
        else:
            self.op_version = op_version
        self.since_version = self.__class__.since_version

        if (self.op_version is not None and
                self.op_version < self.since_version):
            schema = self.find_schema(self.op_version)
            self.since_version = schema.since_version
            self.expected_inputs = schema.expected_inputs
            self.expected_outputs = schema.expected_outputs
            self.input_range = schema.input_range
            self.output_range = schema.output_range
        else:
            self.expected_inputs = self.__class__.expected_inputs
            self.expected_outputs = self.__class__.expected_outputs
            self.input_range = self.__class__.input_range
            self.output_range = self.__class__.output_range
            if self.__class__.__name__ not in {
                    'OnnxScan', 'OnnxLoop', 'OnnxIf'}:
                # TODO: the minimum opset depends on embedded graph
                # by default, it takes the given op_version but the
                # optimal value could be lower.
                self.op_version = self.since_version
            if self.op_version is None:
                self.op_version = self.since_version

        if (self.op_version is not None and
                self.op_version < self.since_version):
            raise RuntimeError(
                "Operator '{}': requested version {} < "
                "{} schema version.".format(
                    self.__class__.__name__,
                    self.op_version, self.since_version))

        self.state = None
        self.domain = domain
        self.kwargs = kwargs
        self.onnx_prefix_name = None

        # check inputs
        if len(inputs) == 0:
            if self.input_range[0] == self.input_range[1]:
                self.inputs = [_[0] for _ in self.expected_inputs]
            else:
                # The number of inputs may vary.
                self.inputs = None
        else:
            self.inputs = []
            for inp in inputs:
                if isinstance(inp, str):
                    self.inputs.append(OnnxOperator.UnscopedVariable(inp))
                elif isinstance(inp, (OnnxOperator, Variable,
                                      OnnxOperatorItem, OnnxSubOperator)):
                    self.inputs.append(inp)
                elif isinstance(inp, (np.ndarray, coo_matrix)):
                    self.inputs.append(
                        OnnxOperator.ConstantVariable(inp))
                elif isinstance(inp, TensorProto):
                    self.inputs.append(OnnxOperator.ConstantVariable(inp))
                elif isinstance(inp, (OnnxOperator.OnnxOperatorVariable,
                                      OnnxOperator.ConstantVariable)):
                    self.inputs.append(inp)
                elif isinstance(inp, (np.int64, np.float32,
                                      np.float64, np.bool,
                                      np.int8, np.uint8)):
                    self.inputs.append(inp)
                elif isinstance(inp, (float, )):
                    self.inputs.append(np.float64(inp))
                elif isinstance(inp, (int, )):
                    self.inputs.append(np.int64(inp))
                else:
                    raise TypeError(
                        "Unable to interpret the input name for type {} in "
                        "operator '{}' (value={}).".format(
                            type(inp), self.__class__.__name__, inp))

        if self.inputs is not None:
            if (len(self.inputs) < self.input_range[0] or
                    len(self.inputs) > self.input_range[1]):
                raise RuntimeError(
                    "Operator '{}' expects a number of inputs "
                    "in [{}, {}] not {} (expected opset={}, "
                    "class opset={})".format(
                        self.operator_name, *self.input_range,
                        len(self.inputs), op_version, self.op_version))

        # check output
        if (hasattr(output_names, 'outputs') and
                output_names.outputs is not None):
            self.output_names = [out.onnx_name
                                 for out in output_names.outputs]
            self.output_variables = output_names
        else:
            self.output_names = output_names
            self.output_variables = None

        if self.output_names:
            if self.output_variables is None:
                self.output_variables = [None for o in self.output_names]
            for i in range(len(self.output_names)):
                name = self.output_names[i]
                if isinstance(name, Variable):
                    self.output_names[i] = name.onnx_name
                    self.output_variables[i] = name
                elif not isinstance(name, str):
                    raise TypeError("output_names must be a list of strings "
                                    "and element {} is {}".format(
                                        i, type(name)))
            if all(map(lambda x: x is None, self.output_variables)):
                self.output_variables = None

    def __str__(self):
        """
        usual
        """
        return "{}({} in) -> {}".format(
            self.__class__.__name__,
            len(self.inputs) if self.inputs is not None else 0,
            [str(o) for o in self.output_names]
            if self.output_names is not None else "?")

    def set_onnx_name_prefix(self, onnx_prefix_name):
        """
        Provides a name to define a prefix in the onnx graph
        to avoid to get unreadable node names. The method
        does not overwrite an existing name, it propagates
        the prefix to inputs and stops the propagation
        if the prefix is already defined.
        """
        if self.onnx_prefix_name is None:
            self.onnx_prefix_name = onnx_prefix_name
            for inp in self.inputs:
                if hasattr(inp, 'onnx_prefix_name'):
                    inp.set_onnx_name_prefix(onnx_prefix_name)
        return self

    @property
    def onnx_prefix(self):
        if self.onnx_prefix_name is None:
            name = self.__class__.__name__
            if name.startswith("Onnx"):
                name = name[4:]
            return name[:2]
        return self.onnx_prefix_name

    def __getitem__(self, index):
        """
        Returns an accessor to one of the output
        of this node.
        """
        return OnnxOperatorItem(self, index)

    def get_output(self, i):
        """
        Returns the ith output.
        """
        if hasattr(self, 'output_names_'):
            return self.output_names_[i]
        if (self.output_names and i < len(self.output_names) and
                self.output_names[i]):
            return self.output_names[i]
        if i < len(self.expected_outputs):
            return self.expected_outputs[i][0]
        if i < self.output_range[1]:
            if i > 1000:
                raise IndexError(
                    "Too many outputs. You should redesign your operator.")
            return "O%d" % i
        raise IndexError("Output %d does not exist." % i)

    def update_name(self, i, name):
        """
        Updates the name of a variable after it was scoped.
        """
        if (self.output_variables is not None and
                i < len(self.output_variables)):
            raise RuntimeError(
                "Inconsistent, cannot changed variable name "
                "after it was used: '{}' != '{}'".format(
                    self.output_variables[i], name))
        if (hasattr(self, 'output_names_') and
                i < len(self.output_names_) and
                self.output_names_[i] != name):
            raise RuntimeError(
                "Inconsistent, cannot changed variable name "
                "after it was used: '{}' != '{}'".format(
                    self.output_names_[i], name))
        if self.output_names is None:
            self.output_names = []
        while len(self.output_names) <= i:
            self.output_names.append(None)
        self.output_names[i] = name

    def _set_output_names_(self, scope, operator):
        if hasattr(self, 'output_names_'):
            outputs = self.output_names_
        elif self.output_variables is not None:
            outputs = [o.onnx_name for o in self.output_variables]
            self.output_names_ = outputs
        elif self.output_names:
            if not isinstance(self.output_names, (list, tuple)):
                louts = [self.output_names]
            else:
                louts = self.output_names
            if operator is not None and len(louts) != len(operator.outputs):
                raise RuntimeError(
                    "Output mismatch for '{}'\n{}\n{}".format(
                        type(operator.raw_operator),
                        louts, operator.outputs))
            outputs = []
            for iname, name in enumerate(louts):
                if name is None:
                    raise AssertionError(
                        "Issue for operator '{}'.".format(
                            type(operator.raw_operator)))
                if name.startswith('u(') and name[-1] == ')':
                    name = scope.get_unique_variable_name(name[2:-1])
                elif operator is not None:
                    oout = operator.outputs[iname]
                    name = oout.onnx_name
                outputs.append(name)
            self.output_names_ = outputs
        else:
            outputs = []
            for name in self.expected_outputs:
                name = scope.get_unique_variable_name(
                    self.onnx_prefix + "_" + name[0])
                outputs.append(name)
            self.output_names_ = outputs
        return outputs

    def _add_to_inputs(self, operator):
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
                    if i.onnx_name == input.name:
                        inputs.append(i)
                        break
                else:
                    vars = ', '.join(map(lambda o: "'%s'" % o.onnx_name,
                                         operator.inputs))
                    raise RuntimeError("Unable to find variable "
                                       "{} in {}.".format(input, vars))
            else:
                inputs.append(input)
        return inputs

    def add_to(self, scope, container, operator=None):
        """
        Adds outputs to the container if not already added,
        registered the outputs if the node is not final.

        :param scope: scope
        :param container: container
        :param operator: overwrite inputs
        """
        if self.state is None:
            if self.is_deprecated:
                raise RuntimeError(
                    "Node '{}' is deprecated. This API cannot deprecated "
                    "nodes.".format(self.__class__.__name__))
            if (self.op_version is not None and
                    self.op_version < self.since_version):
                raise RuntimeError(
                    "Incompatible versions for node '{}'  op_version {} "
                    "< since_version {}.".format(
                        self.__class__.__name__, self.op_version,
                        self.since_version))
            if self.kwargs.get('op_version', '') is None:
                kwargs = self.kwargs.copy()
                del kwargs['op_version']
            else:
                kwargs = self.kwargs

            self._set_output_names_(scope, operator)
            domain = self.domain
            if domain is None:
                domain = self.__class__.domain
            inputs = self._add_to_inputs(operator)

            self.state = GraphState(
                inputs, self.output_names_, self.operator_name,
                scope, container, None, op_version=self.op_version,
                op_domain=domain, onnx_prefix_name=self.onnx_prefix,
                **kwargs)
            self.state.run(operator=operator)
        self._verify_add_to_()

    def _verify_add_to_(self):
        if self.state is None:
            raise RuntimeError(
                "Graph was not produced for operator '{}': {}."
                "".format(self.__class__.__name__, self))
        for i in self.inputs:
            if hasattr(i, '_verify_add_to_'):
                i._verify_add_to_()

    @property
    def outputs(self):
        """
        Returns the outputs of the node.
        """
        if self.state is None:
            raise RuntimeError("Method add_to was not called.")
        return self.state.outputs

    def _clean_attributes(self, *args, recursive=True):
        """
        Removes attributes in this node and its parents.
        """
        for arg in args:
            if arg == 'state':
                self.state = None
            elif hasattr(self, arg):
                delattr(self, arg)
        if recursive:
            for obj in self.inputs:
                if isinstance(obj, OnnxOperator):
                    obj._clean_attributes(*args, recursive=True)

    def to_onnx(self, inputs=None, outputs=None, other_outputs=None,
                target_opset=None, domain=None):
        """
        Converts this operator into an ONNX graph.

        :param inputs: specific inputs (as a dictionary) or
            default inputs if not specified
        :param outputs: specific outputs
        :param other_outputs: additional outputs to consider
            as graph outputs but not outputs of this particular
            node
        :param target_opset: dictionary with target opset per domain,
            None for the default one
        :param domain: domain of the operator
        """
        if isinstance(target_opset, dict):
            dom = self.domain or ''
            target_opset = target_opset.get(dom, None)
        elif isinstance(target_opset, int):
            if self.domain not in ('', None):
                # The target_opset is for the domain ''
                # We ignore it.
                target_opset = None
        elif target_opset is not None:
            raise TypeError(
                "target_opset must be a dictionary {domain: "
                "target_opset} not %r for operator %r." % (
                    target_opset, self.__class__.__name__))
        if self.domain in ('', None) and target_opset == 1:
            raise RuntimeError("target_opset cannot be 1.")
        if (self.op_version is not None and target_opset is not None and
                self.op_version > target_opset):
            raise RuntimeError(
                "target_opset={} is lower than the version={} requested "
                "for this node '{}'.".format(
                    target_opset, self.op_version, self.__class__.__name__))
        if hasattr(self, "state"):
            # The conversion already happened and needs to be cleaned.
            self._clean_attributes("output_names_", "state")
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
                raise TypeError("Inputs must be Variable or "
                                "tuple(name, type) not {}."
                                "".format(type(obj)))
        inputs = new_inputs
        for name, typ in inputs:
            if typ is None:
                raise RuntimeError("Type input '{}' for operator '{}' "
                                   "is unknown. You should specify "
                                   "input types.".format(
                                       name, self.__class__.__name__))
        target_opset = self.get_latest_tested_opset_version(target_opset)
        container = ModelComponentContainer(target_opset)

        model_name = self.__class__.__name__
        scope = Scope(model_name, target_opset=target_opset,
                      variable_name_set=set(_[0] for _ in inputs))
        for inp in inputs:
            container.add_input(Variable(inp[0], inp[0],
                                         scope=scope, type=inp[1]))
        self.add_to(scope, container)
        if other_outputs is not None:
            for out in other_outputs:
                if not hasattr(out, 'add_to'):
                    raise RuntimeError(
                        "Extra outputs must have method 'add_to'.")
                out.add_to(scope, container)

        # infer shapes
        if outputs:
            if isinstance(outputs, dict):
                outputs = [(k, v) for k, v in outputs.items()]
            shapes = []
            for o in outputs:
                if isinstance(o, Variable):
                    shapes.append(o)
                elif isinstance(o, tuple):
                    if isinstance(o[1], np.ndarray):
                        type_shape = _guess_type(o[1])
                    else:
                        type_shape = o[1]
                    shapes.append(Variable(o[0], o[0], None, type_shape))
                else:
                    raise TypeError("Outputs must be Variable or "
                                    "tuple(name, type).")
        else:
            shapes = infer_outputs(container, container.inputs,
                                   initializer=container.initializers,
                                   target_opset=target_opset)

            if self.output_names:
                shapes = [shape for shape in shapes
                          if shape.onnx_name in self.output_names]

        # add the output to the container
        for shape in shapes:
            container.add_output(shape)

        # convert the graph
        graph = make_graph(
            container.nodes, model_name, container.inputs,
            container.outputs, container.initializers)
        onnx_model = make_model(graph)

        # domains
        _update_domain_version(container, onnx_model)

        # metadata
        opv = min(target_opset,
                  _get_main_opset_version(onnx_model) or target_opset)
        irv = OPSET_TO_IR_VERSION.get(opv, onnx_proto.IR_VERSION)
        onnx_model.ir_version = irv
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
                for i in input.enumerate_nodes():
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
            input = node.inputs[i]
            if isinstance(input, Variable):
                yield (input.onnx_name, input.type)
            elif isinstance(input, OnnxOperator.UnscopedVariable):
                name = input.name
                typ = node.expected_inputs[i]
                yield (name, typ)

    def get_latest_tested_opset_version(self, target_opset=None):
        """
        Returns *op_version*, or the max of all results
        returned by these method applied on every input,
        or ``get_latest_tested_opset_version()``.
        """
        if target_opset is not None:
            return target_opset
        return get_latest_tested_opset_version()


class OnnxSubEstimator(OnnxOperator):
    """
    This operator is used to call the converter of a model
    while converting another one.
    See :ref:`l-custom-parser-alternative`.
    """

    since_version = 1
    expected_inputs = None
    expected_outputs = None
    input_range = [1, 1e9]
    output_range = [1, 1e9]

    def __init__(self, skl_op, *inputs, op_version=None,
                 output_names=None,
                 domain=None, **kwargs):
        OnnxOperator.__init__(
            self, *inputs, op_version=op_version,
            output_names=output_names, domain=domain, **kwargs)
        self.operator_instance = skl_op

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
                pass
            elif self.output_names:
                if not isinstance(self.output_names, (list, tuple)):
                    louts = [self.output_names]
                else:
                    louts = self.output_names
                outputs = []
                for name in louts:
                    if name.startswith('u(') and name[-1] == ')':
                        name = scope.get_unique_variable_name(name[2:-1])
                    outputs.append(name)
                self.output_names_ = outputs
            else:
                self.output_names_ = None

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
                        if i.onnx_name == input.name:
                            inputs.append(i)
                            break
                    else:
                        vars = ', '.join(map(lambda o: "'%s'" % o.onnx_name,
                                             operator.inputs))
                        raise RuntimeError("Unable to find variable "
                                           "{} in {}.".format(input, vars))
                else:
                    inputs.append(input)
            self.state = GraphState(
                inputs, self.output_names_, self.operator_instance,
                scope, container, None, op_version=self.op_version,
                op_domain=None, onnx_prefix_name=self.onnx_prefix,
                **kwargs)
            self.state.run(operator=operator)

    @property
    def outputs(self):
        """
        Returns the outputs of the node.
        """
        if self.state is None:
            raise RuntimeError("Method add_to was not called.")
        return self.state.outputs
