# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import inspect
import re
import six
import sys
import traceback
import warnings
import numpy as np
from onnx import onnx_pb as onnx_proto
from onnxconverter_common.onnx_ops import __dict__ as dict_apply_operation
from ..proto import TensorProto
from ..proto.onnx_helper_modified import (
    make_node, ValueInfoProto, make_tensor, make_attribute
)
from .interface import ModelContainer
from .utils import get_domain


def _get_operation_list():
    """
    Investigates this module to extract all ONNX functions
    which needs to be converted with these functions.
    """
    regs = [re.compile("container.add_node[(]'([A-Z][a-zA-Z0-9]*)', "
                       "\\[?input_name"),
            re.compile("container.add_node[(]'([A-Z][a-zA-Z0-9]*)', "
                       "\\[\\]"),
            re.compile("container.add_node[(]'([A-Z][a-zA-Z0-9]*)', "
                       "inputs"),
            re.compile("scope, '([A-Z][a-zA-Z0-9]*)', \\[?input_name"),
            re.compile("op_type = '([A-Z][a-zA-Z0-9]*)'")]
    res = {}
    for k, v in dict_apply_operation.items():
        if k.startswith("apply_") and callable(v):
            found = None
            source = inspect.getsource(v)
            for reg in regs:
                g = reg.search(source)
                if g:
                    found = g.groups()[0]
                    break
            if found is None:
                warnings.warn("Unable to find an ONNX name in function "
                              "'{0}', source=\n{1}".format(k, source))
            res[found] = v
    return res


def _build_options(model, defined_options, default_values):
    opts = {} if default_values is None else default_values
    if defined_options is not None:
        opts.update(defined_options.get(type(model), {}))
        opts.update(defined_options.get(id(model), {}))
    return opts


_apply_operation_specific = _get_operation_list()


class RawModelContainerNode(object):
    """
    This node is the carrier of the model we want to convert.
    It provides an abstract layer so that our parsing
    framework can work with models generated by different tools.
    """

    def __init__(self, raw_model, dtype):
        """
        :param raw_model: *scikit-learn* model to convert
        """
        self._raw_model = raw_model
        self.dtype = dtype
        if dtype == np.float32:
            self.proto_dtype = onnx_proto.TensorProto.FLOAT
        elif dtype == np.float64:
            self.proto_dtype = onnx_proto.TensorProto.DOUBLE
        elif dtype == np.int64:
            self.proto_dtype = onnx_proto.TensorProto.INT64
        else:
            raise ValueError("dtype should be either np.float32, "
                             "np.float64, np.int64.")

    @property
    def raw_model(self):
        return self._raw_model

    @property
    def input_names(self):
        """
        This function should return a list of strings. Each string
        corresponds to an input variable name.
        :return: a list of string
        """
        raise NotImplementedError()

    @property
    def output_names(self):
        """
        This function should return a list of strings. Each string
        corresponds to an output variable name.
        :return: a list of string
        """
        raise NotImplementedError()


class SklearnModelContainerNode(RawModelContainerNode):
    """
    Main container for one *scikit-learn* model.
    Every converter adds nodes to an existing container
    which is converted into a *ONNX* graph by an instance of
    :class:`Topology <skl2onnx.common._topology.Topology>`.
    """

    def __init__(self, sklearn_model, dtype):
        super(SklearnModelContainerNode, self).__init__(sklearn_model, dtype)
        # Scikit-learn models have no input and output specified,
        # so we create them and store them in this container.
        self._inputs = []
        self._outputs = []

    @property
    def input_names(self):
        return [variable.raw_name for variable in self._inputs]

    @property
    def output_names(self):
        return [variable.raw_name for variable in self._outputs]

    def add_input(self, variable):
        # The order of adding variables matters. The final model's
        # input names are sequentially added as this list
        if variable not in self._inputs:
            self._inputs.append(variable)

    def add_output(self, variable):
        # The order of adding variables matters. The final model's
        # output names are sequentially added as this list
        if variable not in self._outputs:
            self._outputs.append(variable)


class ModelComponentContainer(ModelContainer):
    """
    In the conversion phase, this class is used to collect all materials
    required to build an *ONNX* *GraphProto*, which is encapsulated in a
    *ONNX* *ModelProto*.
    """

    def __init__(self, target_opset, options=None, dtype=None):
        """
        :param target_opset: number, for example, 7 for *ONNX 1.2*, and
                             8 for *ONNX 1.3*.
        :param dtype: float type to be used for every float coefficient
        :param options: see :ref:`l-conv-options`
        """
        if dtype is None:
            raise ValueError("dtype must be specified, it should be either "
                             "np.float32 or np.float64.")
        # Inputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.inputs = []
        # Outputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.outputs = []
        # ONNX tensors (type: TensorProto). They are initializers of
        # ONNX GraphProto.
        self.initializers = []
        # Intermediate variables in ONNX computational graph. They are
        # ValueInfoProto in ONNX.
        self.value_info = []
        # ONNX nodes (type: NodeProto) used to define computation
        # structure
        self.nodes = []
        # ONNX operators' domain-version pair set. They will be added
        # into opset_import field in the final ONNX model.
        self.node_domain_version_pair_sets = set()
        # The targeted ONNX operator set (referred to as opset) that
        # matches the ONNX version.
        self.target_opset = target_opset
        # Additional options given to converters.
        self.options = options

        self.dtype = dtype
        if dtype == np.float32:
            self.proto_dtype = onnx_proto.TensorProto.FLOAT
        elif dtype == np.float64:
            self.proto_dtype = onnx_proto.TensorProto.DOUBLE
        elif dtype == np.int64:
            self.proto_dtype = onnx_proto.TensorProto.INT64
        else:
            raise ValueError("dtype should be either np.float32, "
                             "np.float64, np.int64.")

    def __str__(self):
        """
        Shows internal information.
        """
        rows = []
        if self.inputs:
            rows.append("INPUTS")
            for inp in self.inputs:
                rows.append(
                    "  " + str(inp).replace(" ", "").replace("\n", " "))
        if self.outputs:
            rows.append("OUTPUTS")
            for out in self.outputs:
                rows.append(
                    "  " + str(out).replace(" ", "").replace("\n", " "))
        if self.initializers:
            rows.append("INITIALIZERS")
            for ini in self.initializers:
                rows.append(
                    "  " + str(ini).replace(" ", "").replace("\n", " "))
        if self.value_info:
            rows.append("NODES")
            for val in self.value_info:
                rows.append(
                    "  " + str(val).replace(" ", "").replace("\n", " "))
        if self.nodes:
            rows.append("PROTO")
            for nod in self.nodes:
                rows.append(
                    "  " + str(nod).replace(" ", "").replace("\n", " "))
        return "\n".join(rows)

    def _make_value_info(self, variable):
        value_info = ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        if variable.type.doc_string:
            value_info.doc_string = variable.type.doc_string
        return value_info

    def add_input(self, variable):
        """
        Adds our *Variable* object defined _parser.py into the the input
        list of the final ONNX model.

        :param variable: The Variable object to be added
        """
        self.inputs.append(self._make_value_info(variable))

    def add_output(self, variable):
        """
        Adds our *Variable* object defined *_parser.py* into the the
        output list of the final ONNX model.

        :param variable: The Variable object to be added
        """
        self.outputs.append(self._make_value_info(variable))

    def add_initializer(self, name, onnx_type, shape, content, can_cast=True):
        """
        Adds a *TensorProto* into the initializer list of the final
        ONNX model.

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g.,
                          TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list
                        or a float array).
        :param can_cast: the method can take the responsability
            to cast the constant
        :return: created tensor
        """
        if (can_cast and isinstance(content, np.ndarray) and
                onnx_type in (TensorProto.FLOAT, TensorProto.DOUBLE) and
                onnx_type != self.proto_dtype):
            content = content.astype(self.dtype)
            onnx_type = self.proto_dtype

        if isinstance(content, TensorProto):
            tensor = TensorProto()
            tensor.data_type = content.data_type
            tensor.name = name
            tensor.raw_data = content.raw_data
            tensor.dims.extend(content.dims)
        elif shape is None:
            tensor = make_attribute(name, content)
        else:
            if any(d is None for d in shape):
                raise ValueError('Shape of initializer cannot contain None')
            tensor = make_tensor(name, onnx_type, shape, content)
        self.initializers.append(tensor)
        return tensor

    def add_value_info(self, variable):
        self.value_info.append(self._make_value_info(variable))

    def _check_operator(self, op_type):
        """
        Checks that if *op_type* is one of the operators defined in
        :mod:`skl2onnx.common._apply_container`, then it was called
        from a function defined in this submodule by looking
        into the callstack. The test is enabled for *python >= 3.6*.
        """
        if (op_type in _apply_operation_specific and
                sys.version_info[:2] >= (3, 6)):
            tb = traceback.extract_stack()
            operation = []
            fct = _apply_operation_specific[op_type]
            skl2 = False
            for b in tb:
                if "_apply_operation" in b.filename and b.name == fct.__name__:
                    operation.append(b)
                    if not skl2 and "skl2onnx" in b.filename:
                        skl2 = True
            if skl2 and len(operation) == 0:
                raise RuntimeError(
                    "Operator '{0}' should be added with function "
                    "'{1}' in submodule _apply_operation.".format(
                        op_type, fct.__name__))

    def add_node(self, op_type, inputs, outputs, op_domain='', op_version=1,
                 name=None, **attrs):
        """
        Adds a *NodeProto* into the node list of the final ONNX model.
        If the input operator's domain-version information cannot be
        found in our domain-version pool (a Python set), we may add it.

        :param op_type: A string (e.g., Pool and Conv) indicating the
                        type of the NodeProto
        :param inputs: A list of strings. They are the input variables'
                       names of the considered NodeProto
        :param outputs: A list of strings. They are the output
                        variables' names of the considered NodeProto
        :param op_domain: The domain name (e.g., ai.onnx.ml) of the
                          operator we are trying to add.
        :param op_version: The version number (e.g., 0 and 1) of the
                           operator we are trying to add.
        :param name: name of the node, this name cannot be empty
        :param attrs: A Python dictionary. Keys and values are
                      attributes' names and attributes' values,
                      respectively.
        """
        if name is None or not isinstance(
                name, str) or name == '':
            name = "N%d" % len(self.nodes)
        existing_names = set(n.name for n in self.nodes)
        if name in existing_names:
            name += "-N%d" % len(self.nodes)

        if op_domain is None:
            op_domain = get_domain()
        self._check_operator(op_type)

        if isinstance(inputs, (six.string_types, six.text_type)):
            inputs = [inputs]
        if isinstance(outputs, (six.string_types, six.text_type)):
            outputs = [outputs]
        if not isinstance(inputs, list) or not all(
                isinstance(s, (six.string_types, six.text_type))
                for s in inputs):
            type_list = ','.join(list(str(type(s)) for s in inputs))
            raise ValueError('Inputs must be a list of string but get [%s]'
                             % type_list)
        if (not isinstance(outputs, list) or
                not all(isinstance(s, (six.string_types, six.text_type))
                        for s in outputs)):
            type_list = ','.join(list(str(type(s)) for s in outputs))
            raise ValueError('Outputs must be a list of string but get [%s]'
                             % type_list)
        upd = {}
        for k, v in attrs.items():
            if v is None:
                raise ValueError('Failed to create ONNX node. Undefined '
                                 'attribute pair (%s, %s) found' % (k, v))
            if (isinstance(v, np.ndarray) and
                    v.dtype in (np.float32, np.float64) and
                    v.dtype != self.dtype):
                upd[k] = v.astype(self.dtype)

        if upd:
            attrs.update(upd)
        try:
            node = make_node(op_type, inputs, outputs, name=name, **attrs)
        except ValueError as e:
            raise ValueError("Unable to create node '{}' with name='{}'."
                             "".format(op_type, name)) from e
        node.domain = op_domain

        self.node_domain_version_pair_sets.add((op_domain, op_version))
        self.nodes.append(node)

    def get_options(self, model, default_values=None):
        """
        Returns additional options for a model.
        It first looks by class then by id (``id(model)``).
        :param model: model being converted
        :param default_values: default options (it is modified by
                               the function)
        :return: dictionary
        """
        return _build_options(model, self.options, default_values)
