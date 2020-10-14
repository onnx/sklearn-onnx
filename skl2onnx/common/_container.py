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
from scipy.sparse import coo_matrix
from onnx import onnx_pb as onnx_proto
from onnx.defs import onnx_opset_version, get_all_schemas_with_history
import onnx.onnx_cpp2py_export.defs as C
from onnxconverter_common.onnx_ops import __dict__ as dict_apply_operation
from ..proto import TensorProto
from ..proto.onnx_helper_modified import (
    make_node, ValueInfoProto, make_tensor, make_attribute
)
try:
    from ..proto import SparseTensorProto
    from ..proto.onnx_helper_modified import make_sparse_tensor
except ImportError:
    # onnx is too old.
    SparseTensorProto = None
    make_sparse_tensor = None
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
                if 'squeeze' in k:
                    # implementation of apply_squeeze, apply_unsqueeze
                    # does not follow the same schema
                    continue
                if k in {'apply_less_or_equal', 'apply_greater_or_equal'}:
                    continue
                warnings.warn("Unable to find an ONNX name in function "
                              "'{0}', source=\n{1}".format(k, source))
            res[found] = v
    return res


def _build_options(model, defined_options, default_values, allowed_options):
    opts = {} if default_values is None else default_values
    if defined_options is not None:
        opts.update(defined_options.get(type(model), {}))
        opts.update(defined_options.get(id(model), {}))
    if allowed_options not in (None, 'passthrough'):
        for k, v in opts.items():
            if k not in allowed_options:
                raise NameError(
                    "Option '{}' not in {} for class '{}'.".format(
                        k, list(sorted(allowed_options)),
                        model.__class__.__name__))
            allowed = allowed_options[k]
            if allowed is not None and v not in allowed and v is not None:
                raise ValueError(
                    "Unexpected value [{!r}] for option '{}'"
                    " (it must be in {}) for model '{}'.".format(
                        v, k, allowed, model.__class__.__name__))
    elif len(opts) != 0 and allowed_options != 'passthrough':
        raise RuntimeError(
            "Options {} are not registerd for model '{}'.".format(
                list(sorted(opts)), model.__class__.__name__))
    return opts


_apply_operation_specific = _get_operation_list()


class _WhiteBlackContainer:

    def __init__(self, white_op=None, black_op=None):
        self._white_op = white_op
        self._black_op = black_op

    def is_allowed(self, node_type):
        """
        Tells if a node is white listed or not black listed.
        """
        if isinstance(node_type, (list, tuple, set)):
            return all(map(self.is_allowed, node_type))
        try:
            self.check_white_black_list(node_type)
            return True
        except RuntimeError:
            return False

    def check_white_black_list(self, node_type):
        """
        Checks a node type is allowed according to white
        and black lists.
        """
        if self._white_op:
            if node_type not in self._white_op:
                raise RuntimeError(
                    "Operator '{}' is not white listed.".format(node_type))
        if self._black_op:
            if node_type in self._black_op:
                raise RuntimeError(
                    "Operator '{}' is black listed.".format(node_type))


class RawModelContainerNode(_WhiteBlackContainer):
    """
    This node is the carrier of the model we want to convert.
    It provides an abstract layer so that our parsing
    framework can work with models generated by different tools.
    """

    def __init__(self, raw_model, dtype, white_op=None, black_op=None):
        """
        :param raw_model: *scikit-learn* model to convert
        """
        _WhiteBlackContainer.__init__(
            self, white_op=white_op, black_op=black_op)
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

    def __init__(self, sklearn_model, dtype,
                 white_op=None, black_op=None):
        super(SklearnModelContainerNode, self).__init__(
              sklearn_model, dtype, white_op=white_op, black_op=black_op)
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


class ModelComponentContainer(ModelContainer, _WhiteBlackContainer):
    """
    In the conversion phase, this class is used to collect all materials
    required to build an *ONNX* *GraphProto*, which is encapsulated in a
    *ONNX* *ModelProto*.
    """

    def __init__(self, target_opset, options=None, dtype=None,
                 registered_models=None,
                 white_op=None, black_op=None):
        """
        :param target_opset: number, for example, 7 for *ONNX 1.2*, and
                             8 for *ONNX 1.3*.
        :param dtype: float type to be used for every float coefficient
        :param options: see :ref:`l-conv-options`
        :param registered_models: registered models
        :param white_op: white list of ONNX nodes allowed
            while converting a pipeline, if empty, all are allowed
        :param black_op: black list of ONNX nodes allowed
            while converting a pipeline, if empty, none are blacklisted
        """
        if dtype is None:
            raise ValueError("dtype must be specified, it should be either "
                             "np.float32 or np.float64.")
        _WhiteBlackContainer.__init__(
            self, white_op=white_op, black_op=black_op)
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
        # All registered models.
        self.registered_models = registered_models

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

    def add_options(self, model_id, options):
        """
        Adds an option, for example,
        ``add_options(id(clr), {'raw_scores': True})``
        tells the converter associated to ``clr`` to
        use raw score instead of probabilities.

        :param model_id: class or ``id(instance)``
        :param options: dictionary with the new values
        """
        if options is None:
            return
        if self.options is None:
            self.options = {}
        if model_id not in self.options:
            self.options[model_id] = None
        if self.options[model_id] is None:
            self.options[model_id] = {}
        self.options[model_id].update(options)

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
        if (can_cast and isinstance(content, (np.ndarray, coo_matrix)) and
                onnx_type in (TensorProto.FLOAT, TensorProto.DOUBLE) and
                onnx_type != self.proto_dtype):
            content = content.astype(self.dtype)
            onnx_type = self.proto_dtype

        sparse_tensor = None
        tensor = None

        if isinstance(content, TensorProto):
            tensor = TensorProto()
            tensor.data_type = content.data_type
            tensor.name = name
            tensor.raw_data = content.raw_data
            tensor.dims.extend(content.dims)
        elif shape is None and isinstance(
                content, (np.float32, np.float64, np.int32, np.int64, float)):
            tensor = make_tensor(name, onnx_type, [], [content])
        elif (SparseTensorProto is not None and
                isinstance(content, SparseTensorProto)):
            raise NotImplementedError("Not implemented yet.")
        elif shape is None:
            tensor = make_attribute(name, content)
        elif isinstance(content, coo_matrix):
            if SparseTensorProto is None:
                raise RuntimeError(
                    "Sparse matrices require SparseTensorProto. Update onnx.")
            values_tensor = make_tensor(
                name + "_v", data_type=onnx_type,
                dims=(len(content.data), ), vals=content.data)
            indices = [i * content.shape[1] + j
                       for i, j in zip(content.row, content.col)]
            indices_tensor = make_tensor(
                name=name + "_i", data_type=TensorProto.INT64,
                dims=(len(indices), ), vals=indices)
            dense_shape = list(content.shape)
            sparse_tensor = make_sparse_tensor(
                values_tensor, indices_tensor, dense_shape)
        else:
            if any(d is None for d in shape):
                raise ValueError('Shape of initializer cannot contain None.')
            tensor = make_tensor(name, onnx_type, shape, content)

        if tensor is not None:
            self.initializers.append(tensor)
            return tensor
        elif sparse_tensor is not None:
            self.add_node('Constant', [], [name], sparse_value=sparse_tensor,
                          op_version=self.target_opset, name=name + '_op')
            return sparse_tensor
        else:
            raise RuntimeError(
                "Either tensor or sparse_tensor should be defined.")

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
        self.check_white_black_list(op_type)

    def add_node(self, op_type, inputs, outputs, op_domain='', op_version=None,
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
        if op_version is None:
            op_version = self._get_op_version(op_domain, op_type)

        if isinstance(inputs, (six.string_types, six.text_type)):
            inputs = [inputs]
        if isinstance(outputs, (six.string_types, six.text_type)):
            outputs = [outputs]
        common = set(inputs) & set(outputs)
        if common:
            raise RuntimeError("inputs and outputs cannot have "
                               "variables in common {}".format(common))
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
        if 'dtype' in attrs:
            raise RuntimeError("dtype should not be a parameter.")
        try:
            node = make_node(op_type, inputs, outputs, name=name,
                             _dtype=self.dtype, **attrs)
        except ValueError as e:
            raise ValueError("Unable to create node '{}' with name='{}'."
                             "".format(op_type, name)) from e
        node.domain = op_domain

        self.node_domain_version_pair_sets.add((op_domain, op_version))
        self.nodes.append(node)
        if (self.target_opset is not None and
                op_version is not None and
                op_version > self.target_opset_any_domain(op_domain)):
            raise RuntimeError(
                "Opset number {} is higher than targeted opset {} for "
                "node '{}' (domain: '{}').".format(
                    op_version, self.target_opset, node.op_type, op_domain))

    def target_opset_any_domain(self, domain):
        if isinstance(self.target_opset, dict):
            if domain in self.target_opset:
                to = self.target_opset[domain]
            else:
                to = None
            if to is None and domain == '':
                to = onnx_opset_version()
            if to is None:
                smap = C.schema_version_map()
                if domain in smap:
                    to = smap[domain][1]
            if to is not None:
                return to
            # The domain is not registered in onnx, it is probably
            # a custom domain. We assume the version is one.
            return 1
        return self.target_opset

    @property
    def target_opset_onnx(self):
        return self.target_opset_any_domain('')

    def _get_op_version(self, domain, op_type):
        """
        Determines the highest version of operator
        *op_type* below or equal to *target_opset*.
        """
        if not hasattr(self, '_op_versions'):
            self._build_op_version()
        key = domain, op_type
        vers = self._op_versions.get(key, None)
        if vers is None:
            warnings.warn(
                "Unable to find operator '{}' in domain '{}' in ONNX, "
                "op_version is forced to 1.".format(
                    op_type, domain))
            vers = [1]
        highest = self.target_opset_any_domain(domain)
        pos = len(vers) - 1
        while pos >= 0:
            if vers[pos] <= highest:
                return vers[pos]
            pos -= 1
        raise RuntimeError(
            "Unable to find a suitable version for operator '{}' "
            "in domain '{}'. Available versions: {}.".format(
                op_type, domain, vers))

    def _build_op_version(self):
        res = {}
        for schema in get_all_schemas_with_history():
            dom = schema.domain
            name = schema.name
            vers = schema.since_version
            if (dom, name) not in res:
                res[dom, name] = set()
            res[dom, name].add(vers)
        self._op_versions = {}
        for k, v in res.items():
            self._op_versions[k] = list(sorted(v))

    def _get_allowed_options(self, model):
        if self.registered_models is not None:
            if inspect.isfunction(model):
                if model not in self.registered_models['aliases']:
                    return None
                alias = self.registered_models['aliases'][model]
            else:
                if type(model) not in self.registered_models['aliases']:
                    return {}
                alias = self.registered_models['aliases'][type(model)]
            conv = self.registered_models['conv'][alias]
            allowed = conv.get_allowed_options()
            if allowed is None:
                return {}
            return allowed
        clname = (str(model) if inspect.isfunction(model)
                  else model.__class__.__name__)
        raise NotImplementedError(
            "No registered models, no known allowed options "
            "for model '{}'.".format(clname))

    def validate_options(self, operator):
        """
        Validates every operator allows the options
        given by the user at converter time
        for an operator.
        """
        skl_op = operator.raw_operator
        self.get_options(skl_op)

    def get_options(self, model, default_values=None):
        """
        Returns additional options for a model.
        It first looks by class then by id (``id(model)``).
        :param model: model being converted
        :param default_values: default options (it is modified by
                               the function)
        :return: dictionary
        """
        return _build_options(
            model, self.options, default_values,
            self._get_allowed_options(model))

    def has_options(self, model, option_name):
        """
        Tells if a model allows one specific options.

        :param model: model being converted
        :return: boolean
        """
        opts = self._get_allowed_options(model)
        return option_name in opts
