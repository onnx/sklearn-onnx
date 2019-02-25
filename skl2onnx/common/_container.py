# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import inspect
import re
import traceback
import six
import sys
from ..proto import helper
from .interface import ModelContainer
from ._apply_operation import __dict__ as dict_apply_operation
from .utils import get_domain


def _get_operation_list():
    """
    Investigates this module to extract all ONNX functions
    which needs to be converted with these functions.
    """
    regs = [re.compile("container.add_node[(]'([A-Z][a-zA-Z0-9]*)', \\[?input_name"),
            re.compile("scope, '([A-Z][a-zA-Z0-9]*)', \\[?input_name")]
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
                raise RuntimeError("Unable to find an ONNX name in function '{0}', source=\n{1}".format(
                    k, source))
            res[found] = v
    return res


_apply_operation_specific = _get_operation_list()


class RawModelContainerNode(object):
    '''
    This node is the carrier of the model we want to convert.
    It provides an abstract layer so that our parsing
    framework can work with models generated by different tools.
    '''

    def __init__(self, raw_model):
        """
        :param raw_model: *scikit-learn* model to convert
        """
        self._raw_model = raw_model

    @property
    def raw_model(self):
        return self._raw_model

    @property
    def input_names(self):
        '''
        This function should return a list of strings. Each string corresponds to an input variable name.
        :return: a list of string
        '''
        raise NotImplementedError()

    @property
    def output_names(self):
        '''
        This function should return a list of strings. Each string corresponds to an output variable name.
        :return: a list of string
        '''
        raise NotImplementedError()


class SklearnModelContainerNode(RawModelContainerNode):
    """
    Main container for one *scikit-learn* model.
    Every converter adds nodes to an existing container
    which is converted into a *ONNX* graph by
    an instance of :class:`Topology <skl2onnx.common._topology.Topology>`.
    """

    def __init__(self, sklearn_model):
        super(SklearnModelContainerNode, self).__init__(sklearn_model)
        # Scikit-learn models have no input and output specified, so we create them and store them in this container.
        self._inputs = []
        self._outputs = []

    @property
    def input_names(self):
        return [variable.raw_name for variable in self._inputs]

    @property
    def output_names(self):
        return [variable.raw_name for variable in self._outputs]

    def add_input(self, variable):
        # The order of adding variables matters. The final model's input names are sequentially added as this list
        if variable not in self._inputs:
            self._inputs.append(variable)

    def add_output(self, variable):
        # The order of adding variables matters. The final model's output names are sequentially added as this list
        if variable not in self._outputs:
            self._outputs.append(variable)


class ModelComponentContainer(ModelContainer):
    '''
    In the conversion phase, this class is used to collect all materials required
    to build an *ONNX* *GraphProto*, which is encapsulated in a *ONNX* *ModelProto*.
    '''

    def __init__(self, target_opset, options=None):
        '''
        :param target_opset: number, for example, 7 for *ONNX 1.2*, and 8 for *ONNX 1.3*.
        :param targeted_onnx: A string, for example, '1.1.2' and '1.2'.
        :param options: see :ref:`l-conv-options`
        '''
        # Inputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.inputs = []
        # Outputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.outputs = []
        # ONNX tensors (type: TensorProto). They are initializers of ONNX GraphProto.
        self.initializers = []
        # Intermediate variables in ONNX computational graph. They are ValueInfoProto in ONNX.
        self.value_info = []
        # ONNX nodes (type: NodeProto) used to define computation structure
        self.nodes = []
        # ONNX operators' domain-version pair set. They will be added into opset_import field in the final ONNX model.
        self.node_domain_version_pair_sets = set()
        # The targeted ONNX operator set (referred to as opset) that matches the ONNX version.
        self.target_opset = target_opset
        # Additional options given to converters.
        self.options = options

    def _make_value_info(self, variable):
        value_info = helper.ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        if variable.type.doc_string:
            value_info.doc_string = variable.type.doc_string
        return value_info

    def add_input(self, variable):
        '''
        Adds our *Variable* object defined _parser.py into the the input list
        of the final ONNX model.

        :param variable: The Variable object to be added
        '''
        self.inputs.append(self._make_value_info(variable))

    def add_output(self, variable):
        '''
        Adds our *Variable* object defined *_parser.py* into the the output list
        of the final ONNX model.

        :param variable: The Variable object to be added
        '''
        self.outputs.append(self._make_value_info(variable))

    def add_initializer(self, name, onnx_type, shape, content):
        '''
        Adds a *TensorProto* into the initializer list of the final ONNX model.

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g., TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list or a float array).
        :return: created tensor
        '''
        if any(d is None for d in shape):
            raise ValueError('Shape of initializer cannot contain None')
        tensor = helper.make_tensor(name, onnx_type, shape, content)
        self.initializers.append(tensor)
        return tensor

    def add_value_info(self, variable):
        self.value_info.append(self._make_value_info(variable))
    
    def _check_operator(self, op_type):
        """
        Checks that if *op_type* is one of the operator defined in
        :mod:`skl2onnx.common._apply_container`, then it was called
        from a function defined in this sub module by looking
        into the callstack. The test is enabled for *python >= 3.6*.
        """
        if op_type in _apply_operation_specific and sys.version_info[:2] >= (3, 6):
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
                raise RuntimeError("Operator '{0}' should be added with function '{1}' in submodule _apply_operation.\n{2}".format(
                    op_type, fct.__name__, "\n".join(files)))                

    def add_node(self, op_type, inputs, outputs, op_domain='', op_version=1, **attrs):
        '''
        Adds a *NodeProto* into the node list of the final ONNX model. If the input operator's domain-version information
        cannot be found in our domain-version pool (a Python set), we may add it.

        :param op_type: A string (e.g., Pool and Conv) indicating the type of the NodeProto
        :param inputs: A list of strings. They are the input variables' names of the considered NodeProto
        :param outputs: A list of strings. They are the output variables' names of the considered NodeProto
        :param op_domain: The domain name (e.g., ai.onnx.ml) of the operator we are trying to add.
        :param op_version: The version number (e.g., 0 and 1) of the operator we are trying to add.
        :param attrs: A Python dictionary. Keys and values are attributes' names and attributes' values, respectively.
        '''
        if op_domain in ('', None):
            op_domain = get_domain()
        self._check_operator(op_type)

        if isinstance(inputs, (six.string_types, six.text_type)):
            inputs = [inputs]
        if isinstance(outputs, (six.string_types, six.text_type)):
            outputs = [outputs]
        if not isinstance(inputs, list) or not all(isinstance(s, (six.string_types, six.text_type)) for s in inputs):
            type_list = ','.join(list(str(type(s)) for s in inputs))
            raise ValueError('Inputs must be a list of string but get [%s]' % type_list)
        if not isinstance(outputs, list) or not all(isinstance(s, (six.string_types, six.text_type)) for s in outputs):
            type_list = ','.join(list(str(type(s)) for s in outputs))
            raise ValueError('Outputs must be a list of string but get [%s]' % type_list)
        for k, v in attrs.items():
            if v is None:
                raise ValueError('Failed to create ONNX node. Undefined attribute pair (%s, %s) found' % (k, v))

        node = helper.make_node(op_type, inputs, outputs, **attrs)
        node.domain = op_domain

        self.node_domain_version_pair_sets.add((op_domain, op_version))
        self.nodes.append(node)

    def get_options(self, model, default_values=None):
        """
        Returns additional options for a model.
        It first looks by class then by id (``id(model)``).
        :param model: model being converted
        :param default_values: default options (it is modified by the function)
        :return: dictionary
        """
        opts = {} if default_values is None else default_values
        if self.options is not None:
            opts.update(self.options.get(type(model), {}))
            opts.update(self.options.get(id(model), {}))
        return opts
