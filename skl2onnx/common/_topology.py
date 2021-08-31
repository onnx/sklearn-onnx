# SPDX-License-Identifier: Apache-2.0


import re
import warnings
import pprint
from logging import getLogger
from collections import OrderedDict
import numpy as np
from onnx import onnx_pb as onnx_proto
from onnxconverter_common.data_types import (  # noqa
    DataType, TensorType,
    FloatType, Int64Type, StringType,
    DictionaryType, FloatTensorType,  # noqa
    Int64TensorType, SequenceType,  # noqa
    StringTensorType, DoubleTensorType,
    Int32TensorType, BooleanTensorType,
    DoubleTensorType)
try:
    from onnxconverter_common.data_types import (
        Int8TensorType, UInt8TensorType)
except ImportError:
    Int8TensorType = None
    UInt8TensorType = None
from ..proto import (
    get_opset_number_from_onnx,
    get_latest_tested_opset_version
)
from ..proto.onnx_helper_modified import (
    make_graph, make_model, make_tensor_value_info
)
from . import _registration
from . import utils
from .exceptions import MissingShapeCalculator, MissingConverter
from ._container import ModelComponentContainer, _build_options
from .onnx_optimisation_identity import onnx_remove_node_identity

type_fct = type


try:
    from onnxconverter_common.topology import OPSET_TO_IR_VERSION
    assert OPSET_TO_IR_VERSION[14] is not None
except (ImportError, KeyError):
    OPSET_TO_IR_VERSION = {
        1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
        7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 7,
        13: 7, 14: 7
    }

OPSET_ML_TO_OPSET = {1: 11, 2: 13}

logger = getLogger('skl2onnx')


class Variable:
    """
    Defines a variable which holds any data defined
    from *ONNX* types.
    """
    _UNIQUE_NUMBER_ = 0

    def __init__(self, raw_name, onnx_name, scope, type=None):
        """
        :param raw_name: A string indicating the variable's name in the
                         original model. Usually, it's the seed string
                         used to created its ONNX name (i.e., the
                         field *onnx_name* below).
        :param onnx_name: A string indicating the variable's name in
                          the converted model
        :param scope: A string. It's the name of the scope where this
                      variable is declared
        :param type: A type object defined in .common.data_types.py;
                     e.g., FloatTensorType
        """
        if not isinstance(raw_name, str):
            raise TypeError(
                "raw_name must be a string not '%s'." % raw_name.__class__)
        if not isinstance(onnx_name, str) or '(' in onnx_name:
            if onnx_name.startswith('u(') and onnx_name[-1] == ')':
                onnx_name0 = onnx_name
                if scope is None:
                    onnx_name = "UU%03dUU" % Variable._UNIQUE_NUMBER_
                    Variable._UNIQUE_NUMBER_ += 1
                else:
                    onnx_name = scope.get_unique_variable_name("U")
                logger.debug(
                    '[Var] rename raw_name=%r, onnx_name=%r into %r' % (
                        raw_name, onnx_name0, onnx_name))
            else:
                raise TypeError(
                    "onnx_name must be a string not %r." % onnx_name)

        if type is not None:
            shape = type.shape
            if shape is not None:
                not_none = [v for v in shape if v is not None]
                if len(not_none) and min(not_none) == 0:
                    raise RuntimeError(
                        "A variable cannot be empty, raw_name=%r, "
                        "onnx_name=%r, shape=%r, type=%r." % (
                            raw_name, onnx_name, shape, type))

        self._raw_name = raw_name
        self._onnx_name = onnx_name
        self._scope = scope
        self._type = type
        self._parent = None

        # The following fields are bool variables used in parsing and
        # compiling stages
        self._is_fed = None
        self._is_root = None
        self._is_leaf = None
        if self.type is not None and not isinstance(self.type, DataType):
            raise TypeError(
                "shape must be a DataType not {}.".format(self.type))
        if isinstance(self.type, TensorType):
            shape = self.type.shape
            if not isinstance(shape, (list, tuple)):
                try:
                    shape = list(shape)
                except TypeError:
                    raise TypeError("shape must be a tuple or a list not "
                                    "{}.".format(type_fct(shape)))
            for dim in shape:
                if dim is None:
                    continue
                if not isinstance(dim, (int, np.int32, np.int64)):
                    raise TypeError("shape must contains integers not "
                                    "'{}'.".format(dim))
        logger.debug('[Var] +%s' % self)

    @property
    def raw_name(self):
        return self._raw_name

    @property
    def onnx_name(self):
        return self._onnx_name

    @property
    def scope(self):
        return self._scope

    @property
    def type(self):
        return self._type

    @property
    def is_fed(self):
        return self._is_fed

    @property
    def is_root(self):
        return self._is_root

    @property
    def is_leaf(self):
        return self._is_leaf

    def init_status(self, is_fed=None, is_root=None, is_leaf=None):
        if is_fed is not None and is_fed != self.is_fed:
            logger.debug('[Var] update is_fed=%r for %r, parent=%r' % (
                is_fed, self, self._parent))
            self._is_fed = is_fed
        if is_root is not None and is_root != self.is_root:
            logger.debug('[Var] update is_root=%r for %r' % (is_root, self))
            self._is_root = is_root
        if is_leaf is not None and is_leaf != self.is_leaf:
            logger.debug('[Var] update is_leaf=%r for %r' % (is_leaf, self))
            self._is_leaf = is_leaf

    def __setattr__(self, name, value):
        if name == "type":
            self.set_type(value)
        elif name == "onnx_name":
            raise AttributeError("You must use method set_onnx_name.")
        elif name in {"is_fed", "is_root", "is_leaf"}:
            raise AttributeError("You must use method init_status.")
        elif name in {'scope', 'raw_name'}:
            raise AttributeError("scope or raw_name cannot be changed.")
        self.__dict__[name] = value

    def set_type(self, new_type):
        logger.debug('[Var] update type= for %r' % self)
        self._type = new_type

    def set_onnx_name(self, onnx_name):
        if onnx_name != self._onnx_name:
            logger.debug('[Var] update onnx_name, from %r to %r in %r' % (
                self.onnx_name, onnx_name, self))
            self._onnx_name = onnx_name

    def set_parent(self, operator):
        if self._parent is not None:
            raise RuntimeError(
                "This variable is already the output of operator %r. "
                "It cannot be the output of %r." % (self._parent, operator))
        logger.debug('[Var] set parent for %r, parent=%r' % (
            self, operator))
        self._parent = operator

    def get_first_dimension(self):
        """
        Returns the first dimension (batch dimension) or
        None if not specified (shape is empty).
        """
        if (self.type is None or self.type.shape is None or
                len(self.type.shape) == 0):
            return None
        return self.type.shape[0]

    @property
    def full_name(self):
        """
        Return a globally unique variable ID
        """
        return self.onnx_name

    def __repr__(self):
        return ("Variable('{0}', '{1}', type={2})".format(
                self.raw_name, self.onnx_name, self.type))

    @staticmethod
    def from_pb(obj):
        """
        Creates a data type from a protobuf object.
        """
        def get_dim(d):
            r = d.dim_value
            if "dim_param" in str(d):
                return None
            if r == 0:
                # dim_value is 0 when it is 0 or undefined
                return 0 if "0" in str(d) else None
            return r

        def get_shape(tt):
            return [get_dim(tt.shape.dim[i])
                    for i in range(len(tt.shape.dim))]

        if hasattr(obj, 'extend'):
            return [Variable.from_pb(o) for o in obj]

        name = obj.name
        if obj.type.tensor_type:
            tt = obj.type.tensor_type
            elem = tt.elem_type
            shape = get_shape(tt)
            if elem == onnx_proto.TensorProto.FLOAT:
                ty = FloatTensorType(shape)
            elif elem == onnx_proto.TensorProto.BOOL:
                ty = BooleanTensorType(shape)
            elif elem == onnx_proto.TensorProto.DOUBLE:
                ty = DoubleTensorType(shape)
            elif elem == onnx_proto.TensorProto.STRING:
                ty = StringTensorType(shape)
            elif elem == onnx_proto.TensorProto.INT64:
                ty = Int64TensorType(shape)
            elif elem == onnx_proto.TensorProto.INT32:
                ty = Int32TensorType(shape)
            elif (UInt8TensorType is not None and
                    elem == onnx_proto.TensorProto.UINT8):
                ty = UInt8TensorType(shape)
            elif (Int8TensorType is not None and
                    elem == onnx_proto.TensorProto.INT8):
                ty = Int8TensorType(shape)
            elif elem == 0:
                ty = FloatTensorType(shape)
            else:
                raise NotImplementedError(
                    "Unsupported type '{}' (elem_type={}).".format(
                        type(obj.type.tensor_type), elem))
        else:
            raise NotImplementedError("Unsupported type '{}' as "
                                      "a string ({}).".format(
                                          type(obj), obj))

        return Variable(name, name, None, ty)

    def __iter__(self):
        "Enables expression such as `a,b = self`."
        yield self.onnx_name
        yield self.type

    def __getitem__(self, index):
        if index == 0:
            return self.onnx_name
        if index == 1:
            return self.type
        raise IndexError("Unreachable element at index %d." % index)


class VariableStr(Variable):
    """
    Defines a variable a string. This should be avoided.
    """

    def __init__(self, name, scope=None, type=None):
        Variable.__init__(self, name, name, scope=scope, type=type)

    @property
    def raw_name(self):
        return self._raw_name

    @property
    def onnx_name(self):
        if self._onnx_name.startswith("u("):
            raise RuntimeError(
                "Variable should be renamed as onnx_name=%r."
                "" % self._onnx_name)
        return self._onnx_name


class Operator:
    """
    Defines an operator available in *ONNX*.
    """
    class OperatorList(list):
        def __init__(self, parent, kind):
            super(Operator.OperatorList, self).__init__()
            self.parent = parent
            self.kind = kind

        def __eq__(self, second):
            raise NotImplementedError(
                "Operator equal not implemented and not needed.")

        def append(self, v):
            if not isinstance(v, Variable):
                raise TypeError(
                    "Input and output must be of type Variable not %r."
                    "" % type(v))
            if self.kind == 'Out':
                v.set_parent(self.parent)
            super(Operator.OperatorList, self).append(v)
            logger.debug("[Op] add %s %r to %r" % (self.kind, v, self.parent))

        def extend(self, vs):
            for v in vs:
                self.append(v)

        def __getitem__(self, i):
            v = list.__getitem__(self, i)
            if isinstance(i, int) and not isinstance(v, Variable):
                raise TypeError("Element %d must be a Variable not %r." % (
                    i, type(v)))
            return v

        def __setitem__(self, i, v):
            raise LookupError(
                "Setter should not be used to modify an element.")

        def set_element(self, i, v):
            "Updates element i."
            if not isinstance(v, Variable):
                raise TypeError(
                    "Value v must be a Variable not %r." % type(v))
            logger.debug("[Op] %s-change element %d from %r to %r in %r" % (
                self.kind, i, self[i], v, self.parent))
            list.__setitem__(self, i, v)

        def to_string(self):
            names = []
            for o in self:
                if hasattr(o, 'onnx_name'):
                    names.append(o.onnx_name)
                else:
                    names.append('"%s"' % str(o))
            return ",".join(names)

    def __init__(self, onnx_name, scope, type, raw_operator,
                 target_opset, scope_inst):
        """
        :param onnx_name: A unique ID, which is a string
        :param scope: The name of the scope where this operator is
                      declared. It's a string.
        :param type: A object which uniquely characterizes the type of
                     this operator. For example, it can be a string,
                     pooling, if this operator is associated with a
                     CoreML pooling layer.
        :param raw_operator: The original operator which defines this operator;
                             for example, a scikit-learn Imputer and
                             a CoreML Normalizer.
        :param target_opset: The target opset number for the converted model.
        :param scope_inst: :class:`Scope` instance the operator belongs to
        """
        if isinstance(raw_operator, str):
            raise RuntimeError("Parameter raw_operator must be an object not "
                               "a string '{0}'.".format(raw_operator))
        # operator name in the converted model, if raw_operator
        # is not None, output_shapes can be guessed
        # from the raw model. Otherwise, it can be guessed
        # from the input shapes.
        self.onnx_name = onnx_name
        self.scope = scope
        self.type = type
        self.raw_operator = raw_operator
        self.inputs = Operator.OperatorList(self, 'In')
        self.outputs = Operator.OperatorList(self, 'Out')
        self._is_evaluated = None
        self.target_opset = target_opset
        self.scope_inst = scope_inst
        logger.debug('[Op] +%r' % self)

    def new_raw_operator(self, raw_operator, alias):
        """
        Returns a shallow copy of this operator,
        changes the raw_operator but keeps the same inputs
        and outputs.
        """
        op = Operator(self.onnx_name, self.scope, alias, raw_operator,
                      self.target_opset, self.scope_inst)
        op.inputs = self.inputs
        op.outputs = self.outputs
        return op

    def __repr__(self):
        try:
            textop = repr(self.raw_operator)
        except KeyError:
            # The line above fails for python 3.7
            textop = type(self.raw_operator)
        if isinstance(textop, str) and "\n" in textop:
            textop = textop.replace('\n', '').replace(' ', '')
        return ("Operator(type='{0}', onnx_name='{1}', inputs='{2}', "
                "outputs='{3}', raw_operator={4})".format(
                    self.type, self.onnx_name,
                    self.inputs.to_string(),
                    self.outputs.to_string(),
                    textop))

    def __setattr__(self, name, value):
        if name in ('inputs', 'outputs'):
            if (isinstance(value, list) and
                    not isinstance(value, Operator.OperatorList)):
                if name == 'inputs':
                    self.inputs = Operator.OperatorList(self, 'In')
                    self.inputs.extend(value)
                    return
                if name == 'outputs':
                    self.outputs = Operator.OperatorList(self, 'Out')
                    self.outputs.extend(value)
                    return
            if not isinstance(value, Operator.OperatorList):
                raise TypeError(
                    "inputs or outputs must be of type Operator.OperatorList.")
        self.__dict__[name] = value

    @property
    def is_evaluated(self):
        return self._is_evaluated

    def init_status(self, is_evaluated=None):
        if is_evaluated is not None and is_evaluated != self.is_evaluated:
            logger.debug('[Op] update is_evaluated=%r for %r' % (
                is_evaluated, self))
            self._is_evaluated = is_evaluated

    @property
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        return self.onnx_name

    @property
    def input_full_names(self):
        """
        Return all input variables' names
        """
        return [variable.full_name for variable in self.inputs]

    @property
    def output_full_names(self):
        """
        Return all output variables' names
        """
        return [variable.full_name for variable in self.outputs]

    @property
    def original_operator(self):
        """
        Return the original operator/layer
        """
        return self.raw_operator

    def infer_types(self):
        # Invoke a core inference function
        if self.type is None:
            raise MissingShapeCalculator(
                "Unable to find a shape calculator for type '{}'.".format(
                    type(self.raw_operator)))
        try:
            shape_calc = _registration.get_shape_calculator(self.type)
        except ValueError:
            raise MissingShapeCalculator(
                "Unable to find a shape calculator for alias '{}' "
                "and type '{}'.".format(self.type, type(self.raw_operator)))
        logger.debug("[Shape0] %r fed %r - %r" % (
            self,
            "".join(str(i.is_fed) for i in self.inputs),
            "".join(str(i.is_fed) for i in self.outputs)))
        shape_calc(self)


class Scope:
    """
    Every node of an *ONNX* graph must be unique. This class holds the list
    of existing name for every node already defined in graph. It also
    provides functions to create a unique unused name.
    """

    def __init__(self, name, parent_scopes=None, variable_name_set=None,
                 operator_name_set=None, target_opset=None,
                 custom_shape_calculators=None, options=None,
                 registered_models=None):
        """
        :param name: A string, the unique ID of this scope in a
                     Topology object
        :param parent_scopes: A list of Scope objects. The last element
                              should be the direct parent scope (i.e.,
                              where this scope is declared).
        :param variable_name_set: A set of strings serving as the name
                                  pool of variables
        :param operator_name_set: A set of strings serving as the name
                                  pool of operators
        :param target_opset: The target opset number for the converted
                             model.
        :param custom_conversion_functions: a dictionary for specifying
                                the user customized conversion function
        :param custom_shape_calculators: a dictionary for specifying
                                the user customized shape calculator
        :param options: see :ref:`l-conv-options`
        :param registered_models: registered models
        """
        self.name = name
        self.parent_scopes = parent_scopes if parent_scopes else list()
        self.onnx_variable_names = (
            variable_name_set if variable_name_set is not None else set())
        self.onnx_operator_names = (
            operator_name_set if operator_name_set is not None else set())
        self.target_opset = target_opset
        self.custom_shape_calculators = custom_shape_calculators

        # An one-to-many map from raw variable name to ONNX variable
        # names. It looks like
        # (key, value) = (raw_name, [onnx_name, onnx_name1, onnx_name2, ..., onnx_nameN]) # noqa
        # The last name may hide all other names in this scope.
        self.variable_name_mapping = {}

        # A map of local variables defined in this scope.
        # (key, value) = (onnx_name, variable)
        self.variables = OrderedDict()
        self.input_variables = []
        self.output_variables = []

        # A map of local operators defined in this scope.
        # (key, value) = (onnx_name, operator)
        self.operators = {}

        # Additional options given to converters.
        self.options = options

        # Registered models
        self.registered_models = registered_models

    def get(self, var_name, default_value):
        "Returns variable with 'name' or default value is not found."
        return self.variables.get(var_name, default_value)

    def temp(self):
        """
        Creates a new Scope with the same options but no names.
        """
        scope = Scope(
            'temp', parent_scopes=self.parent_scopes,
            target_opset=self.target_opset,
            custom_shape_calculators=self.custom_shape_calculators,
            options=self.options,
            registered_models=self.registered_models)
        return scope

    def has_variable_name(self, name):
        """
        Tells if a variable is already registered.
        """
        return name in self.onnx_variable_names

    def get_shape_calculator(self, model_type):
        """
        Returns the shape calculator for the given model type.

        :param model_type: model type such as *LogisticRegression*
        :return: alias or None if not found
        """
        return self.custom_shape_calculators.get(model_type, None)

    def get_unique_variable_name(self, seed):
        """
        Creates a unique variable ID based on the given seed.
        """
        if not isinstance(seed, str):
            raise TypeError("Parameter seed must be a string not {}."
                            "".format(type(seed)))
        name = Topology._generate_unique_name(seed, self.onnx_variable_names)
        return name

    def get_unique_operator_name(self, seed):
        """
        Creates a unique operator ID based on the given seed.
        """
        return Topology._generate_unique_name(seed, self.onnx_operator_names)

    def declare_local_variable(self, raw_name, type=None, prepend=False,
                               missing_type=False):
        """
        This function may create a new variable in this scope. If
        *raw_name* has been used to create other variables, the new
        variable will hide all other variables created using *raw_name*.
        """
        if type is None and not missing_type:
            raise RuntimeError("Unknown type for %r." % raw_name)
        # Get unique ID for the new variable
        onnx_name = self.get_unique_variable_name(raw_name)

        # Create the variable
        variable = Variable(raw_name, onnx_name, self.name, type)
        self.variables[onnx_name] = variable

        if raw_name in self.variable_name_mapping:
            # Hide existing variables with the same raw_name
            if not prepend:
                self.variable_name_mapping[raw_name].append(onnx_name)
            else:
                self.variable_name_mapping[raw_name].insert(0, onnx_name)
        else:
            self.variable_name_mapping[raw_name] = [onnx_name]
        return variable

    def declare_local_input(self, raw_name, type=None, prepend=False):
        """
        Calls `declare_local_variable`. Registers this variable
        as an input.
        """
        var = self.declare_local_variable(
            raw_name, type=type, prepend=prepend)
        self.input_variables.append(var)
        return var

    def declare_local_output(self, raw_name, type=None, prepend=False,
                             missing_type=False):
        """
        Calls `declare_local_variable`. Registers this variable
        as an output.
        """
        var = self.declare_local_variable(
            raw_name, type=type, prepend=prepend,
            missing_type=missing_type)
        self.output_variables.append(var)
        return var

    def declare_local_operator(self, type, raw_model=None):
        """
        This function is used to declare new local operator.
        """
        onnx_name = self.get_unique_operator_name(str(type))
        operator = Operator(onnx_name, self.name, type, raw_model,
                            self.target_opset, scope_inst=self)
        self.operators[onnx_name] = operator
        return operator

    def _get_allowed_options(self, model, fail=True):
        if self.registered_models is not None:
            if type(model) not in self.registered_models['aliases']:
                if fail:
                    raise NotImplementedError(
                        "No registered models, no known allowed options "
                        "for model '{}'.".format(model.__class__.__name__))
                return {}
            alias = self.registered_models['aliases'][type(model)]
            conv = self.registered_models['conv'][alias]
            allowed = conv.get_allowed_options()
            return allowed
        raise NotImplementedError(
            "No registered models, no known allowed options "
            "for model '{}'.".format(model.__class__.__name__))

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

    def get_options(self, model, default_values=None, fail=True):
        """
        Returns additional options for a model.
        It first looks by class then by id (``id(model)``).
        :param model: model being converted
        :param default_values: default options (it is modified by
                               the function)
        :param fail: fails if option it not found
        :return: dictionary
        """
        return _build_options(
            model, self.options, default_values,
            self._get_allowed_options(model, fail=fail),
            fail=fail)

    def replace_raw_operator(self, op1, op2, alias):
        """
        Replaces every raw operator op1 by op2.
        The function uses `id()` to detect op1.
        """
        for v in self.operators.values():
            if id(v.raw_operator) == id(op1):
                logger.debug('[Scope] replace %d by %d in %r.' % (
                    id(v.raw_operator), id(op1), v))
                v.raw_operator = op2
                v.type = alias


class Topology:
    """
    Holds instances on :class:`Scope <skl2onnx.common._topology.Scope>` and
    :class:`SklearnModelContainer
    <skl2onnx.common._container.SklearnModelContainer>`.
    These are filled by the converters while a pipeline is being converted.
    """

    def __init__(self, model, default_batch_size=1, initial_types=None,
                 target_opset=None, custom_conversion_functions=None,
                 custom_shape_calculators=None, registered_models=None):
        """
        Initializes a *Topology* object, which is an intermediate
        representation of a computational graph.

        :param model: RawModelContainer object or one of its derived
                      classes. It contains the original model.
        :param default_batch_size: batch_size prepend to scalar and
                                   array types from CoreML. It's usually
                                   1 or None.
        :param initial_types: A list providing some types for some
                              root variables.
        Each element is a tuple of a variable name and a type defined
        in *data_types.py*.
        :param custom_conversion_functions: a dictionary for specifying
                                the user customized conversion function
        :param custom_shape_calculators: a dictionary for specifying the
                                        user customized shape calculator
        :param registered_models: registered models
        """
        self.scopes = []
        self.raw_model = model
        self.scope_names = set()
        self.variable_name_set = set()
        self.operator_name_set = set()
        self.initial_types = initial_types if initial_types else list()
        self.default_batch_size = default_batch_size
        self.target_opset = target_opset
        self.custom_conversion_functions = (
            custom_conversion_functions if custom_conversion_functions else {})
        self.custom_shape_calculators = (
            custom_shape_calculators if custom_shape_calculators else {})

        for k in self.custom_conversion_functions:
            if not callable(k):
                raise TypeError("Keys in custom_conversion_functions must be "
                                "types not strings.")
        for k in self.custom_shape_calculators:
            if not callable(k):
                raise TypeError("Keys in custom_shape_calculators must be "
                                "types not strings.")

        # A map of local overwritten model aliases.
        self.model_aliases = {}
        all_model_types = (set(self.custom_conversion_functions)
                           | set(self.custom_shape_calculators))
        for mtype in all_model_types:
            alias = "{}_{}".format(mtype.__name__, id(self))
            self.model_aliases[mtype] = alias

        # Registered models
        if registered_models is None:
            raise AssertionError()
        self.registered_models = registered_models

    @staticmethod
    def _generate_unique_name(seed, existing_names):
        """
        Produce an unique string based on the seed
        :param seed: a string
        :param existing_names: a set containing strings which cannot be
                               produced
        :return: a string similar to the seed
        """
        if seed == '':
            raise ValueError('Name seed must be a non-empty string.')

        # Make the seed meet C-style naming convention
        # Only alphabets and numbers are allowed
        seed = re.sub('[^0-9a-zA-Z]', '_', seed)
        # The first symbol cannot be a number
        if re.match('^[0-9]', seed):
            seed = '_' + seed

        # If seed has never been seen, we return it as it is. Otherwise,
        # we will append an number to make it unique.
        if seed not in existing_names:
            existing_names.add(seed)
            return seed
        else:
            i = 1
            while seed + str(i) in existing_names:
                i += 1
            new_name = seed + str(i)
            existing_names.add(new_name)
            return new_name

    def get_unique_scope_name(self, seed):
        return Topology._generate_unique_name(seed, self.scope_names)

    def declare_scope(self, seed, parent_scopes=None, options=None):
        """
        Creates a new :class:`Scope <skl2onnx.common._topology.Scope>`
        and appends it to the list of existing scopes.
        """
        if len(self.scopes) != 0:
            raise RuntimeError(
                "Only one scope can be created.")
        scope = Scope(
            self.get_unique_scope_name(seed), parent_scopes,
            self.variable_name_set, self.operator_name_set, self.target_opset,
            custom_shape_calculators=self.custom_shape_calculators,
            options=options, registered_models=self.registered_models)

        # Declare input variables.
        # They should be the inputs of the scikit-learn
        # model you want to convert into ONNX.
        for var_name, initial_type in self.initial_types:
            scope.declare_local_input(var_name, initial_type)
        self.scopes.append(scope)
        return scope

    def unordered_operator_iterator(self):
        for scope in self.scopes:
            for operator in scope.operators.values():
                yield operator

    def unordered_variable_iterator(self):
        for scope in self.scopes:
            for variable in scope.variables.values():
                yield variable

    def call_converter(self, operator, container, verbose=0):
        "Calls converter for operator *operator*."
        mtype = type(operator.raw_operator)
        if mtype in self.custom_conversion_functions:
            conv = self.custom_conversion_functions[mtype]
        elif operator.type in self.custom_conversion_functions:
            conv = self.custom_conversion_functions[operator.type]
        elif hasattr(operator.raw_operator, "onnx_converter"):
            conv = operator.raw_operator.onnx_converter()
        else:
            # Convert the selected operator into some ONNX objects and
            # save them into the container
            try:
                conv = _registration.get_converter(operator.type)
            except ValueError:
                raise MissingConverter(
                    "Unable to find converter for alias '{}' type "
                    "'{}'. You may raise an issue at "
                    "https://github.com/onnx/sklearn-onnx/issues."
                    "".format(operator.type,
                              type(getattr(operator, 'raw_model', None))))

        container.validate_options(operator)
        if verbose > 0:
            print("[call_converter] call converter for %r." % operator.type)
        logger.debug("[Conv] call %r fed %r - %r" % (
            operator,
            "".join(str(i.is_fed) for i in operator.inputs),
            "".join(str(i.is_fed) for i in operator.outputs)))
        conv(self.scopes[0], operator, container)
        logger.debug("[Conv] end - %r" % operator)

    def call_shape_calculator(self, operator):
        "Calls shape_calculator for operator *operator*."
        mtype = type(operator.raw_operator)
        if mtype in self.custom_shape_calculators:
            # overwritten operator.
            source = 'custom'
            shape_calc = self.custom_shape_calculators[mtype]
        elif operator.type in self.custom_shape_calculators:
            source = 'custom'
            shape_calc = self.custom_shape_calculators[operator.type]
        elif hasattr(operator.raw_operator, "onnx_shape_calculator"):
            source = 'onnx_shape_calculator'
            shape_calc = operator.raw_operator.onnx_shape_calculator()
        else:
            source = ""
            shape_calc = None

        if shape_calc is not None:
            logger.debug("[Shape1] %r fed %r - %r (source=%r)" % (
                operator,
                "".join(str(i.is_fed) for i in operator.inputs),
                "".join(str(i.is_fed) for i in operator.outputs),
                source))
            shape_calc(operator)
        else:
            logger.debug('[Shape2] call infer_types for %r' % operator)
            operator.infer_types()

    def _initialize_graph_status_for_traversing(self):
        """
        Initialize the status of all variables and operators before
        traversing the graph. Only used by convert_operators.
        """
        if len(self.scopes) != 1:
            raise RuntimeError(
                "Only one scope is allowed not %d." % len(self.scopes))
        input_names = set(v.onnx_name for v in self.scopes[0].input_variables)
        if len(input_names) == 0:
            raise RuntimeError("No detected inputs.")
        for variable in self.unordered_variable_iterator():
            is_input = variable.onnx_name in input_names
            variable.init_status(is_fed=is_input)

        for operator in self.unordered_operator_iterator():
            operator.init_status(is_evaluated=False)

    def _propagate_status(self, operator, container, fed_variables):
        """
        Propagates status *is_fed* based on output variable
        and node added in the container.
        """
        vars = {}
        for node in container.nodes:
            for i in node.input:
                if i not in vars:
                    vars[i] = []
                vars[i].append(node)

        stack = [v.onnx_name for v in operator.outputs if v.is_fed]
        stack.extend(v.onnx_name for v in operator.inputs if v.is_fed)
        scope = self.scopes[0]
        while len(stack) > 0:
            nodes = {}
            for name in stack:
                if name not in vars:
                    continue
                for n in vars[name]:
                    nodes[id(n)] = n
            stack = []
            for node in nodes.values():
                if all(fed_variables.get(n, False) for n in node.input):
                    for o in node.output:
                        if o not in fed_variables:
                            fed_variables[o] = o
                            stack.append(o)
                            if o in scope.variables:
                                var = scope.variables[o]
                                var.init_status(is_fed=True)

    def convert_operators(self, container=None, verbose=0):
        """
        Calls all converters and shape_calculator for existing
        operators. It also processes new operators created by
        converters.
        """
        def _check_operator_(operator):
            if not isinstance(operator.inputs, Operator.OperatorList):
                raise TypeError(
                    "operator.inputs must be a Operator.OperatorList "
                    "not %r." % type(operator.inputs))
            if not isinstance(operator.outputs, Operator.OperatorList):
                raise TypeError(
                    "operator.outputs must be a Operator.OperatorList "
                    "not %r." % type(operator.outputs))
            if any(not isinstance(i, Variable) for i in operator.inputs):
                raise TypeError(
                    "One input is not a Variable for operator %r - %r."
                    "" % (type(operator.raw_operator), operator))
            if any(not isinstance(i, Variable) for i in operator.outputs):
                raise TypeError(
                    "One output is not a Variable for operator %r - %r."
                    "" % (type(operator.raw_operator), operator))

        def _check_variable_(variable, operator):
            if variable.is_fed:
                add = ["", "--DEBUG-INFO--"]
                add.append("self.variable_name_set=%s" % (
                    pprint.pformat(self.variable_name_set)))
                add.append("self.operator_name_set=%s" % (
                    pprint.pformat(self.operator_name_set)))
                for scope in self.scopes:
                    add.append('---')
                    add.append(pprint.pformat(
                        scope.variable_name_mapping))
                    add.append('---')
                    for var in scope.variables.values():
                        add.append("   is_fed=%s %s" % (
                            getattr(var, 'is_fed', '?'), var))
                    add.append('---')
                    for op in scope.operators.values():
                        add.append("   is_evaluated=%s %s" % (
                            getattr(op, 'is_evaluated', '?'), op))
                add.append('---')
                for v in operator.inputs:
                    add.append(" inputs={}".format(v))
                for v in operator.outputs:
                    add.append(" outputs={}".format(v))
                raise RuntimeError(
                    "A variable is already assigned ({}) "
                    "for operator '{}' (name='{}'). "
                    "operator.is_evaluated={}, inputs.is_fed={}, "
                    "outputs.is_fed={}. "
                    "This may still happen if a converter is a "
                    "combination of sub-estimators and one "
                    "of them is producing this output. "
                    "In that case, an identity node must be "
                    "added.{}".format(
                        variable, operator.type,
                        operator.onnx_name, operator.is_evaluated,
                        [v.is_fed for v in operator.inputs],
                        [v.is_fed for v in operator.outputs],
                        "\n".join(add)))

        if verbose > 0:
            print("[convert_operators] begin")
        self._initialize_graph_status_for_traversing()
        fed_variables = {i.name: i for i in container.initializers}
        changes = 1
        while changes > 0:
            changes = 0
            if verbose > 0:
                print("[convert_operators] new iteration")
            ops = list(self.unordered_operator_iterator())
            for operator in ops:
                _check_operator_(operator)
                for var in operator.inputs:
                    if var.is_fed:
                        fed_variables[var.onnx_name] = var
                if (all(variable.is_fed for variable in operator.inputs) and
                        not operator.is_evaluated):

                    for variable in operator.outputs:
                        _check_variable_(variable, operator)
                        variable.init_status(is_fed=True)

                    self.call_shape_calculator(operator)
                    self.call_converter(operator, container, verbose=verbose)

                    for variable in operator.outputs:
                        variable.init_status(is_fed=True)
                        fed_variables[variable.onnx_name] = variable
                    fed_variables.update(
                        {i.name: i for i in container.initializers})
                    operator.init_status(is_evaluated=True)
                    self._propagate_status(operator, container, fed_variables)
                    changes += 1

                    if verbose > 0:
                        print('[convert_operators] yield %r.' % operator)

            if verbose > 0:
                print("[convert_operators] end iteration")
        if verbose > 0:
            print("[convert_operators] end.")

        # Last verification.
        not_evaluated = []
        for op in self.unordered_operator_iterator():
            if not op.is_evaluated:
                not_evaluated.append(op)
        if len(not_evaluated) > 0:
            rows = ["---VARS---"]
            for var in self.unordered_variable_iterator():
                rows.append(
                    "is_fed=%r is_leaf=%r is_root=%r - %r"
                    "" % (var.is_fed, var.is_leaf, var.is_root, var))
            rows.append("---OPERATORS---")
            for op in self.unordered_operator_iterator():
                rows.append("is_eval=%r - %r" % (op.is_evaluated, op))
            raise RuntimeError(
                "Not all operators have been evaluated. A variable name "
                "is probably misspelled.\n%s"
                "" % "\n".join(rows))

        # Input and output
        if len(self.scopes[0].input_variables) > 0:
            inputs = self.scopes[0].input_variables
        else:
            inputs = [v for v in self.unordered_variable_iterator()
                      if v.is_root]
        for i in inputs:
            container.add_input(i)
        outputs = [v for v in self.unordered_variable_iterator()
                   if v.is_leaf]
        for o in outputs:
            container.add_output(o)


def convert_topology(topology, model_name, doc_string, target_opset,
                     channel_first_inputs=None,
                     options=None, remove_identity=True,
                     verbose=0):
    """
    This function is used to convert our Topology object defined in
    _parser.py into a ONNX model (type: ModelProto).

    :param topology: The Topology object we are going to convert
    :param model_name: GraphProto's name. Let "model" denote the
        returned model. The string "model_name" would be
        assigned to "model.graph.name."
    :param doc_string: A string attached to the produced model
    :param target_opset: number or dictionary,
        for example, 7 for ONNX 1.2, and 8 for ONNX 1.3,
        a dictionary is used to indicate different opset for
        different domains
    :param options: see :ref:`l-conv-options`
    :param remove_identity: removes identity nodes
        include '1.1.2', '1.2', and so on.
    :param verbose: displays information while converting
    :return: a ONNX ModelProto
    """
    if target_opset is None:
        target_opset = get_latest_tested_opset_version()
    if isinstance(target_opset, dict):
        onnx_target_opset = target_opset.get(
            '', get_latest_tested_opset_version())
    else:
        onnx_target_opset = target_opset
    if onnx_target_opset > get_opset_number_from_onnx():
        found = get_opset_number_from_onnx()
        raise RuntimeError(
            "Parameter target_opset {} > {} is higher than the "
            "version of the installed onnx package. See "
            "https://github.com/onnx/onnx/blob/master/docs/"
            "Versioning.md#released-versions"
            ".".format(onnx_target_opset, found))
    if onnx_target_opset > get_latest_tested_opset_version():
        warnings.warn(
            "Parameter target_opset {} > {} is higher than the "
            "the latest tested version"
            ".".format(
                onnx_target_opset,
                get_latest_tested_opset_version()))

    container = ModelComponentContainer(
        target_opset, options=options,
        registered_models=topology.registered_models,
        white_op=topology.raw_model._white_op,
        black_op=topology.raw_model._black_op,
        verbose=verbose)

    # Traverse the graph from roots to leaves
    # This loop could eventually be parallelized.
    topology.convert_operators(container=container, verbose=verbose)
    container.ensure_topological_order()

    if len(container.inputs) == 0:
        raise RuntimeError("No detected inputs after conversion.")
    if len(container.outputs) == 0:
        raise RuntimeError("No detected outputs after conversion.")
    if verbose >= 2:
        print("---NODES---")
        for node in container.nodes:
            print("  %s - %s: %r -> %r" % (
                node.op_type, node.name, node.input, node.output))

    # Create a graph from its main components
    if container.target_opset_onnx < 9:
        # When calling ModelComponentContainer's add_initializer(...),
        # nothing is added into the input list. However, for ONNX target
        # opset < 9, initializers should also be a part of model's
        # (GraphProto) inputs. Thus, we create ValueInfoProto objects
        # from initializers (type: TensorProto) directly and then add
        # them into model's input list.
        extra_inputs = []  # ValueInfoProto list of the initializers
        for tensor in container.initializers:
            # Sometimes (especially when creating optional input values
            # such as RNN's initial hidden state), an initializer is also
            # one of the original model's input, so it has been added into
            # the container's input list. If this is the case, we need to
            # skip one iteration to avoid duplicated inputs.
            if tensor.name in [value_info.name for value_info in
                               container.inputs]:
                continue

            # Initializers are always tensors so we can just call
            # make_tensor_value_info(...).
            value_info = make_tensor_value_info(
                tensor.name, tensor.data_type, tensor.dims)
            extra_inputs.append(value_info)

        # Before ONNX opset 9, initializers were needed to be passed in
        # with inputs.
        graph = make_graph(container.nodes, model_name,
                           container.inputs + extra_inputs,
                           container.outputs, container.initializers)
    else:
        # In ONNX opset 9 and above, initializers are included as
        # operator inputs and therefore do not need to be passed as
        # extra_inputs.
        graph = make_graph(
            container.nodes, model_name, container.inputs,
            container.outputs, container.initializers)

    # Add extra information related to the graph
    graph.value_info.extend(container.value_info)

    # Create model
    onnx_model = make_model(graph)

    # Update domain version
    _update_domain_version(container, onnx_model)

    # Add extra information
    opv = min(onnx_target_opset,
              _get_main_opset_version(onnx_model) or onnx_target_opset)
    irv = OPSET_TO_IR_VERSION.get(opv, onnx_proto.IR_VERSION)
    onnx_model.ir_version = irv
    onnx_model.producer_name = utils.get_producer()
    onnx_model.producer_version = utils.get_producer_version()
    onnx_model.domain = utils.get_domain()
    onnx_model.model_version = utils.get_model_version()
    onnx_model.doc_string = doc_string

    # Removes many identity nodes,
    # the converter may introduct identity nodes
    # after a zipmap operator and onnx <= 1.7 does not
    # support that. It does not use onnxconverter-common
    # as the optimizer only support opset >= 9.
    if remove_identity:
        onnx_model = onnx_remove_node_identity(onnx_model)

    return onnx_model


def _update_domain_version(container, onnx_model):
    # Merge operator sets for the same domain, the largest version
    # number would be kept
    purified_operator_set = dict()
    for op_domain, op_version in container.node_domain_version_pair_sets:
        if op_domain not in purified_operator_set:
            purified_operator_set[op_domain] = op_version
        else:
            purified_operator_set[op_domain] = max(
                purified_operator_set[op_domain], op_version)

    # Fill operator sets
    i = 0
    for op_domain, op_version in purified_operator_set.items():
        if op_version is None:
            continue
        if i == 0 and len(onnx_model.opset_import) == 1:
            # Overwrite the default operator set created by
            # make_model(...)
            op_set = onnx_model.opset_import[0]
        else:
            # Just create one ONNX element in opset_import
            op_set = onnx_model.opset_import.add()
        op_set.domain = op_domain
        op_set.version = op_version
        i += 1
        if container.target_opset_any_domain(op_domain) < op_version:
            raise RuntimeError(
                'The specified opset %d is too low to convert '
                'this model, which requires at least opset '
                '%d.' % (
                    container.target_opset_any_domain(op_domain),
                    op_version))


def _get_main_opset_version(model):
    """
    Returns the main opset version.
    """
    mld = None
    for op in model.opset_import:
        if op.domain == '':
            return op.version
        if op.domain == "ai.onnx.ml":
            mld = op.version
    if mld is not None:
        return OPSET_ML_TO_OPSET.get(mld, None)
    return None
