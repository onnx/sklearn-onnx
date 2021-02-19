# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import re
import warnings
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
from .interface import OperatorBase
from .onnx_optimisation_identity import onnx_remove_node_identity
type_fct = type


try:
    from onnxconverter_common.topology import OPSET_TO_IR_VERSION
    assert OPSET_TO_IR_VERSION[13]
except (ImportError, KeyError):
    OPSET_TO_IR_VERSION = {
        1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
        7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 7,
        13: 7
    }

OPSET_ML_TO_OPSET = {1: 11, 2: 13}


class Variable:
    """
    Defines a variable which holds any data defined
    from *ONNX* types.
    """

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
        self.raw_name = raw_name  #
        self.onnx_name = onnx_name  #
        self.scope = scope
        self.type = type
        # The following fields are bool variables used in parsing and
        # compiling stages
        self.is_fed = None
        self.is_root = None
        self.is_leaf = None
        self.is_abandoned = False
        if self.type is not None and not isinstance(self.type, DataType):
            raise TypeError("shape must be a DataType not {}.".format(
                self.type))
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

    @property
    def full_name(self):
        """
        Return a globally unique variable ID
        """
        return self.onnx_name

    def __repr__(self):
        return ("Variable(raw_name='{0}', onnx_name='{1}', type={2})".format(
                self.raw_name, self.onnx_name, self.type))

    @staticmethod
    def from_pb(obj):
        """
        Creates a data type from a protobuf object.
        """
        def get_shape(tt):
            return [tt.shape.dim[i].dim_value
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


class Operator(OperatorBase):
    """
    Defines an operator available in *ONNX*.
    """

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
        self.inputs = []
        self.outputs = []
        self.is_evaluated = None
        self.is_abandoned = False
        self.target_opset = target_opset
        self.scope_inst = scope_inst

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
        self.variables = {}

        # A map of local operators defined in this scope.
        # (key, value) = (onnx_name, operator)
        self.operators = {}

        # Additional options given to converters.
        self.options = options

        # Registered models
        self.registered_models = registered_models

        # Reserved variables.
        self.reserved = {}

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

    def declare_local_variable(self, raw_name, type=None, prepend=False):
        """
        This function may create a new variable in this scope. If
        *raw_name* has been used to create other variables, the new
        variable will hide all other variables created using *raw_name*.
        """
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

    def reserve_name(self, raw_name):
        """
        Keeps this name to be used by other converters.
        """
        if raw_name in self.reserved:
            raise RuntimeError(
                "Name '{}' already reserved.".format(raw_name))
        self.reserved[raw_name] = self.get_unique_variable_name(raw_name)
        return raw_name

    def unreserve_name(self, name):
        """
        Deletes a name from the reserved list.
        """
        if name not in self.reserved:
            raise RuntimeError(
                "Name '{}' not reserved.".format(name))
        self.onnx_variable_names.discard(name)
        del self.reserved[name]

    def declare_local_operator(self, type, raw_model=None):
        """
        This function is used to declare new local operator.
        """
        onnx_name = self.get_unique_operator_name(str(type))
        operator = Operator(onnx_name, self.name, type, raw_model,
                            self.target_opset, scope_inst=self)
        self.operators[onnx_name] = operator
        return operator

    def delete_local_operator(self, onnx_name):
        """
        Removes the operator whose onnx_name is the input *onnx_name*.
        """
        if (onnx_name not in self.onnx_operator_names or
                onnx_name not in self.operators):
            raise RuntimeError('The operator to remove was not found.')
        self.onnx_operator_names.discard(onnx_name)
        del self.operators[onnx_name]

    def delete_local_variable(self, onnx_name):
        """
        Removes the variable whose *onnx_name* is the input *onnx_name*.
        """
        if (onnx_name not in self.onnx_variable_names or
                onnx_name not in self.variables):
            raise RuntimeError('The variable to remove was not found.')
        self.onnx_variable_names.discard(onnx_name)
        raw_name = self.variables[onnx_name].raw_name
        self.variable_name_mapping[raw_name].remove(onnx_name)
        del self.variables[onnx_name]

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


class Topology:
    """
    Holds instances on :class:`Scope <skl2onnx.common._topology.Scope>` and
    :class:`SklearnModelContainer <skl2onnx.common._container.SklearnModelContainer>`.
    These are filled by the converters while a pipeline is being converted.
    When all converters were called, method
    :meth:`Topology.compile <skl2onnx.common._topology.Topology.compile>`
    must be called to convert the topological graph into *ONNX* graph.
    """ # noqa

    def __init__(self, model, default_batch_size=1, initial_types=None,
                 reserved_variable_names=None, reserved_operator_names=None,
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
        :param reserved_variable_names: A set of strings which are not
                                        allowed to be used as a variable
                                        name
        :param reserved_operator_names: A set of strings which are not
                                        allowed to be used as a operator
                                        name
        :param custom_conversion_functions: a dictionary for specifying
                                the user customized conversion function
        :param custom_shape_calculators: a dictionary for specifying the
                                        user customized shape calculator
        :param registered_models: registered models
        """
        self.scopes = []
        self.raw_model = model
        self.scope_names = set()
        self.variable_name_set = (
                    reserved_variable_names
                    if reserved_variable_names is not None else set())
        self.operator_name_set = (
                    reserved_operator_names
                    if reserved_operator_names is not None else set())
        self.initial_types = initial_types if initial_types else list()
        self.default_batch_size = default_batch_size
        self.target_opset = target_opset
        self.custom_conversion_functions = (
            custom_conversion_functions if custom_conversion_functions else {})
        self.custom_shape_calculators = (
            custom_shape_calculators if custom_shape_calculators else {})

        # This attribute is used in optimizing the graph structure. If
        # root_names is not empty, only the variables specified will be
        # treated as the roots (i.e., set is_fed to True in the
        # beginning of a graph evaluation) of the graph. Specifying all
        # root variables in this list and leaving it empty are
        # equivalent. This attribute directly affects
        # _initialize_graph_status_for_traversing function and
        # indirectly affects _infer_all_shapes and _prune functions.
        self.root_names = list()

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
        scope = Scope(
            self.get_unique_scope_name(seed), parent_scopes,
            self.variable_name_set, self.operator_name_set, self.target_opset,
            custom_shape_calculators=self.custom_shape_calculators,
            options=options, registered_models=self.registered_models)
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

    def topological_operator_iterator(self):
        """
        This is an iterator of all operators in Topology object.
        Operators may be produced in a topological order. If you want to
        simply go though all operators without considering their
        topological structure, please use another function,
        unordered_operator_iterator.
        """
        self._initialize_graph_status_for_traversing()
        priorities = {
            'tensorToProbabilityMap': 2,
            'tensorToLabel': 1
        }
        while not all(operator.is_evaluated for scope in self.scopes
                      for operator in scope.operators.values()):
            is_evaluation_happened = False
            for operator in sorted(self.unordered_operator_iterator(),
                                   key=lambda op: priorities[op.type]
                                   if op.type in priorities else 0):
                if not isinstance(operator.inputs, list):
                    raise TypeError(
                        "operator.inputs must be a list not {}".format(
                            type(operator.inputs)))
                if (all(variable.is_fed for variable in operator.inputs)
                        and not operator.is_evaluated):
                    # Check if over-writing problem occurs (i.e., multiple
                    # operators produce results on one variable).
                    for variable in operator.outputs:
                        # Throw an error if this variable has been treated as
                        # an output somewhere
                        if variable.is_fed:
                            raise RuntimeError(
                                "A variable is already assigned ({}) "
                                "for operator '{}' (name='{}'). This "
                                "may still happen if a converter is a "
                                "combination of sub-operators and one of "
                                "of them is producing this output. "
                                "In that case, an identity node must be "
                                "added.".format(
                                    variable, operator.type,
                                    operator.onnx_name))
                        # Mark this variable as filled
                        variable.is_fed = True
                    # Make this operator as handled
                    operator.is_evaluated = True
                    is_evaluation_happened = True

                    # Send out an operator
                    yield operator

                    # This step may create new nodes if the
                    # the converter is called while looping on
                    # the nodes. The outputs of an operator
                    # are not necessary the inputs of the next
                    # one and but can processed by other ONNX nodes
                    # inserted in the container. As a result, some
                    # variables never have is_fed set to True which
                    # is updated now unless they are an operator
                    # output.
                    known_outputs = {}
                    for op in self.unordered_operator_iterator():
                        for out in op.outputs:
                            if hasattr(out, 'onnx_name'):
                                known_outputs[out.onnx_name] = out
                            else:
                                known_outputs[out] = out
                    for variable in self.unordered_variable_iterator():
                        if variable.is_fed:
                            continue
                        if variable.onnx_name in known_outputs:
                            continue
                        update = (False if self.root_names and
                                  variable.onnx_name not in self.root_names
                                  else True)
                        if update:
                            variable.is_fed = True
                            is_evaluation_happened = True

            # After scanning through the whole computational graph, at
            # least one operator should be evaluated. If not, we need
            # to terminate this procedure to avoid dead lock.
            if not is_evaluation_happened:
                break

    def _check_structure(self):
        """
        This function applies some rules to check if the parsed model is
        proper. Currently, it only checks if isolated variable and
        isolated operator exists.
        """
        # Collect all variable names and operator names
        unused_variables = set()
        unused_operators = set()
        for variable in self.unordered_variable_iterator():
            unused_variables.add(variable.full_name)
        for operator in self.unordered_operator_iterator():
            unused_operators.add(operator.full_name)

        for operator in self.unordered_operator_iterator():
            for variable in operator.inputs:
                # A variable is used by an operator, so we remove the
                # variable from the unused-variable list.
                unused_variables.discard(variable.full_name)
                # A operator has an input, so we remove the operator
                # from the unused-operator list.
                unused_operators.discard(operator.full_name)
            for variable in operator.outputs:
                # A variable is used by an operator, so we remove the
                # variable from the unused-variable list.
                unused_variables.discard(variable.full_name)
                # A operator has an output, so we remove the operator
                # from the unused-operator list.
                unused_operators.discard(operator.full_name)

        if len(unused_variables) > 0:
            raise RuntimeError('Isolated variables exist: %s'
                               % unused_variables)

        if len(unused_operators) > 0:
            raise RuntimeError('Isolated operators exist: %s'
                               % unused_operators)

    def _initialize_graph_status_for_traversing(self):
        """
        Initialize the status of all variables and operators for
        traversing the underline graph
        """
        # In the beginning, we set is_root and is_leaf true. For is_fed,
        # we have two different behaviors depending on whether
        # root_names is empty.
        for variable in self.unordered_variable_iterator():
            # If root_names is set, we only set those variable to be
            # fed. Otherwise, all roots would be fed.
            variable.is_fed = (False if self.root_names and variable.onnx_name
                               not in self.root_names else True)
            variable.is_root = True
            variable.is_leaf = True

        # Then, we flip some flags by applying some simple rules so
        # that only
        #   1. all roots get is_root=True and is_fed=True
        #   2. all leaves get is_leaf=True
        for operator in self.unordered_operator_iterator():
            # All operators are not processed in the beginning
            operator.is_evaluated = False
            for variable in operator.outputs:
                # Output cannot be fed before graph traversing
                variable.is_fed = False
                # If the variable is an output of one operator,
                # it must not be a root
                variable.is_root = False
            for variable in operator.inputs:
                # If the variable is an input of one operator,
                # it must not be a leaf
                variable.is_leaf = False

    def _infer_all_types(self):
        """
        Infer all variables' shapes in the computational graph.
        """
        self._initialize_graph_status_for_traversing()

        # Deliver user-specified types to root variables
        for raw_name, initial_type in self.initial_types:
            # Check all variables declared using raw_name in
            # the whole graph
            for scope in self.scopes:
                # Skip scopes without having the considered variable
                # name
                if raw_name not in scope.variable_name_mapping:
                    continue
                # Assign initial_type to all variables declared using
                # raw_name
                for onnx_name in scope.variable_name_mapping[raw_name]:
                    variable = scope.variables[onnx_name]
                    if variable.is_root:
                        # Assign type to the root; existing type
                        # produced by parser may be overwritten
                        variable.type = initial_type

        # Traverse the graph from roots to leaves
        for operator in self.topological_operator_iterator():
            mtype = type(operator.raw_operator)
            if mtype in self.custom_shape_calculators:
                # overwritten operator.
                self.custom_shape_calculators[mtype](operator)
            elif operator.type in self.custom_shape_calculators:
                self.custom_shape_calculators[operator.type](operator)
            elif hasattr(operator.raw_operator, "onnx_shape_calculator"):
                shape_calc = operator.raw_operator.onnx_shape_calculator()
                shape_calc(operator)
            else:
                operator.infer_types()

    def _resolve_duplicates(self):
        """
        Merge variables connected by identity operator to reduce the
        number of redundant variables
        """
        self._initialize_graph_status_for_traversing()

        # Traverse the graph from roots to leaves
        for operator in self.topological_operator_iterator():
            if operator.type != 'identity':
                continue

            if (any(variable.is_root for variable in operator.inputs) and
                    any(variable.is_leaf for variable in operator.outputs)):
                continue

            # Replace the output variable with the input variable everywhere
            original = operator.inputs[0]
            duplicate = operator.outputs[0]
            for another_scope in self.scopes:
                for another_operator in another_scope.operators.values():
                    for i in range(len(another_operator.inputs)):
                        if (another_operator.inputs[i].onnx_name
                                != duplicate.onnx_name):
                            continue
                        another_operator.inputs[i] = original

            # When original variable's documentation string or
            # denotation is empty but duplicate's is not, we copy that
            # field to the original variable to avoid information loss.
            if not original.type.doc_string and duplicate.type.doc_string:
                original.type.doc_string = duplicate.type.doc_string

            if (isinstance(original.type, TensorType) and
                    isinstance(duplicate.type, TensorType)):
                if not original.type.denotation and duplicate.type.denotation:
                    original.type.denotation = duplicate.type.denotation
                if not original.type.channel_denotations:
                    original.type.channel_denotations = (
                        duplicate.type.channel_denotations)
                elif duplicate.type.channel_denotations:
                    # Merge the channel denotations if available in both
                    # the original and the duplicate
                    for i in range(len(original.type.channel_denotations)):
                        if original.type.channel_denotations[i]:
                            continue
                        original.type.channel_denotations[i] = (
                            duplicate.type.channel_denotations[i])
                # Sometime, shapes of duplicates are different. We try
                # to replace the original variable's unknown dimensions
                # as many as possible because we will get rid of the
                # duplicate.
                if len(original.type.shape) == len(duplicate.type.shape):
                    for i in range(len(original.type.shape)):
                        if original.type.shape[i] is not None:
                            continue
                        original.type.shape[i] = duplicate.type.shape[i]

            # Because we're iterating through the topology, we cannot
            # delete any operator or variable. Otherwise, the traversing
            # function may be broken. We will delete those abandoned
            # ones later.
            duplicate.is_abandoned = True
            operator.is_abandoned = True

        for scope in self.scopes:
            # Find out who is going to be abandoned
            abandoned_operator_names = set(
                onnx_name for onnx_name, operator in scope.operators.items()
                if operator.is_abandoned)
            abandoned_variable_names = set(
                onnx_name for onnx_name, variable in scope.variables.items()
                if variable.is_abandoned)

            # Remove abandoned operators
            for name in abandoned_operator_names:
                scope.delete_local_operator(name)

            # Remove abandoned variables
            for name in abandoned_variable_names:
                scope.delete_local_variable(name)

    def _fix_shapes(self):
        """
        This function applies some rules to adjust graph inputs
        (i.e., roots) before doing shape inference
        """

        # Identify roots of a graph
        self._initialize_graph_status_for_traversing()

        # Scan through all operators and adjust their variables' shapes
        # if needed
        for operator in self.unordered_operator_iterator():
            # Rule 1 (CoreML):
            # Some operator in CoreML only accepts 4-D tensors but
            # their protobuf models might specify a 2-D one.
            # We fix this problem here.
            if operator.type in [
                    'bias', 'concat', 'convolution', 'crop', 'flatten',
                    'scalerPreprocessor', 'lrn', 'meanImagePreprocessor',
                    'padding', 'permute', 'pooling', 'reduce',
                    'reorganizeData', 'reshape', 'scale', 'slice', 'upsample']:
                # We only adjust inputs because outputs will be
                # automatically fixed at our shape inference stage
                for variable in operator.inputs:
                    if variable.is_root:
                        # Convert [N, C] to [N, C, 1, 1] while
                        # [N, C, H, W] is unchanged
                        variable.type.shape += [1] * (
                            4 - len(variable.type.shape))

    def _prune(self):
        # Conduct a dummy evaluation of this topology. It may set all
        # reachable operators evaluated and all reachable variables fed.
        for operator in self.topological_operator_iterator():
            pass

        for scope in self.scopes:
            # Remove unused operators
            abandoned_operator_names = []
            for operator in scope.operators.values():
                if not operator.is_evaluated:
                    abandoned_operator_names.append(operator.onnx_name)
            for onnx_name in abandoned_operator_names:
                scope.delete_local_operator(onnx_name)

            # Remove unused variables
            abandoned_variable_names = []
            for variable in scope.variables.values():
                if not variable.is_fed:
                    abandoned_variable_names.append(variable.onnx_name)
            for onnx_name in abandoned_variable_names:
                scope.delete_local_variable(onnx_name)

    def compile(self):
        """
        This function aims at giving every operator enough information
        so that all operator conversions can happen independently. We
        also want to check, fix, and simplify the network structure
        here.
        """
        self._prune()
        self._resolve_duplicates()
        self._fix_shapes()
        self._infer_all_types()
        self._check_structure()


def convert_topology(topology, model_name, doc_string, target_opset,
                     channel_first_inputs=None,
                     options=None, remove_identity=True):
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

    topology._initialize_graph_status_for_traversing()

    container = ModelComponentContainer(
        target_opset, options=options,
        registered_models=topology.registered_models,
        white_op=topology.raw_model._white_op,
        black_op=topology.raw_model._black_op)

    # Put roots and leaves as ONNX's model into buffers. They will be
    # added into ModelComponentContainer later.
    tensor_inputs = {}
    other_inputs = {}
    tensor_outputs = {}
    other_outputs = {}
    for scope in topology.scopes:
        for variable in scope.variables.values():
            if variable.is_root:
                if isinstance(variable.type, (TensorType, Int64Type,
                                              FloatType, StringType)):
                    tensor_inputs[variable.raw_name] = variable
                else:
                    other_inputs[variable.raw_name] = variable
            if variable.is_leaf:
                if isinstance(variable.type, (TensorType, Int64Type,
                                              FloatType, StringType)):
                    tensor_outputs[variable.onnx_name] = variable
                else:
                    other_outputs[variable.onnx_name] = variable

    # Add roots the graph according to their order in the original model
    invalid_name = []
    nhwc_inputs = []
    if channel_first_inputs is None:
        channel_first_inputs = []
    for name in topology.raw_model.input_names:
        # Check input naming convention
        input_name = name.replace('_', '').replace(":", "").replace("/", "")
        if input_name and (input_name[0].isdigit() or
                           (not input_name.isalnum())):
            invalid_name.append(name)
        if name in tensor_inputs:
            # type: Variable
            onnx_input = tensor_inputs[name]
            if (name in channel_first_inputs or
                    (name.endswith(':0') and
                     name[:-2] in channel_first_inputs)):
                nhwc_inputs.append(onnx_input.full_name)
                s = onnx_input.type.shape
                onnx_input.type.shape = [s[0], s[3], s[1], s[2]]
            container.add_input(onnx_input)

    if invalid_name:
        warnings.warn('Some input names are not compliant with ONNX naming '
                      'convention: %s' % invalid_name)
    for name in topology.raw_model.input_names:
        if name in other_inputs:
            container.add_input(other_inputs[name])

    # Add leaves the graph according to their order in
    # the original model
    invalid_name = []
    for name in topology.raw_model.output_names:
        # Check output naming convention
        output_name = name.replace('_', '').replace(":", "").replace("/", "")
        if output_name and (output_name[0].isdigit() or
                            (not output_name.isalnum())):
            invalid_name.append(name)
        if name in tensor_outputs:
            container.add_output(tensor_outputs[name])
    if invalid_name:
        warnings.warn('Some output names are not compliant with ONNX naming '
                      'convention: %s' % invalid_name)
    for name in topology.raw_model.output_names:
        if name in other_outputs:
            container.add_output(other_outputs[name])

    # Traverse the graph from roots to leaves
    # This loop could eventually be parallelized.
    for operator in topology.topological_operator_iterator():
        scope = next(scope for scope in topology.scopes
                     if scope.name == operator.scope)
        mtype = type(operator.raw_operator)
        if mtype in topology.custom_conversion_functions:
            conv = topology.custom_conversion_functions[mtype]
        elif operator.type in topology.custom_conversion_functions:
            conv = topology.custom_conversion_functions[operator.type]
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
        conv(scope, operator, container)

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
