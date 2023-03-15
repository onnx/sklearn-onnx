# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
import numpy as np
from scipy.sparse import coo_matrix
from onnx import GraphProto
from ..proto import onnx_proto, TensorProto
from ..common.data_types import (
    guess_proto_type, _guess_numpy_type, _guess_type_proto_str,
    _guess_type_proto, FloatType, DoubleType, Int64Type, copy_type)
from ..common._topology import Variable
from ..common._registration import get_shape_calculator, get_converter


logger = getLogger('skl2onnx')


class GraphStateVar:
    pass


class GraphState:

    def __init__(self, inputs, output_names, operator_name, scope,
                 container, converter, onnx_prefix_name=None,
                 options=None, expected_inputs=None,
                 expected_outputs=None, input_range=None,
                 output_range=None, operator=None,
                 run_converters=False, input_types=None, **attrs):

        logger.debug(
            "[State] +%s n_inputs=%r n_outputs=%r",
            operator_name, -1 if inputs is None else len(inputs),
            -1 if output_names is None else len(output_names))
        self.inputs = inputs
        self._output_names = output_names
        self._input_range = input_range.copy() if input_range else [1, 1e9]
        self._output_range = output_range.copy() if output_range else [1, 1e9]
        self.scope = scope
        self.run_converters = run_converters
        self.operator = operator
        if hasattr(operator_name, 'fit'):
            from .. import get_model_alias
            self.operator_instance = operator_name
            self.is_model = True
            self.operator_name = get_model_alias(type(operator_name))
        elif operator_name.__class__.__name__ == "WrappedModelAlias":
            self.operator_instance = operator_name.model
            self.is_model = True
            self.operator_name = operator_name.alias
        else:
            self.operator_name = operator_name
            self.is_model = False
        self.container = container
        self.converter = converter
        self._expected_inputs = (
            None if expected_inputs is None else expected_inputs.copy())
        self._expected_outputs = (
            None if expected_outputs is None else expected_outputs.copy())
        self.computed_inputs_ = None
        self.computed_outputs_ = None
        self.sub_op_ = None
        self.onnx_prefix_name = onnx_prefix_name
        self.attrs = attrs
        self.options = options
        self.input_types = input_types

        for att in ['inputs', '_expected_inputs',
                    '_expected_outputs', 'computed_inputs_',
                    'computed_outputs_', '_outputs']:
            v = getattr(self, att, None)
            if v is None:
                continue
            if not isinstance(v, list):
                raise TypeError(
                    "Attribute %r must be a list not %r."
                    "" % (att, type(v)))
            for i, vi in enumerate(v):
                if hasattr(vi, 'state') or hasattr(vi, 'onx_op'):
                    continue
                if not isinstance(vi, (tuple, str, Variable, GraphStateVar)):
                    raise TypeError(
                        "Unexpected type %r for element %d of attribute %r "
                        "in %r." % (type(vi), i, att, v))
                if isinstance(vi, tuple) and len(vi) != 2:
                    raise ValueError(
                        "Unexpected value %r for element %d of attribute %r."
                        "" % (vi, i, att))
            change = []
            for vi in v:
                change.append((vi, None) if isinstance(vi, str) else vi)

        if self._output_names is not None:
            res = []
            if self._expected_outputs is not None:
                for i in range(0, len(self._expected_outputs)):
                    if i < len(self._output_names):
                        res.append(
                            (self._output_names[i],
                             self._expected_outputs[i][1]))
                    else:
                        res.append(self._expected_outputs[i])
            for i in range(len(res), len(self._output_names)):
                res.append((self._output_names[i], None))
            self._expected_outputs = res

        if self._expected_outputs is not None:
            res = []
            for p in self._expected_outputs:
                if isinstance(p[1], str) and p[1].startswith('tensor('):
                    res.append((p[0], _guess_type_proto_str(p[1], None)))
                else:
                    res.append(p)
            self._expected_outputs = res

        if self._expected_inputs is not None:
            res = []
            for p in self._expected_inputs:
                if isinstance(p[1], str) and p[1].startswith('tensor('):
                    res.append((p[0], _guess_type_proto_str(p[1], None)))
                else:
                    res.append(p)
            self._expected_inputs = res

    @property
    def onnx_prefix(self):
        if self.onnx_prefix_name is None:
            return self.operator_name
        return self.onnx_prefix_name + "_" + self.operator_name

    @property
    def outputs(self):
        self.run()
        return self.computed_outputs_

    def _get_var_name(self, var, in_out, operator=None, index=None):
        "input: True for output, False for input"
        if hasattr(var, 'add_to'):
            var.add_to(self.scope, self.container, operator=operator,
                       run_converters=self.run_converters)
            outputs = var.outputs
            if isinstance(outputs, list):
                vars = []
                for var in outputs:
                    if isinstance(var, (Variable, tuple)):
                        vars.append(var)
                    elif isinstance(var, str):
                        vars.append((var, None))
                if len(vars) == 0:
                    raise RuntimeError(
                        "Empty inputs outputs=%s var=%s in_out=%s "
                        "operator=%r." % (outputs, var, in_out, operator))
                return vars
            raise RuntimeError("Unexpected output type {}".format(outputs))

        def __fct__(var, operator):
            if isinstance(var, Variable):
                return [var]
            if isinstance(var, (np.ndarray, np.bool_, np.int64,
                                np.float32, np.float64,
                                np.int8, np.uint8)):
                return [self._add_constant(var)]
            if hasattr(var, 'ConstantValue'):
                return [
                    self._add_constant(var.ConstantValue, scope=self.scope)]
            if isinstance(var, str):
                return [(var, None)]
            if isinstance(var, tuple) and len(var) == 2:
                return [var]
            try:
                a, b = var
                return [(a, b)]
            except ValueError:
                pass
            raise RuntimeError("Unexpected type for parameter 'var': {0}."
                               "".format(type(var)))

        try:
            v = __fct__(var, operator)
        except TypeError as e:
            raise RuntimeError(
                "Unable to process one variable %s and operator=%s "
                "(name=%r)." % (var, operator, self.operator_name)) from e
        if v is None or not isinstance(v, list) or len(v) == 0:
            raise TypeError(
                "Unexpected type or empty value %r - %s." % (type(v), v))
        if in_out and self._output_names is not None and index is not None:
            if len(v) != 1:
                raise RuntimeError(
                    "Mismatch number of outputs between %s and %s." % (
                        v, self._output_names[index]))
            v2 = self.scope.get(var[0], None)
            if v2 is not None:
                v = [v2]
            try:
                vn = v[0][0]
            except IndexError as e:
                raise ValueError(
                    "Unexpected output %s in operator name %r."
                    "" % (vn, self.operator_name)) from e
            if (index >= len(self._output_names) and
                    index >= self._output_range[0]):
                return None
            try:
                vin = self._output_names[index]
            except IndexError as e:
                raise ValueError(
                    "Unexpected index %s in operator name %r with ."
                    "output names %s." % (
                        index, self.operator_name,
                        self._output_names)) from e
            if vn != vin:
                raise RuntimeError(
                    "Mismatched output name %r between %s and %s." % (
                        vn, v, vin))
        return v

    def _add_constant(self, cst, scope):

        def _ty_astype(cst):
            astype = cst.dtype
            try:
                ty = guess_proto_type(_guess_numpy_type(cst.dtype, cst.shape))
            except NotImplementedError as e:
                st = str(astype).lower()
                if st.startswith('u') or st.startswith("<u"):
                    ty = onnx_proto.TensorProto.STRING
                    astype = None
                    cst = np.array([s.encode('utf-8') for s in cst])
                else:
                    raise NotImplementedError(
                        "Unable to guess ONNX type from type {}. "
                        "You may raise an issue at https://github.com/onnx/"
                        "sklearn-onnx/issues.".format(
                            cst.dtype)) from e
            return cst, ty, astype

        if isinstance(cst, np.ndarray):
            shape = cst.shape
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            cst, ty, astype = _ty_astype(cst)
            if astype is not None:
                cst = cst.astype(astype)
            if ty == onnx_proto.TensorProto.STRING:
                value = [s.encode('utf-8') for s in cst.flatten()]
            else:
                value = cst.flatten()
            self.container.add_initializer(
                name, ty, shape, value)
            return (name, _guess_numpy_type(cst.dtype, cst.shape))

        if isinstance(cst, coo_matrix):
            shape = cst.shape
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            cst, ty, astype = _ty_astype(cst)
            self.container.add_initializer(
                name, ty, shape, cst.astype(astype))
            return (name, _guess_numpy_type(cst.dtype, cst.shape))

        if isinstance(cst, TensorProto):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            self.container.add_initializer(name, None, None, cst)
            return (name, _guess_type_proto(cst, None))

        if isinstance(cst, np.int64):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.INT64
            self.container.add_initializer(name, ty, None, cst)
            return (name, Int64Type())

        if isinstance(cst, np.float32):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.FLOAT
            self.container.add_initializer(name, ty, None, float(cst))
            return (name, FloatType())

        if isinstance(cst, np.float64):
            name = self.scope.get_unique_variable_name(
                self.onnx_prefix + 'cst')
            ty = TensorProto.DOUBLE
            self.container.add_initializer(name, ty, None, float(cst))
            return (name, DoubleType())

        raise NotImplementedError(
            "Unable to add a constant of type {}. "
            "You may raise an issue at https://github.com/onnx/"
            "sklearn-onnx/issues.".format(type(cst)))

    @staticmethod
    def _get_output_name(output_names, output, scope):
        if isinstance(output, Variable):
            return output
        if isinstance(output, str):
            if output in output_names:
                return (output, None)
            return (scope.get_unique_variable_name(output), None)
        if isinstance(output, tuple):
            if output[0] in output_names:
                return output
            return (scope.get_unique_variable_name(output[0]), output[1])
        raise NotImplementedError(
            "Unexpected output type {} [{}]. "
            "You may raise an issue at https://github.com/onnx/"
            "sklearn-onnx/issues.".format(type(output), output))

    @staticmethod
    def _update_inputs(inputs, names, scope, expected_inputs,
                       input_range, input_types=None):
        new_inputs = []
        for inp in inputs:
            if isinstance(inp, (Variable, tuple, GraphStateVar)):
                new_inputs.append(inp)
                continue
            if hasattr(inp, 'get_output_type_inference'):
                etype = inp.get_output_type_inference(inputs)
                new_inputs.extend(etype)
                continue
            raise TypeError(
                "Unable to infer shape of inputs %r (type is %r)"
                "." % (inp, type(inp)))

        for i in range(0, len(new_inputs)):
            inp = new_inputs[i]
            if isinstance(inp, tuple) and len(inp) == 2:
                if input_types is not None and i < len(input_types):
                    stype = input_types[i]
                else:
                    stype = None if isinstance(inp[1], str) else inp[1]
                if scope is not None:
                    if inp[0] in scope.variables:
                        var = scope.variables[inp[0]]
                        if stype is not None:
                            var.check_compatible_type(stype)
                    else:
                        onnx_name = scope.get_unique_variable_name(inp[0])
                        var = Variable(
                            inp[0], onnx_name, type=stype, scope=scope)
                        scope.register_variable(var)
                else:
                    var = Variable(inp[0], inp[0], type=stype, scope=scope)
                new_inputs[i] = var
                inp = new_inputs[i]
            elif isinstance(inp, GraphStateVar):
                new_inputs[i] = inp.as_variable(scope)
                inp = new_inputs[i]
            elif not isinstance(inp, Variable):
                raise TypeError(
                    "Inputs %d - %r must be of type Variable." % (i, inp))

            if names is not None:
                try:
                    onnx_name = (
                        names[i] if isinstance(names[i], str)
                        else names[i][0])
                except IndexError as e:
                    raise IndexError(
                        "Wrong index %d, list=%s." % (i, names)) from e
                inp.set_onnx_name(onnx_name)

        # Second pass.
        if expected_inputs is not None:
            memo = {}
            for i, (name, ct) in enumerate(expected_inputs):
                if ct in memo:
                    memo[ct].append(i)
                else:
                    memo[ct] = [i]
            for i in range(0, len(new_inputs)):
                inp = new_inputs[i]
                if inp.type is None:
                    ct = expected_inputs[i][1]
                    if ct in memo:
                        for j in memo[ct]:
                            if (j >= len(new_inputs) and
                                    j >= input_range[0]):
                                continue
                            if new_inputs[j].type is not None:
                                new_inputs[i].set_type(
                                    new_inputs[j].type.__class__())
                                break

        # Overwrite types if input_types is specified.
        if input_types is not None:
            for i in range(len(new_inputs)):
                if i >= len(input_types):
                    raise RuntimeError(
                        "Mismatch between computed inputs[%d]=%r and "
                        "overwritten input_types[%d]=%r." % (
                            i, new_inputs, i, input_types))
                if input_types[i] is not None:
                    new_inputs[i].type = input_types[i]
        return new_inputs

    @staticmethod
    def _update_contraints(vars1, expected1, vars2, expected2, debug=None):
        memo = {}
        for va, ex in [(vars1, expected1), (vars2, expected2)]:
            if va is None or ex is None:
                continue
            for v, ct in zip(va, ex):
                if (isinstance(v, str) or (
                        hasattr(v, 'type') and v.type is None)):
                    continue
                vt = (copy_type(v.type)
                      if hasattr(v, 'type') else copy_type(v[1]))
                if isinstance(vt, str):
                    continue
                key = ct[1]
                if isinstance(key, str) and key[0] in ('T', 'I', 'V'):
                    if not isinstance(vt, str) and key not in memo:
                        memo[key] = []
                    memo[key].append(vt)

        for k, v in memo.items():
            if len(set(_.__class__ for _ in v)) != 1:
                raise RuntimeError(
                    "Conflicted constraint %r, got types %r operator=%s"
                    "." % (k, v, debug))
        for i in range(0, len(vars1)):
            inp = vars1[i]
            if isinstance(inp, str):
                continue
            if hasattr(inp, 'type') and inp.type is None:
                ct = expected1[i][1]
                if ct in memo:
                    vars1[i].set_type(copy_type(memo[ct][0]))
            elif isinstance(inp, tuple):
                ct = expected1[i][1]
                if ct in memo:
                    vars1[i] = (inp[0], copy_type(memo[ct][0]))

    def run(self):
        if self.computed_outputs_ is None:

            # We need to register all names in subgraphs and raise
            # an exception if the names are already taken.
            for k, v in self.attrs.items():
                if isinstance(v, GraphProto):
                    try:
                        self.scope.declare_existing_subgraph_name(v)
                    except NameError as e:
                        raise RuntimeError(
                            "A name exists both in the subgraph and "
                            "in the main graph. Use set_onnx_name_prefix to "
                            "to rename one of them, attribute=%r, "
                            "op_type=%r." % (
                                k, self.operator_name)) from e

            if self.operator is not None:
                expected_outputs = self.operator.outputs
            else:
                if self._expected_outputs is not None:
                    eoli = []
                    for i, o in enumerate(self._expected_outputs):
                        v = self._get_var_name(o, True, index=i)
                        if v is None:
                            continue
                        eoli.extend(v)
                    expected_outputs = eoli
                else:
                    expected_outputs = None

            logger.debug(
                "[State.run] id=%d op_name=%r is_model=%r "
                "expected_outputs=%r",
                id(self), self.operator_name, self.is_model, expected_outputs)

            inputs = []
            for i in self.inputs:
                v = self._get_var_name(i, False, index=None)
                inputs.extend(v)

            self.computed_inputs_ = GraphState._update_inputs(
                self.inputs, inputs, scope=self.scope,
                expected_inputs=self._expected_inputs,
                input_range=self._input_range,
                input_types=self.input_types)

            logger.debug(
                "[State.run] id=%d op_name=%r computed_inputs_=%r",
                id(self), self.operator_name, self.computed_inputs_)

            name = self.scope.get_unique_operator_name(self.onnx_prefix)
            if self.is_model:
                if self.sub_op_ is not None:
                    raise NotImplementedError(
                        "Attribute 'sub_op_' is not empty.")

                # a model is converted into a subgraph
                sub_op_inputs = self.computed_inputs_
                for v in sub_op_inputs:
                    if not isinstance(v, Variable):
                        raise TypeError(
                            "Every input variable must be a Variable not %r,"
                            " v=%r." % (type(v), v))
                    scope = v.scope
                    if hasattr(scope, 'variables'):
                        if v.onnx_name not in scope.variables:
                            raise RuntimeError(
                                "Variable %r missing from scope "
                                "(operator=%r, model=%r), list=%r." % (
                                    v, self.operator,
                                    type(self.operator_instance),
                                    list(sorted(self.scope.variables))))

                # output are not defined, we need to call a parser.
                from .._parse import _parse_sklearn
                self.scope.add_options(
                    id(self.operator_instance), self.options)
                try:
                    sub_outputs = _parse_sklearn(
                        self.scope, self.operator_instance, sub_op_inputs,
                        alias=self.operator_name)
                except RuntimeError as e:
                    raise RuntimeError(
                        "Unable to run parser for model type %r, inputs=%r "
                        "(input_types=%r)." % (
                            type(self.operator_instance), sub_op_inputs,
                            self.input_types)) from e
                set_input_names = set(v.onnx_name for v in sub_op_inputs)
                sub_op = None
                for op in self.scope.operators.values():
                    for inp in op.inputs:
                        if inp.onnx_name in set_input_names:
                            sub_op = op
                if (sub_outputs is None or
                        None in sub_outputs):
                    raise RuntimeError(
                        "Wrong result when parsing model {}.".format(
                            type(self.operator_instance)))

                # Checks operator outputs
                for out in sub_outputs:
                    if not isinstance(out, Variable):
                        raise TypeError(
                            "Output %s must be of type Variable." % out)
                self.sub_op_ = sub_op
                sub_op.outputs = sub_outputs

                shape_calc = get_shape_calculator(self.operator_name)
                logger.debug(
                    "[StateShape] call %r fed %r - %r", sub_op,
                    "".join(str(i.is_fed) for i in sub_op.inputs),
                    "".join(str(i.is_fed) for i in sub_op.outputs))
                shape_calc(sub_op)
                logger.debug("[StateShape] end - %r", sub_op)

                # Add Identity nodes to be consistent with `is_fed`
                # in Topology.
                if sub_op.outputs is not None and len(sub_op.outputs) > 0:
                    outputs = [
                        self.scope.declare_local_variable(
                            o.onnx_name, type=o.type)
                        for o in sub_op.outputs]
                elif (expected_outputs is not None and
                        len(expected_outputs) > 0):
                    outputs = [
                        self._get_output_name(
                            self._output_names, o, self.scope)
                        for o in expected_outputs]
                else:
                    raise RuntimeError(
                        "sub_op.outputs is None as well as expected_outputs "
                        "for operator %r." % sub_op)

                if len(outputs) != len(sub_op.outputs):
                    raise RuntimeError(
                        "Mismatched number of outputs %s and %s." % (
                            outputs, sub_op.outputs))

                for i, out in enumerate(sub_op.outputs):
                    var = outputs[i]
                    self.container.add_node(
                        'Identity', [out.onnx_name], [var[0]],
                        name=self.scope.get_unique_operator_name("SubOpId"))
                self.computed_outputs_ = outputs
                self.computed_inputs2_ = sub_op.inputs
                self.computed_outputs2_ = [
                    (v[0], v[1]) for v in self.computed_outputs_]

                if self.run_converters:
                    # The parser was run on sub-operators but not the
                    # converter.
                    conv = get_converter(self.operator_name)
                    logger.debug(
                        "[StateConv] %r fed %r - %r", sub_op,
                        "".join(str(i.is_fed) for i in sub_op.inputs),
                        "".join(str(i.is_fed) for i in sub_op.outputs))
                    conv(self.scope, sub_op, self.container)
                    logger.debug("[StateConv] %r - end.", sub_op)
                else:
                    if (expected_outputs is not None and
                            len(sub_op.outputs) == len(expected_outputs)):
                        for v1, v2 in zip(sub_op.outputs, expected_outputs):
                            if isinstance(v2, tuple):
                                v2 = v2[0]
                            if (hasattr(v1, 'onnx_name') and
                                    hasattr(v2, 'onnx_name')):
                                if v1.onnx_name != v2.onnx_name:
                                    # One identity is missing
                                    n = self.scope.get_unique_operator_name(
                                        'idgstate')
                                    self.container.add_node(
                                        'Identity', [v1.onnx_name],
                                        [v2.onnx_name], name=n)
            else:

                def _name_(obj):
                    if isinstance(obj, tuple) and len(obj) == 2:
                        return obj[0]
                    if hasattr(obj, 'onnx_name'):
                        return obj.onnx_name
                    raise TypeError(
                        "Unable to extract variable name from %r." % obj)

                # only one node is added
                if self.options is not None:
                    raise RuntimeError(
                        "Options must be empty for node %r but is it %r." % (
                            self.operator_name, self.options))
                outputs = [
                    self._get_output_name(self._output_names, o, self.scope)
                    for o in expected_outputs]
                input_names = [_name_(i) for i in inputs]
                output_names = [_name_(i) for i in outputs]
                self.container.add_node(
                    self.operator_name, input_names, output_names,
                    name=name, **self.attrs)

                computed_outputs = [
                    (name, ct[1]) for name, ct in zip(
                        output_names, self._expected_outputs)]
                self._update_contraints(
                    computed_outputs, self._expected_outputs,
                    self.computed_inputs_, self._expected_inputs,
                    debug=self.operator_name)

                # Registers the variables into scope.
                self.computed_outputs_ = []
                for name, kind in computed_outputs:
                    if isinstance(kind, str):
                        self.computed_outputs_.append((name, kind))
                    else:
                        var = self.scope.declare_local_variable(
                            name, kind, missing_type=True)
                        # name already comes from
                        # scope.get_unique_variable_name
                        var.set_onnx_name(name)
                        var.init_status(is_fed=True)
                        self.computed_outputs_.append(var)

            logger.debug('[State.run] end id=%d', id(self))
