# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union
from onnx import helper, ModelProto, TensorProto, ValueInfoProto, TypeProto

from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def get_tensor_shape(obj: Union[ValueInfoProto, TypeProto]) -> Tuple[int, ...]:
    """
    Returns the shape if that makes sense for this object.
    """
    if isinstance(obj, ValueInfoProto):
        return get_tensor_shape(obj.type)
    if not isinstance(obj, TypeProto):
        raise TypeError(f"Unexpected type {type(obj)!r}.")
    shape = []
    for d in obj.tensor_type.shape.dim:
        v = d.dim_value if d.dim_value > 0 else d.dim_param
        shape.append(v)
    if len(shape) == 0:
        shape = None
    else:
        shape = [None if s == 0 else s for s in shape]
    return shape


def get_tensor_elem_type(obj: Union[ValueInfoProto, TypeProto]) -> int:
    """
    Returns the element type if that makes sense for this object.
    """
    if isinstance(obj, ValueInfoProto):
        return get_tensor_elem_type(obj.type)
    if not isinstance(obj, TypeProto):
        raise TypeError(f"Unexpected type {type(obj)!r}.")
    return obj.tensor_type.elem_type


def _copy_inout(inout, scope, new_name):
    shape = get_tensor_shape(inout)
    elem_type = get_tensor_elem_type(inout)
    value_info = helper.make_tensor_value_info(new_name, elem_type, shape)
    return value_info


def _clean_variable_name(name, scope):
    return scope.get_unique_variable_name(name)


def _clean_operator_name(name, scope):
    return scope.get_unique_operator_name(name)


def _clean_initializer_name(name, scope):
    return scope.get_unique_variable_name(name)


def add_onnx_graph(
    scope: Scope,
    operator: Operator,
    container: ModelComponentContainer,
    onx: ModelProto,
):
    """
    Adds a whole ONNX graph to an existing one following
    :epkg:`skl2onnx` API assuming this ONNX graph implements
    an `operator <http://onnx.ai/sklearn-onnx/api_summary.htmlskl2onnx.common._topology.Operator>`_.

    :param scope: scope (to get unique names)
    :param operator: operator
    :param container: container
    :param onx: ONNX graph
    """
    graph = onx.graph
    name_mapping = {}
    node_mapping = {}
    for node in graph.node:
        name = node.name
        if name is not None:
            node_mapping[node.name] = _clean_initializer_name(node.name, scope)
        for o in node.input:
            name_mapping[o] = _clean_variable_name(o, scope)
        for o in node.output:
            name_mapping[o] = _clean_variable_name(o, scope)
    for o in graph.initializer:
        name_mapping[o.name] = _clean_operator_name(o.name, scope)

    inputs = [_copy_inout(o, scope, name_mapping[o.name]) for o in graph.input]
    outputs = [_copy_inout(o, scope, name_mapping[o.name]) for o in graph.output]

    for inp, to in zip(operator.inputs, inputs):
        n = helper.make_node(
            "Identity",
            [inp.onnx_name],
            [to.name],
            name=_clean_operator_name("Identity", scope),
        )
        container.nodes.append(n)

    for inp, to in zip(outputs, operator.outputs):
        n = helper.make_node(
            "Identity",
            [inp.name],
            [to.onnx_name],
            name=_clean_operator_name("Identity", scope),
        )
        container.nodes.append(n)

    for node in graph.node:
        n = helper.make_node(
            node.op_type,
            [name_mapping[o] for o in node.input],
            [name_mapping[o] for o in node.output],
            name=node_mapping[node.name] if node.name else None,
            domain=node.domain if node.domain else None,
        )
        n.attribute.extend(node.attribute)
        container.nodes.append(n)

    for o in graph.initializer:
        as_str = o.SerializeToString()
        tensor = TensorProto()
        tensor.ParseFromString(as_str)
        tensor.name = name_mapping[o.name]
        container.initializers.append(tensor)

    # opset
    for oimp in onx.opset_import:
        container.node_domain_version_pair_sets.add((oimp.domain, oimp.version))
