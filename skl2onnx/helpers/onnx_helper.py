# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from io import BytesIO
import onnx
from onnx import shape_inference
from ..proto.onnx_helper_modified import (
    make_node, make_tensor_value_info, make_graph,
    make_model, ValueInfoProto
)
from onnx import onnx_pb as onnx_proto
from ..common._topology import Variable


def load_onnx_model(onnx_file_or_bytes):
    """
    Loads an *ONNX* file.

    :param onnx_file_or_bytes: *ONNX* file or bytes
    :return: *ONNX* model
    """
    if isinstance(onnx_file_or_bytes, str):
        with open(onnx_file_or_bytes, "rb") as f:
            return onnx.load(f)
    elif hasattr(onnx_file_or_bytes, 'read'):
        return onnx.load(onnx_file_or_bytes)
    else:
        b = BytesIO(onnx_file_or_bytes)
        return onnx.load(b)


def save_onnx_model(model, filename=None):
    """
    Saves a model as a file or bytes.

    :param model: *ONNX* model
    :param filename: filename or None to return bytes
    :return: bytes
    """
    content = model.SerializeToString()
    if filename is not None:
        if hasattr(filename, 'write'):
            filename.write(content)
        else:
            with open(filename, "wb") as f:
                f.write(content)
    return content


def enumerate_model_node_outputs(model, add_node=False):
    """
    Enumerates all the nodes of a model.

    :param model: ONNX graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :return: enumerator
    """
    if not hasattr(model, "graph"):
        raise TypeError("Parameter model is not an ONNX model but "
                        "{}".format(type(model)))
    for node in model.graph.node:
        for out in node.output:
            yield (out, node) if add_node else out


def enumerate_model_initializers(model, add_node=False):
    """
    Enumerates all the initializers of a model.

    :param model: ONNX graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :return: enumerator
    """
    for node in model.graph.initializer:
        yield (node.name, node) if add_node else node.name


def select_model_inputs_outputs(model, outputs=None, inputs=None):
    """
    Takes a model and changes its outputs.

    :param model: *ONNX* model
    :param inputs: new inputs
    :param outputs: new outputs
    :return: modified model

    The function removes unneeded files.
    """
    if inputs is not None:
        raise NotImplementedError("Parameter inputs cannot be empty.")
    if outputs is None:
        raise RuntimeError("Parameter outputs cannot be None.")
    if not isinstance(outputs, list):
        outputs = [outputs]

    mark_var = {}
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
    for inp in model.graph.input:
        mark_var[inp.name] = 0
    for out in outputs:
        if out not in mark_var:
            raise ValueError("Output '{}' not found in model.".format(out))
        mark_var[out] = 1

    nodes = model.graph.node[::-1]
    mark_op = {}
    for node in nodes:
        mark_op[node.name] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            if mark_op[node.name] == 1:
                continue
            mod = False
            for out in node.output:
                if mark_var[out] == 1:
                    mark_op[node.name] = 1
                    mod = True
                    break
            if not mod:
                continue

            nb += 1
            for inp in node.input:
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes if mark_op[node.name] == 1]

    var_out = []
    for out in outputs:
        value_info = ValueInfoProto()
        value_info.name = out
        var_out.append(value_info)
    graph = make_graph(keep_nodes, model.graph.name, model.graph.input,
                       var_out, model.graph.initializer)
    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string

    if len(onnx_model.graph.input) != len(model.graph.input):
        raise RuntimeError("Input mismatch {} != {}".format(
                        len(onnx_model.input), len(model.input)))
    return onnx_model


def infer_outputs(op_type, inputs, outputs=None, initializer=None, **atts):
    """
    Infers outputs type and shapes given an ONNX operator.
    """
    if isinstance(op_type, str):
        required_outputs = []
        if outputs:
            for o in outputs:
                if hasattr(o, 'onnx_name'):
                    required_outputs.append(o.onnx_name)
                elif isinstance(o, str):
                    required_outputs.append(o)
                else:
                    raise TypeError("Unable to require output {}.".format(o))
        node = make_node(op_type, [i.onnx_name for i in inputs],
                         required_outputs, **atts)
        node = [node]
    elif hasattr(op_type, 'nodes'):
        node = op_type.nodes
    else:
        raise RuntimeError("Unable to build ONNX nodes from type {}.".format(
            type(op_type)))

    input_init = inputs.copy()
    if initializer:
        input_init.extend(initializer)
    onnx_inputs = []
    for input in input_init:
        if isinstance(input, Variable):
            onnx_type = input.type.to_onnx_type()
            tensor_type = onnx_type.tensor_type
            shape = [tensor_type.shape.dim[i].dim_value
                     for i in range(len(tensor_type.shape.dim))]
            inp = make_tensor_value_info(input.onnx_name,
                                         tensor_type.elem_type,
                                         tuple(shape))
            onnx_inputs.append(inp)
        elif isinstance(input, onnx.TensorProto):
            v = make_tensor_value_info(
                input.name, input.data_type.real,
                list(d for d in input.dims))
            onnx_inputs.append(v)
        elif isinstance(input, onnx.AttributeProto):
            value_info = ValueInfoProto()
            value_info.name = input.name
            onnx_type = onnx_proto.TypeProto()
            onnx_type.tensor_type.elem_type = input.type
            value_info.type.CopyFrom(onnx_type)
            onnx_inputs.append(value_info)
        else:
            onnx_inputs.append(input)

    graph = make_graph(node, 'infer_shapes',
                       onnx_inputs, [])
    original_model = make_model(graph, producer_name='skl2onnx')
    domains = {}
    for n in node:
        domains[n.domain] = max(domains.get(n.domain, 1),
                                getattr(n, 'op_version', 1))
    for i, (k, v) in enumerate(domains.items()):
        if i == 0 and len(original_model.opset_import) == 1:
            op_set = original_model.opset_import[0]
        else:
            op_set = original_model.opset_import.add()
        op_set.domain = k
        op_set.version = 10

    inferred_model = shape_inference.infer_shapes(original_model)
    shapes = Variable.from_pb(inferred_model.graph.value_info)
    if len(shapes) == 0:
        raise RuntimeError("Shape inference fails.\n"
                           "*Inputs*\n{}\n*Model*\n{}'".format(
                            onnx_inputs, original_model))
    return shapes
