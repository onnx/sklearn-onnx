# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx
from io import BytesIO
from onnx import helper


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


def enumerate_model_node_outputs(model):
    """
    Enumerates all the node of a model.
    """
    if not hasattr(model, "graph"):
        raise TypeError("*model* is not an *ONNX* model but {}".format(type(model)))
    for node in model.graph.node:
        for out in node.output:
            yield out


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
        raise NotImplementedError("Inputs cannot be changed yet.")
    if outputs is None:
        raise RuntimeError("outputs and inputs are None")
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
                if mark_var[inp] == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes if mark_op[node.name] == 1]
    
    var_out = []
    for out in outputs:
        value_info = helper.ValueInfoProto()
        value_info.name = out
        var_out.append(value_info)
    graph = helper.make_graph(keep_nodes, model.graph.name, model.graph.input, var_out,
                              model.graph.initializer)
    onnx_model = helper.make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    
    if len(onnx_model.graph.input) != len(model.graph.input):
        raise RuntimeError("Input mismatch {} != {}".format(len(onnx_model.input), len(model.input)))
    return onnx_model
