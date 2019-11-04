# Modified file from
# https://github.com/onnx/onnx/blob/master/onnx/helper.py.
import collections
import numbers

from onnx import (
    TensorProto, AttributeProto,
    NodeProto, GraphProto
)
from onnx.helper import (  # noqa
    make_tensor, make_model, make_graph, _to_bytes_or_false,
    make_tensor_value_info, ValueInfoProto
)
from onnx.numpy_helper import from_array  # noqa
from typing import (
    Text, Sequence, Any, Optional,
    List, cast
)
import numpy as np  # type: ignore


def make_node(
        op_type,  # type: Text
        inputs,  # type: Sequence[Text]
        outputs,  # type: Sequence[Text]
        name=None,  # type: Optional[Text]
        doc_string=None,  # type: Optional[Text]
        domain=None,  # type: Optional[Text]
        **kwargs  # type: Any
        ):  # type: (...) -> NodeProto
    """Construct a NodeProto.

    Arguments:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for NodeProto
        doc_string (string, default None): optional documentation
        string for NodeProto
        domain (string, default None): optional domain for NodeProto.
            If it's None, we will just use default domain (which is empty)
        **kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.
    """
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if kwargs:
        node.attribute.extend(
            make_attribute(key, value)
            for key, value in sorted(kwargs.items()))
    return node


def make_attribute(
        key,  # type: Text
        value,  # type: Any
        dtype=None,  # type: [np.float32, np.float64]
        domain='',  # type: Text
        doc_string=None  # type: Optional[Text]
        ):  # type: (...) -> AttributeProto
    """Makes an AttributeProto based on the value type."""
    def getshape(obj):
        if hasattr(obj, 'shape'):
            return obj.shape
        else:
            return (len(obj), )

    attr = AttributeProto()
    attr.name = key
    if doc_string:
        attr.doc_string = doc_string

    is_iterable = isinstance(value, collections.abc.Iterable)
    bytes_or_false = _to_bytes_or_false(value)

    use_float64 = dtype == np.float64 and domain not in ('', 'ai.onnx.ml')

    if isinstance(value, np.float32):
        attr.f = value
        attr.type = AttributeProto.FLOAT
    elif isinstance(value, (float, np.float64)):
        if use_float64:
            attr.type = AttributeProto.TENSOR
            attr.t.CopyFrom(
                make_tensor(
                    key, TensorProto.DOUBLE, (len(value), ), [value]))
        else:
            attr.f = value
            attr.type = AttributeProto.FLOAT
    elif isinstance(value, np.int32):
        attr.i = value
        attr.type = AttributeProto.INT
    elif isinstance(value, np.int64):
        attr.i = value
        attr.type = AttributeProto.INT
    elif isinstance(value, numbers.Integral):
        attr.i = value
        attr.type = AttributeProto.INT
    elif bytes_or_false:
        assert isinstance(bytes_or_false, bytes)
        attr.s = bytes_or_false
        attr.type = AttributeProto.STRING
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
        attr.type = AttributeProto.TENSOR
    elif isinstance(value, GraphProto):
        attr.g.CopyFrom(value)
        attr.type = AttributeProto.GRAPH
    # third, iterable cases
    elif is_iterable:
        byte_array = [_to_bytes_or_false(v) for v in value]
        if all(isinstance(v, np.float32) for v in value):
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
            # return make_attribute(
            #     key, doc_string=doc_string,
            #     value=make_tensor(
            #         key, TensorProto.FLOAT,
            #         getshape(value), value))
        elif all(isinstance(v, np.float64) for v in value):
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
            # return make_attribute(
            #    key, doc_string=doc_string,
            #     value=make_tensor(
            #         key, TensorProto.DOUBLE,
            #         getshape(value), value))
        elif all(isinstance(v, float) for v in value):
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
        elif all(isinstance(v, np.int32) for v in value):
            attr.ints.extend(int(v) for v in value)
            attr.type = AttributeProto.INTS
            # return make_attribute(
            #     key, doc_string=doc_string,
            #     value=make_tensor(
            #         key, TensorProto.INT32,
            #         getshape(value), value))
        elif all(isinstance(v, np.int64) for v in value):
            attr.ints.extend(int(v) for v in value)
            attr.type = AttributeProto.INTS
            # return make_attribute(
            #     key, doc_string=doc_string,
            #     value=make_tensor(
            #         key, TensorProto.INT64,
            #         getshape(value), value))
        elif all(isinstance(v, numbers.Integral) for v in value):
            # Turn np.int32/64 into Python built-in int.
            attr.ints.extend(int(v) for v in value)
            attr.type = AttributeProto.INTS
        elif all(byte_array):
            attr.strings.extend(cast(List[bytes], byte_array))
            attr.type = AttributeProto.STRINGS
        elif all(isinstance(v, TensorProto) for v in value):
            attr.tensors.extend(value)
            attr.type = AttributeProto.TENSORS
        elif all(isinstance(v, GraphProto) for v in value):
            attr.graphs.extend(value)
            attr.type = AttributeProto.GRAPHS
        else:
            raise ValueError(
                "You passed in an iterable attribute but I cannot figure out "
                "its applicable type, key='{}', type={}, types={}.".format(
                    key, type(value),
                    [type(_) for _, __ in zip(value, range(0, 5))]))
    else:
        raise ValueError(
            "Value '{}' is not valid attribute data type for attribute "
            "'{}'.".format(value, key))
    return attr


def get_attribute_value(attr):  # type: (AttributeProto) -> Any
    if attr.HasField('f'):
        return attr.f
    elif attr.HasField('i'):
        return attr.i
    elif attr.HasField('s'):
        return attr.s
    elif attr.HasField('t'):
        return attr.t
    elif attr.HasField('g'):
        return attr.g
    elif len(attr.floats):
        return list(attr.floats)
    elif len(attr.double):
        return list(attr.doubles)
    elif len(attr.ints):
        return list(attr.ints)
    elif len(attr.strings):
        return list(attr.strings)
    elif len(attr.tensors):
        return list(attr.tensors)
    elif len(attr.graphs):
        return list(attr.graphs)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr))
