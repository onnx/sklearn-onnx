# SPDX-License-Identifier: Apache-2.0
import numpy as np
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_class_labels(scope: Scope, operator: Operator,
                                 container: ModelComponentContainer):
    classes = np.array(operator.classes)

    name = scope.get_unique_variable_name(
        operator.outputs[0].full_name + 'cst')
    if classes.dtype in (np.int64, np.int32):
        container.add_initializer(
            name, onnx_proto.TensorProto.INT64, list(classes.shape),
            classes.tolist())
    elif classes.dtype in (np.str_, str, np.object):
        container.add_initializer(
            name, onnx_proto.TensorProto.STRING, list(classes.shape),
            classes.tolist())
    else:
        raise TypeError(
            "Unexpected type %r for class labels (labels=%r)"
            "." % (classes.dtype, operator.classes))

    container.add_node(
        'Identity', name, operator.outputs[0].full_name)


register_converter(
    'SklearnClassLabels', convert_sklearn_class_labels)
