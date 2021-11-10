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
        operator.outputs[0].full_name + '_cst')
    if classes.dtype in (np.int64, np.int32):
        container.add_initializer(
            name, onnx_proto.TensorProto.INT64, list(classes.shape),
            classes.tolist())
    else:
        clids = np.arange(len(classes), dtype=np.int64)
        container.add_initializer(
            name, onnx_proto.TensorProto.INT64, list(clids.shape),
            clids.tolist())
        namele = scope.get_unique_variable_name(
            operator.outputs[0].full_name + '_le')
        container.add_node(
            'LabelEncoder', name, namele, op_domain='ai.onnx.ml', op_version=2,
            default_string='0', keys_int64s=clids,
            values_strings=classes.tolist(),
            name=scope.get_unique_operator_name('class_labels_le'))
        name = namele

    container.add_node(
        'Identity', name, operator.outputs[0].full_name)


register_converter(
    'SklearnClassLabels', convert_sklearn_class_labels)
