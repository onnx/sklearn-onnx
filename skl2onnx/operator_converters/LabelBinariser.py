# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_reshape
from ..common._registration import register_converter
import numpy as np


def to_onnx_label_binariser(scope, operator, container):
    op = operator.raw_operator
    classes = op.classes_
    zeros_tensor = np.zeros((len(classes))).astype(np.int)
    unit_tensor = np.ones((len(classes))).astype(np.int)

    reshaped_input_name = scope.get_unique_variable_name('reshaped_input')
    shape_result_name = scope.get_unique_variable_name('shape_result')
    zeros_matrix_name = scope.get_unique_variable_name('zeros_matrix')
    unit_matrix_name = scope.get_unique_variable_name('unit_matrix')
    classes_tensor_name = scope.get_unique_variable_name('classes_tensor')
    equal_condition_tensor_name = scope.get_unique_variable_name(
                                                'equal_condition_tensor')
    zeros_tensor_name = scope.get_unique_variable_name('zero_tensor')
    unit_tensor_name = scope.get_unique_variable_name('unit_tensor')

    class_dtype = onnx_proto.TensorProto.STRING
    if np.issubdtype(op.classes_.dtype, np.signedinteger):
        class_dtype = onnx_proto.TensorProto.INT64
    else:
        classes = np.array([s.encode('utf-8') for s in classes])

    container.add_initializer(classes_tensor_name, class_dtype,
                              [len(classes)], classes)
    container.add_initializer(zeros_tensor_name, onnx_proto.TensorProto.INT64,
                              zeros_tensor.shape, zeros_tensor)
    container.add_initializer(unit_tensor_name, onnx_proto.TensorProto.INT64,
                              unit_tensor.shape, unit_tensor)

    apply_reshape(scope, operator.inputs[0].full_name, reshaped_input_name,
                  container, desired_shape=[-1, 1])
    container.add_node('Shape', reshaped_input_name, shape_result_name,
                       name=scope.get_unique_operator_name('shape'))
    container.add_node('Tile', [zeros_tensor_name, shape_result_name],
                       zeros_matrix_name, op_version=6,
                       name=scope.get_unique_operator_name('tile'))
    container.add_node('Tile', [unit_tensor_name, shape_result_name],
                       unit_matrix_name, op_version=6,
                       name=scope.get_unique_operator_name('tile'))
    container.add_node('Equal', [classes_tensor_name, reshaped_input_name],
                       equal_condition_tensor_name,
                       name=scope.get_unique_operator_name('equal'))
    container.add_node(
            'Where',
            [equal_condition_tensor_name, unit_matrix_name, zeros_matrix_name],
            operator.output_full_names,
            name=scope.get_unique_operator_name('where'), op_version=9)


register_converter('SklearnLabelBinarizer', to_onnx_label_binariser)
