# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._apply_operation import apply_mul
from ..common._registration import register_converter
from ..proto import onnx_proto


def convert_sklearn_multiply(scope, operator, container):
    operand_name = scope.get_unique_variable_name(
        'operand')

    container.add_initializer(operand_name, onnx_proto.TensorProto.FLOAT,
                              [], [operator.operand])

    apply_mul(scope, [operator.inputs[0].full_name, operand_name],
              operator.outputs[0].full_name, container)


register_converter('SklearnMultiply', convert_sklearn_multiply)
