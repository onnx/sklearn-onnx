# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_cast, apply_concat
from ..common._registration import register_converter


def convert_sklearn_concat(scope, operator, container):
    concat_output_name = scope.get_unique_variable_name('concat_output')
    apply_concat(scope, [s for s in operator.input_full_names], operator.outputs[0].full_name, container, axis=1)
    # apply_concat(scope, [s for s in operator.input_full_names], concat_output_name, container, axis=1)
    # apply_cast(scope, concat_output_name, operator.outputs[0].full_name, container,
    #           to=onnx_proto.TensorProto.FLOAT)


register_converter('SklearnConcat', convert_sklearn_concat)
