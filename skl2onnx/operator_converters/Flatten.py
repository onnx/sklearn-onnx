# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_cast, apply_concat
from ..common._registration import register_converter


def convert_sklearn_flatten(scope, operator, container):
    concat_output_name = scope.get_unique_variable_name('flatten_output')
    name = scope.get_unique_operator_name('Flatten')
    container.add_node('Flatten', operator.inputs[0].full_name,
                       operator.outputs[0].full_name, name=name,
                       axis=1)


register_converter('SklearnFlatten', convert_sklearn_flatten)
