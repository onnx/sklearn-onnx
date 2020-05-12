# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._apply_operation import apply_concat, apply_cast
from ..common._registration import register_converter


def convert_sklearn_concat(scope, operator, container):
    exptype = operator.outputs[0]
    new_inputs = []
    for inp in operator.inputs:
        if inp.type == exptype:
            new_inputs.append(inp.full_name)
            continue
        name = scope.get_unique_variable_name("{}_cast".format(inp.full_name))
        res = exptype.type.to_onnx_type()
        et = res.tensor_type.elem_type
        apply_cast(scope, inp.full_name, name, container, to=et)
        new_inputs.append(name)

    apply_concat(scope, new_inputs, operator.outputs[0].full_name,
                 container, axis=1)


register_converter('SklearnConcat', convert_sklearn_concat)
