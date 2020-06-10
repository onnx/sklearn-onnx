# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._apply_operation import apply_cast
from ..common._registration import register_converter


def convert_sklearn_cast(scope, operator, container):
    inp = operator.inputs[0]
    exptype = operator.outputs[0]
    res = exptype.type.to_onnx_type()
    et = res.tensor_type.elem_type
    apply_cast(scope, inp.full_name, exptype.full_name,
               container, to=et)


register_converter('SklearnCastTransformer', convert_sklearn_cast)
register_converter('SklearnCast', convert_sklearn_cast)
