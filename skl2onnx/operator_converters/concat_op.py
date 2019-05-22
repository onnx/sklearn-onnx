# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._apply_operation import apply_concat
from ..common._registration import register_converter


def convert_sklearn_concat(scope, operator, container):
    apply_concat(scope, [s for s in operator.input_full_names],
                 operator.outputs[0].full_name, container, axis=1)


register_converter('SklearnConcat', convert_sklearn_concat)
