# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._apply_operation import apply_identity
from ..common._registration import register_converter


def convert_sklearn_identity(scope, operator, container):
    apply_identity(
        scope, operator.inputs[0].full_name,
        operator.outputs[0].full_name, container,
        operator_name=scope.get_unique_operator_name('CIdentity'))


register_converter('SklearnIdentity', convert_sklearn_identity)
