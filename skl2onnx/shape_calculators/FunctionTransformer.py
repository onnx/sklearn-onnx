# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType


def calculate_sklearn_function_transformer_output_shapes(operator):
    '''
    This operator is used only to merge columns in a pipeline.
    Only id function is supported.
    '''
    if operator.raw_operator.func is not None:
        raise RuntimeError("FunctionTransformer is not supported unless the "
                           "transform function is None (= identity).")
    N = operator.inputs[0].type.shape[0]
    C = 0
    for variable in operator.inputs:
        if variable.type.shape[1] != 'None':
            C += variable.type.shape[1]
        else:
            C = 'None'
            break

    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator('SklearnFunctionTransformer',
                          calculate_sklearn_function_transformer_output_shapes)
