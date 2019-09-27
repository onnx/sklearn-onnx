# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.data_types import FloatType, Int64Type, StringType, TensorType
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_concat(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    for i in range(len(operator.inputs)):
        if len(operator.inputs[i].type.shape) != 2:
            operator.outputs[0].type.shape = [None, None]
            return

    N = operator.inputs[0].type.shape[0]
    C = 0
    seen_types = []
    for i in operator.inputs:
        if isinstance(i.type, TensorType):
            if i.type.shape[1] is None:
                C = None
                break
            C += i.type.shape[1]
        elif isinstance(i.type, (Int64Type, FloatType, StringType)):
            C += 1
        else:
            C = None
            break
        nt = i.type.__class__.__name__
        if len(seen_types) == 0:
            seen_types.append(nt)
        elif nt != seen_types[0]:
            inps = "\n".join(str(v) for v in operator.inputs)
            raise RuntimeError("Columns must have the same type. "
                               "C++ backends do not support mixed types. "
                               "Inputs:\n"
                               + inps)
    operator.outputs[0].type.shape = [N, C]


register_shape_calculator('SklearnConcat', calculate_sklearn_concat)
register_shape_calculator('SklearnGenericUnivariateSelect',
                          calculate_sklearn_concat)
register_shape_calculator('SklearnMultiply', calculate_sklearn_concat)
register_shape_calculator('SklearnRFE', calculate_sklearn_concat)
register_shape_calculator('SklearnRFECV', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFdr', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFpr', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFromModel', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectFwe', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectKBest', calculate_sklearn_concat)
register_shape_calculator('SklearnSelectPercentile', calculate_sklearn_concat)
register_shape_calculator('SklearnVarianceThreshold', calculate_sklearn_concat)
