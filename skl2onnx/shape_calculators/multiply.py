# SPDX-License-Identifier: Apache-2.0
import copy

from ..common._registration import register_shape_calculator


def calculate_sklearn_multiply(operator):
    for variable, output in zip(operator.inputs, operator.outputs):
        output.type = copy.copy(variable.type)


register_shape_calculator("SklearnMultiply", calculate_sklearn_multiply)
