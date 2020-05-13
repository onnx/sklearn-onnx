# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator


def random_projection_shape_calculator(operator):
    """Shape calculator for PowerTransformer"""
    inputs = operator.inputs[0]
    op = operator.raw_operator
    n = inputs.type.shape[0]
    c = op.components_.shape[0]
    operator.outputs[0].type.shape = [n, c]


register_shape_calculator('SklearnGaussianRandomProjection',
                          random_projection_shape_calculator)
